import re
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils import make_preprocessor
from src.agents import Agent
from src.rewards import check_answer
from transformers import DataCollatorWithPadding

def evaluate_agent(agent, device, split="train", num_batches=25, batch_size=8):
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from src.utils import make_preprocessor
    from src.rewards import check_answer
    import torch
    import numpy as np

    agent.actor.eval()

    preprocess = make_preprocessor(agent.tokenizer, max_length=128)
    ds = load_dataset("gsm8k", "main", split=split, cache_dir="data/gsm8k")
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(agent.tokenizer)
    loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)

    all_rewards = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        prompt_ids  = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            traj, _ = agent.generate({
                "input_ids": prompt_ids,
                "attention_mask": prompt_mask
            })

        completions = [t.split("{{#assistant}}")[3].strip() for t in traj]
        rewards = check_answer(completions, batch["labels"])
        all_rewards.extend(rewards)

    avg_reward = np.mean(all_rewards)
    wandb.log({"eval_avg_reward": avg_reward})
    print(f"[Eval] Average Reward over {num_batches} batches: {avg_reward:.4f}")
    return avg_reward

# TODO get args from yaml
batch_size = 8
epochs = 1
shuffle = True
gamma = 0.99
device = "cuda"
lr_actor = 1e-5
lr_critic = 1e-3
warmup_steps = 50

wandb.init(
    project="ppo-llama3",
    entity="WanderingInductionHeads",
    name="ppo-llama3-run",
    config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "batch_size": batch_size,
        "epochs": epochs,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "gamma": gamma
    }
)

agent = Agent("meta-llama/Llama-3.2-1B-Instruct", init_critic=True, device=device)

preprocess = make_preprocessor(agent.tokenizer, max_length=128)

ds = load_dataset("gsm8k", "main", split="train", cache_dir="data/gsm8k")
tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

data_collator = DataCollatorWithPadding(agent.tokenizer)
loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)

optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)
optimizer_critic = torch.optim.Adam(agent.value_head.parameters(), lr=lr_critic)

evaluate_agent(agent, device)

n_retrain = 4  # Number of times to reuse the rollout buffer per batch

for idx, batch in enumerate(loader):
    print(f"Batch [{idx+1}]/[{len(loader)}]")

    prompt_ids = batch["input_ids"].to(device)
    prompt_mask = batch["attention_mask"].to(device)
    B, P = prompt_ids.shape

    # === Generate responses ===
    traj, new_seqs = agent.generate({
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask
    })

    completions = [t.split("{{#assistant}}")[3].strip() for t in traj]
    rewards = check_answer(completions, batch["labels"])
    lengths = [len(seq) for seq in new_seqs]
    max_len = max(lengths)

    wandb.log({"average_rewards": np.mean(rewards)})

    # === Compute discounted returns ===
    all_returns = []
    for r, L in zip(rewards, lengths):
        ret = [(gamma ** (L - t - 1)) * r for t in range(L)]
        ret += [0.0] * (max_len - L)
        all_returns.append(ret)
    G = torch.tensor(all_returns, dtype=torch.float32, device=device)

    # === Prepare inputs ===
    pad_id = agent.tokenizer.pad_token_id
    new_ids = torch.full((B, max_len), pad_id, dtype=prompt_ids.dtype, device=device)
    for i, seq in enumerate(new_seqs):
        new_ids[i, :len(seq)] = torch.tensor(seq, device=device)

    full_input_ids = torch.cat([prompt_ids, new_ids], dim=1)
    gen_mask = (new_ids != pad_id)
    full_attention_mask = torch.cat([prompt_mask, gen_mask], dim=1)

    # === Run critic once and store values ===
    with torch.no_grad():
        hidden = agent.get_hidden_states({
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask
        })
        gen_hidden = hidden[:, P:, :]
        values = agent.get_values(gen_hidden)

    # === Compute GAE advantages ===
    advantages = torch.zeros_like(values)
    lambda_ = 0.95
    for b in range(B):
        adv = 0.0
        T = lengths[b]
        for t in reversed(range(T)):
            delta = G[b, t] - values[b, t]
            adv = delta + gamma * lambda_ * adv
            advantages[b, t] = adv
    advantages[~gen_mask] = 0.0
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # === Rollout buffer ===
    rollout_buffer = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "actions": new_ids,
        "advantages": advantages,
        "returns": G,
        "gen_mask": gen_mask,
        "values": values
    }

    for rollout_idx in range(n_retrain):
        # === Critic update ===
        agent.critic.train()
        agent.value_head.train()
        optimizer_critic.zero_grad()

        hidden = agent.get_hidden_states({
            "input_ids": rollout_buffer["input_ids"],
            "attention_mask": rollout_buffer["attention_mask"]
        })
        gen_hidden = hidden[:, P:, :]
        values = agent.get_values(gen_hidden)

        critic_loss = torch.nn.functional.mse_loss(
            values[rollout_buffer["gen_mask"]],
            rollout_buffer["returns"][rollout_buffer["gen_mask"]],
            reduction="mean"
        )
        critic_loss.backward()
        optimizer_critic.step()

        # === Actor update ===
        agent.actor.train()
        outputs = agent.actor(
            input_ids=rollout_buffer["input_ids"],
            attention_mask=rollout_buffer["attention_mask"],
            return_dict=True
        )
        logits = outputs.logits[:, P:, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        actor_log_probs = log_probs.gather(2, rollout_buffer["actions"].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_logits = agent.reference(
                input_ids=rollout_buffer["input_ids"],
                attention_mask=rollout_buffer["attention_mask"],
                return_dict=True
            ).logits[:, P:, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_log_probs = ref_log_probs.gather(2, rollout_buffer["actions"].unsqueeze(-1)).squeeze(-1)

        log_ratio = actor_log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)
        clip_eps = 0.1
        unclipped = ratio * rollout_buffer["advantages"]
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * rollout_buffer["advantages"]
        loss = -torch.min(unclipped, clipped)

        if rollout_buffer["gen_mask"].any():
            loss = loss[rollout_buffer["gen_mask"]].mean()
        else:
            continue

        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1)
        entropy_bonus = (entropy * rollout_buffer["gen_mask"]).sum() / rollout_buffer["gen_mask"].sum()
        loss = loss - 0.01 * entropy_bonus

        optimizer_actor.zero_grad()
        loss.backward()
        optimizer_actor.step()

        wandb.log({
            "ppo_loss": loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_bonus": entropy_bonus.item(),
            "adv_mean": rollout_buffer["advantages"].mean().item(),
            "adv_std": rollout_buffer["advantages"].std().item(),
            "ratio_mean": ratio[rollout_buffer["gen_mask"]].mean().item(),
            "ratio_std": ratio[rollout_buffer["gen_mask"]].std().item()
        })

    if idx + 1 >= 100:
        break

evaluate_agent(agent, device)
wandb.finish()