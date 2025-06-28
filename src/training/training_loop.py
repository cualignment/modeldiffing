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

agent = Agent("meta-llama/Llama-3.2-1B-Instruct", device)

preprocess = make_preprocessor(agent.tokenizer, max_length=128)

ds = load_dataset("gsm8k", "main", split="train", cache_dir="data/gsm8k")
tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

data_collator = DataCollatorWithPadding(agent.tokenizer)
loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)

optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)
optimizer_critic = torch.optim.Adam(agent.value_head.parameters(), lr=lr_critic)

evaluate_agent(agent, device)

for idx, batch in enumerate(loader):
    print(f"Batch [{idx+1}]/[{len(loader)}]")
    # === 1. Prepare inputs ===
    prompt_ids  = batch["input_ids"].to(device)      # [B, P]
    prompt_mask = batch["attention_mask"].to(device) # [B, P]
    B, P = prompt_ids.shape

    # === 2. Generate responses ===
    traj, new_seqs = agent.generate({
        "input_ids": prompt_ids,
        "attention_mask": prompt_mask
    })
    
    rewards  = check_answer([t.split("{{#assistant}}")[3].strip() for t in traj], batch["labels"])
    lengths  = [len(seq) for seq in new_seqs]
    max_len  = max(lengths)
    gamma    = 0.99

    wandb.log({"average_rewards": np.mean(rewards)})
    wandb.log({"average_lengths": np.mean(lengths)})
    wandb.log({"max_length": max_len})

    # === 3. Compute discounted returns G ===
    all_returns = []
    for r, L in zip(rewards, lengths):
        ret = [(gamma ** (L - t - 1)) * r for t in range(L)]
        ret += [0.0] * (max_len - L)
        all_returns.append(ret)

    G = torch.tensor(all_returns, dtype=torch.float32, device=device)  # [B, max_len]

    # === 4. Build generated input tensor ===
    pad_id = agent.tokenizer.pad_token_id
    new_ids = torch.full((B, max_len), pad_id, dtype=prompt_ids.dtype, device=device)
    for i, seq in enumerate(new_seqs):
        new_ids[i, :len(seq)] = torch.tensor(seq, device=device)

    full_input_ids      = torch.cat([prompt_ids, new_ids], dim=1)         # [B, P+T]
    gen_mask = (new_ids != pad_id)  # [B, T], dtype=bool
    full_attention_mask = torch.cat([prompt_mask, gen_mask], dim=1)       # [B, P+T]

    # === 5. Train Critic ===
    agent.critic.train()
    agent.value_head.train()
    optimizer_critic.zero_grad()

    hidden = agent.get_hidden_states({
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask
    })  # [B, P+T, H]

    gen_hidden = hidden[:, P:, :]           # [B, T, H]
    values     = agent.get_values(gen_hidden)  # [B, T]

    critic_loss = torch.nn.functional.mse_loss(
        values[gen_mask], G[gen_mask], reduction="mean"
    )
    critic_loss.backward()
    optimizer_critic.step()
    # print(f"Critic MSE: {critic_loss.item():.4f}")
    wandb.log({"critic_mse": critic_loss.item()})

    if idx <= warmup_steps:
        continue

    # === 6. Compute Advantage ===
    values     = values.detach()

    # Compute GAE estimate for advantages
    lambda_ = 0.95
    advantages = torch.zeros_like(values)

    for b in range(B):
        T = lengths[b]
        adv = 0.0
        for t in reversed(range(T)):
            # Get value estimates
            v_t     = values[b, t]
            v_next  = values[b, t+1] if t + 1 < T else 0.0

            # Temporal difference
            delta = G[b, t] - v_t

            # GAE recursion
            adv = delta + gamma * lambda_ * adv
            advantages[b, t] = adv

    # Optionally normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Masked to zero out padded tokens
    advantages[~gen_mask] = 0.0

    # === 8. Train Actor ===
    agent.actor.train()
    outputs = agent.actor(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        return_dict=True
    )
    logits     = outputs.logits[:, P:, :]                          # [B, T, V]
    log_probs  = torch.log_softmax(logits, dim=-1)                 # [B, T, V]
    actions    = new_ids                                           # [B, T]
    vocab_size = log_probs.shape[-1]

    actor_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]
    
    with torch.no_grad():
        ref_logits = agent.reference(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            return_dict=True
        ).logits[:, P:, :]                                         # [B, T, V]
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        ref_log_probs = ref_log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # === 9. PPO Clipped Loss ===
    log_ratio = actor_log_probs - ref_log_probs     # [B, T]
    ratio     = torch.exp(log_ratio)
    clip_eps  = 0.1

    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    
    loss = -torch.min(unclipped, clipped)
    if gen_mask.any():
        loss = loss[gen_mask].mean()
    else:
        continue

    # === 10. Optional: Entropy bonus ===
    entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1)  # [B, T]
    entropy_bonus = (entropy * gen_mask).sum() / gen_mask.sum()
    loss = loss - 0.01 * entropy_bonus
    
    optimizer_actor.zero_grad()
    loss.backward()
    optimizer_actor.step()

    wandb.log({
        "ppo_loss": loss.item(), 
        "entropy_bonus": entropy_bonus.item(),
        "critic_loss": critic_loss.item(),
        "pad_ratio": (~gen_mask).sum().item() / gen_mask.numel(),
        "adv_mean": advantages.mean().item(),
        "adv_std": advantages.std().item(),
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "ratio_mean": ratio[gen_mask].mean().item(),
        "ratio_std": ratio[gen_mask].std().item()
    })

evaluate_agent(agent, device)
wandb.finish()