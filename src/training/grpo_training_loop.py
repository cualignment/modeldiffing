import re
import torch
import copy
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
            traj, _ = agent.generate({"input_ids": prompt_ids, "attention_mask": prompt_mask})
        completions = [t.split("{{#assistant}}")[3].strip() for t in traj]
        rewards = check_answer(completions, batch["labels"])
        all_rewards.extend(rewards)
    avg_reward = np.mean(all_rewards)
    wandb.log({"eval_avg_reward": avg_reward})
    print(f"[Eval] Average Reward over {num_batches} batches: {avg_reward:.4f}")
    return avg_reward

# === Config ===
batch_size = 2
shuffle = True
gamma = 0.99
device = "cuda"
lr_actor = 1e-5
clip_eps = 0.1
beta = 0.01
n_retrain = 4

wandb.init(
    project="grpo-llama3",
    entity="WanderingInductionHeads",
    name="grpo-llama3-run",
    config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "batch_size": batch_size,
        "lr_actor": lr_actor,
        "gamma": gamma,
    }
)

agent = Agent("meta-llama/Llama-3.2-1B-Instruct", init_critic=False, device=device)
agent.old_policy = copy.deepcopy(agent.actor).eval()

preprocess = make_preprocessor(agent.tokenizer, max_length=128)
ds = load_dataset("gsm8k", "main", split="train", cache_dir="data/gsm8k")
tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
data_collator = DataCollatorWithPadding(agent.tokenizer)
loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)
optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)

# evaluate_agent(agent, device)

n_rollouts_per_prompt = 4  # new setting: how many completions per sample

# === Multi-Rollout GRPO Training Loop (robust padding) ===
# Collect rollouts first, then pad every tensor to the *global* max length


for idx, batch in enumerate(loader):
    print(f"Batch [{idx+1}]/[{len(loader)}]")
    prompt_ids = batch["input_ids"].to(device)
    prompt_mask = batch["attention_mask"].to(device)
    B, P = prompt_ids.shape

    # Storage for rollouts of this batch
    roll_input_ids = []
    roll_attn_mask = []
    roll_actions   = []
    roll_advs      = []
    roll_gen_mask  = []
    roll_rewards   = []  # <-- store scalar rewards for logging

    pad_id = agent.tokenizer.pad_token_id

    for _ in range(n_rollouts_per_prompt):
        traj, new_seqs = agent.generate({"input_ids": prompt_ids, "attention_mask": prompt_mask})
        completions = [t.split("{{#assistant}}")[3].strip() for t in traj]
        rewards = check_answer(completions, batch["labels"])  # list[float] length = B
        roll_rewards.extend(rewards)  # accumulate
        lengths = [len(seq) for seq in new_seqs]
        max_new_len = max(lengths)

        # Build new_ids tensor (B, max_new_len)
        new_ids = torch.full((B, max_new_len), pad_id, dtype=prompt_ids.dtype, device=device)
        for i, seq in enumerate(new_seqs):
            new_ids[i, :len(seq)] = torch.tensor(seq, device=device)

        # Combine with prompt
        full_input = torch.cat([prompt_ids, new_ids], dim=1)
        gen_mask   = (new_ids != pad_id)
        full_attn  = torch.cat([prompt_mask, gen_mask], dim=1)

        # Advantage: normalized reward repeated per token
        raw_r = torch.tensor(rewards, dtype=torch.float32, device=device)
        norm_r = (raw_r - raw_r.mean()) / (raw_r.std() + 1e-8)
        adv_full = torch.zeros_like(new_ids, dtype=torch.float32)
        for i, L in enumerate(lengths):
            adv_full[i, :L] = norm_r[i]

        # Store rollout tensors
        roll_input_ids.append(full_input)
        roll_attn_mask.append(full_attn)
        roll_actions.append(new_ids)
        roll_advs.append(adv_full)
        roll_gen_mask.append(gen_mask)

    # === Global padding to max length across *all* collected rollouts ===
    def pad_list(t_list, pad_val):
        max_len = max(t.size(1) for t in t_list)
        return [
            (t if t.size(1) == max_len else torch.nn.functional.pad(t, (0, max_len - t.size(1)), value=pad_val))
            for t in t_list
        ]

    roll_input_ids = pad_list(roll_input_ids, pad_id)
    roll_attn_mask = pad_list(roll_attn_mask, 0)
    roll_actions   = pad_list(roll_actions, pad_id)
    roll_advs      = pad_list(roll_advs, 0.0)
    roll_gen_mask  = pad_list(roll_gen_mask, 0)

    # Stack/concat along batch dimension
    rollout_buffer = {
        "input_ids": torch.cat(roll_input_ids, dim=0),
        "attention_mask": torch.cat(roll_attn_mask, dim=0),
        "actions": torch.cat(roll_actions, dim=0),
        "advantages": torch.cat(roll_advs, dim=0),
        "gen_mask": torch.cat(roll_gen_mask, dim=0)
    }

    # === Policy Update ===
    for _ in range(n_retrain):
        agent.actor.train()
        outs = agent.actor(
            input_ids=rollout_buffer["input_ids"],
            attention_mask=rollout_buffer["attention_mask"],
            return_dict=True
        )
        logits = outs.logits[:, P:, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_lp = log_probs.gather(2, rollout_buffer["actions"].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            old_lp = torch.log_softmax(
                agent.old_policy(
                    input_ids=rollout_buffer["input_ids"],
                    attention_mask=rollout_buffer["attention_mask"],
                    return_dict=True
                ).logits[:, P:, :], dim=-1
            ).gather(2, rollout_buffer["actions"].unsqueeze(-1)).squeeze(-1)
            ref_lp = torch.log_softmax(
                agent.reference(
                    input_ids=rollout_buffer["input_ids"],
                    attention_mask=rollout_buffer["attention_mask"],
                    return_dict=True
                ).logits[:, P:, :], dim=-1
            )

        ratio = torch.exp(selected_lp - old_lp)
        unclipped = ratio * rollout_buffer["advantages"]
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * rollout_buffer["advantages"]
        ppo_loss = -torch.min(unclipped, clipped)

        kl_div = torch.sum(torch.exp(ref_lp) * (ref_lp - log_probs), dim=-1)

        mask = rollout_buffer["gen_mask"]
        loss = (ppo_loss[mask].mean() + beta * kl_div[mask].mean())

        optimizer_actor.zero_grad()
        loss.backward()
        optimizer_actor.step()

    # Update old policy weights once per outer batch
    agent.old_policy.load_state_dict(agent.actor.state_dict())

    # === Logging (scalar-safe) ===
    avg_reward_batch = float(np.mean(roll_rewards)) if roll_rewards else 0.0
    wandb.log({
        "grpo_loss": loss.item(),
        "ppo_loss": ppo_loss[mask].mean().item(),
        "kl_div": kl_div[mask].mean().item(),
        "adv_mean": rollout_buffer["advantages"][mask].mean().item(),
        "adv_std": rollout_buffer["advantages"][mask].std().item(),
        "ratio_mean": ratio[mask].mean().item(),
        "ratio_std": ratio[mask].std().item(),
        "avg_reward": avg_reward_batch
    })

    if (idx + 1) % 25 == 0:
        break

evaluate_agent(agent, device)
wandb.finish()