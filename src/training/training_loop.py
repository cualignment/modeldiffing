import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils import make_preprocessor
from src.agents import Agent
from src.rewards import check_answer
from transformers import DataCollatorWithPadding

# TODO get args from yaml
batch_size = 4
epochs = 1
shuffle = True
gamma = 0.8
device = "cuda"
lr = 1e-4

agent = Agent("meta-llama/Llama-3.2-1B-Instruct", device)

preprocess = make_preprocessor(agent.tokenizer, max_length=128)

ds = load_dataset("gsm8k", "main", split="train", cache_dir=...)
tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

data_collator = DataCollatorWithPadding(agent.tokenizer)
loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)

optimizer_actor = torch.optim.Adam(agent.actor.parameters(), lr=1e-6)
optimizer_critic = torch.optim.Adam(agent.value_head.parameters(), lr=1e-4)

for batch in loader:
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
    gen_mask            = (new_ids != pad_id).long()                      # [B, T]
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

    critic_loss = ((values - G)**2 * gen_mask).sum() / gen_mask.sum()
    critic_loss.backward()
    optimizer_critic.step()
    print(f"Critic MSE: {critic_loss.item():.4f}")

    # === 6. Compute Advantage ===
    values     = values.detach()
    advantages = (G - values) * gen_mask
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # === 7. Train Actor ===
    agent.actor.train()
    outputs = agent.actor(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        return_dict=True
    )
    logits     = outputs.logits[:, P:, :]                          # [B, T, V]
    log_probs  = torch.log_softmax(logits, dim=-1)                 # [B, T, V]
    actions    = new_ids                                           # [B, T]
    actor_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]

    with torch.no_grad():
        ref_logits = agent.reference(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            return_dict=True
        ).logits[:, P:, :]                                         # [B, T, V]
        ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
        ref_log_probs = ref_log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # === 8. PPO Clipped Loss ===
    log_ratio = actor_log_probs - ref_log_probs     # [B, T]
    ratio     = torch.exp(log_ratio)
    clip_eps  = 0.2

    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    ppo_loss  = -torch.min(unclipped, clipped)
    ppo_loss  = (ppo_loss * gen_mask).sum() / gen_mask.sum()

    # === 9. Optional: Entropy bonus ===
    # entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1)  # [B, T]
    # entropy_bonus = (entropy * gen_mask).sum() / gen_mask.sum()
    # total_loss = ppo_loss - 0.01 * entropy_bonus

    optimizer_actor.zero_grad()
    total_loss.backward()
    optimizer_actor.step()
    print(f"PPO Loss: {ppo_loss.item():.4f}")