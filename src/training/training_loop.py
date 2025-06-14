import re
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils import make_preprocessor
from src.agents import Agent
from src.rewards import check_answer
from transformers import DataCollatorWithPadding

agent = Agent("meta-llama/Llama-3.2-1B-Instruct", "cuda")

preprocess = make_preprocessor(agent.tokenizer, max_length=128)

ds = load_dataset("gsm8k", "main", split="train", cache_dir=...)
tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids","attention_mask","labels"])



# TODO get args from yaml
batch_size = 10
epochs = 1
shuffle = True
gamma = 0.8

data_collator = DataCollatorWithPadding(agent.tokenizer)
loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator, shuffle=shuffle)

for batch in loader:

    x = {
      "input_ids":      batch["input_ids"],
      "attention_mask": batch["attention_mask"],
    }
  
    y = batch["labels"]

    # generate trajectories   
    traj, tokens = agent.generate(x)

    # remove prompts
    traj = [t.split("{{#assistant}}")[3].strip() for t in traj]

    # compute rewards
    rewards = check_answer(traj, batch["labels"])
    print(rewards)

    # compute discouted returns G_t


    # compute advantages

    # compute losses

    # update gradients 
    
    for epoch in range(epochs):
        # compute reward
        # compute loss
        # update 
        pass

    break
