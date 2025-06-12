import re
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.preprocess import preprocess

ds = load_dataset("gsm8k", "main", split="train", cache_dir="/workspace/modeldiffing/dataset")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})


tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

tokenized.set_format(type="torch",
                     columns=["input_ids", "attention_mask", "labels"])

# TODO get args from yaml
batch_size = 16
shuffle = True
loader = DataLoader(tokenized,
                    batch_size=batch_size,
                    shuffle=shuffle)

for batch in loader:
    # model generate 

    for epoch in epochs:
        # compute reward
        # compute loss
        # update 
        pass

    x = {
      "input_ids":      batch["input_ids"],
      "attention_mask": batch["attention_mask"],
    }
    y = batch["labels"]
    print(x)
    print("===")
    print(y)
    break
