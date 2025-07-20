# finetune.py
import json
import random
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data(path):
    with open(path, 'r') as f:
        lines = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(lines)

def label_to_id(label):
    return 1 if label.lower() == "positive" else 0

def main(args):
    data = load_data(args.data)
    data = data.map(lambda x: {'label': label_to_id(x['label'])})

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    data = data.map(tokenize, batched=True)
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        learning_rate=args.lr,
        logging_dir="./logs",
        save_strategy="no",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
    )

    trainer.train()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training data.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    main(args)
