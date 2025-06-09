# # # # # # # # # # # # # # # # # # # # # # # 
#              SecSplitLLM                  #
#           train_baseline.py               # 
#                                           #
#   *Use this file to confirm that the      #
#    setup (libraries, GPU/CPU, logging)    #
#    is functional.                         #
#   *Provides a reference point for         #
#    evaluating the impact of DP, SMPC,     #
#    and split learning in the hybrid       #
#    framework.                             #
#   
#    Usage: python train_baseline.py 
#           --sample_fraction [NUM]
#           --learning_rate [NUM]
#           --batch_size [NUM]    
#           --epochs  [NUM]    
#           --output_dir [NAME]
#
#   Example usage: 
#       python train_baseline.py --epochs 2 
#       --batch_size 8 --learning_rate 2e-5
#             
# # # # # # # # # # # # # # # # # # # # # # # 

import argparse
import os
import time
import torch
import numpy as np
import wandb
import evaluate
import csv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# This callback helps us keep track of important stuff each epoch:
# time taken, GPU memory, loss, and accuracy.
# Saves it in a CSV and sends it to W&B for easy tracking.
class SimpleLoggerCallback(TrainerCallback):
    def __init__(self, log_path="training_log.csv"):
        self.log_path = log_path
        self.epoch_start_time = None
        # Write CSV header right away so we don't lose track
        with open(self.log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "duration_sec", "gpu_mem_allocated_gb", "train_loss", "eval_accuracy"])

    # Mark the time when an epoch starts — need this to calculate how long it took later.
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    # Grab the training loss whenever we get logs from Trainer
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_train_loss = logs["loss"]

    # When epoch ends, calculate duration, GPU memory used, and log everything
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        duration = time.time() - self.epoch_start_time
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        train_loss = getattr(self, "last_train_loss", None)
        eval_acc = logs.get("eval_accuracy") if logs else None

        # Print some quick info to console so we can see progress live
        print(f"Epoch {int(state.epoch)} done in {duration:.2f}s | GPU mem: {gpu_mem:.2f} GB | Loss: {train_loss:.4f} | Val Acc: {eval_acc}")

        # Log all the important metrics to WandB
        wandb.log({
            f"epoch_{int(state.epoch)}_duration_sec": duration,
            f"epoch_{int(state.epoch)}_gpu_mem_allocated_GB": gpu_mem,
            "train_loss": train_loss,
            "eval_accuracy": eval_acc
        })

        # Append the epoch stats to our CSV log file for offline use
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([int(state.epoch), duration, gpu_mem, train_loss, eval_acc])

# Simple metric calculation — just accuracy here for SST-2
def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Tokenize input sentences - pad and truncate to max length 128 for consistency
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

def main():
    # Argument parser so you can easily tweak settings from command line
    parser = argparse.ArgumentParser(description="Fine-tune GPT2 on SST-2 dataset for sentiment classification")
    
    # Only expose these parameters for user tweaking
    parser.add_argument("--sample_fraction", type=float, default=0.001, help="Fraction of dataset to use for faster experiments")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save models and logs")

    args = parser.parse_args()

    # Fixed parameters — locked in for GPT2 + SST-2 setup
    model_name = "gpt2"
    dataset_name = "glue"
    dataset_config = "sst2"
    run_name = "gpt2-sst2"

    print("Logging into WandB so we can track this run...")
    wandb.login()
    os.environ["WANDB_PROJECT"] = run_name

    print(f"Loading dataset: {dataset_name} with config {dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    # If you want to speed up experimentation, just use a fraction of the data
    if args.sample_fraction < 1.0:
        for split in ["train", "validation"]:
            dataset[split] = dataset[split].shuffle(seed=42).select(range(int(args.sample_fraction * len(dataset[split]))))

    print(f"Loading tokenizer and model '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT2 doesn't have a padding token by default, so set it to eos token for padding
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Tokenizing dataset... this might take a minute")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Setup training parameters — tweak these to your liking
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to="wandb",
        run_name=run_name,
    )

    # Put everything together for Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[SimpleLoggerCallback(log_path=os.path.join(args.output_dir, "training_log.csv"))],
    )

    # print("Starting training — keep an eye on your GPU and wandb dashboard!")
    trainer.train()

    print("Training finished. Running final evaluation...")
    results = trainer.evaluate()
    print("Evaluation results:", results)

    print(f"Saving the fine-tuned model and tokenizer in {args.output_dir}")
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

# This runs the main function when you execute the script directly
if __name__ == "__main__":
    main()
