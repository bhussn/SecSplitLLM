# Instructions for Setup of Fine-Tune BERT on SST-2 (Sentiment Classification)

## Overview
This project fine-tunes 'bert-base-uncased' using the SST-2 dataste via the Hugging Face's Trainer API

## Setup
1. Open the Google Colab notebook
2. Execute the notebook cells

# Benchmark Summary
## Training Details
  - Dataset: SST-2 (Stanford Sentiment Treebank)
  - Model: `bert-base-uncased` 
  - Batch size: 8
  - Epochs: 1
  - Metrics tracked: Training loss, Validation accuracy, GPU memory usage, Training       time
  - Logs to Weights & Biases

## Performance Metrics
| Epoch | Training Loss | Validation Loss | Accuracy |
| ----- | ------------- | --------------- | -------- |
| 1     | 0.348400      | 0.477652        | 86.21%   |

- Training time: 659.60 seconds (~11 minutes)

## Resource Usage
- GPU Type: Tesla T4
- GPU Memory Used: 5,892 MiB out of 15,360 MiB
- GPU Utilization: 91%
- Driver/CUDA Version: 550.54.15 / 12.4

## Observations and Interpretations of graphs from wandb
![image](https://github.com/user-attachments/assets/7e00d3c6-b32a-48eb-85ab-00aec659ff13)
![image](https://github.com/user-attachments/assets/7d5d3a03-cc73-403e-a775-0a0f48c75baa)


- The overall trend is that the loss is decreasing, meaning the model is learning and improving on the training data.
- There are a lot of fluctuations, which is normal for training deep learning models. The spiky pattern can be due to a high learning rate or small batch size. However, is it possibly overfitting?
- The loss range is starting around 0.7 and decreasing to around 0.3 or below. This shows good learning significant learning progress.
- Overall, the model is learning and loss is trending downward.
