# Instructions for Setup of Fine-Tuning GPT-2 on SST-2 Dataset

## Overview 

This project fine-tunes a pre-trained 'GPT2' model on binary sentiment classification using the SST-2 (Stanford Sentiment Treebank) dataset via the Hugging Face🤗 Trainer API.

## Setup (currently only on Google Colab)

1. Copy the Colab notebook (need to do this)

2. Execute the notebook cells to start training

# Benchmark Summary: Training

## Test 1

### Training Details

- Dataset: SST-2 (Stanford Sentiment Treebank)
- Model: 'GPT2'
- Batch Size: 8% of Dataset
- Ammount of Epochs: 1
- Metrics tracked: Training loss, Validation Accuracy, GPU memory usage and Training time per epoch

## Performance Metrics

| Epoch  | Training loss  | Validation Loss  | Accuracy |
| :------| :------------- | :--------------- | :------- |
| 1      | 1.886900       | 1.872230         | 40%      |

- Training Time: 1 minutes and 7 seconds

## Resource Usage

- GPU Type: Tesla T4
- GPU Memory Used: 13.2/15 GB
- DISK Memory Used: 44.4/112.6 GB
- RAM Memory Used: 5.8/12.7 GB
- GPU Utilization: 88%
- Driver/CUDA version: 550.54.15

## Data Visualization

![W B Chart 6_8_2025, 2_46_21 PM](https://github.com/user-attachments/assets/05e77f45-2474-48bf-949b-2c62d34506fc)

- Using 1 percent of the total dataset resulted in 100% accuraccy. As the program executes the training loss seems to be lowering slowly.

![W B Chart 6_8_2025, 2_47_53 PM](https://github.com/user-attachments/assets/da081430-e3db-49dc-b348-109ac3f8ebe7)

- There seems to be progess in learning trending on improvement.

- ## Test 2

### Training Details

- Dataset: SST-2 (Stanford Sentiment Treebank)
- Model: 'GPT2'
- Batch Size: 1% of Dataset
- Ammount of Epochs: 1
- Metrics tracked: Training loss, Validation Accuracy, GPU memory usage and Training time per epoch

## Performance Metrics

| Epoch  | Training loss  | Validation Loss  | Accuracy |
| :------| :------------- | :--------------- | :------- |
| 1      | 0.864000       | 0.009613         | 100%     |

- Training Time: 55 seconds

## Resource Usage

- GPU Type: Tesla T4
- GPU Memory Used: 11.1/15 GB
- Disk Memory Used: 48.8/112.6 GB
- RAM Memory Used: 5.8/12.7 GB
- GPU Utilization: 74%
- Driver/CUDA version: 550.54.15

## Data Visualization

![W B Chart 6_9_2025, 2_11_42 PM](https://github.com/user-attachments/assets/dde99eff-883b-46e4-9c15-51e4283eb3d4)

- Using 1 percent of the total dataset resulted in 100% accuraccy. The program executes the training loss seems to be lowering more rapidly.

![W B Chart 6_9_2025, 2_12_15 PM](https://github.com/user-attachments/assets/1f98f168-b6e7-4410-8db5-308bb9a4214b)


- There seems to be progess in learning trending on improvement, equal to previous testing.

- ## Test 3

### Training Details

- Dataset: SST-2 (Stanford Sentiment Treebank)
- Model: 'GPT2'
- Batch Size: 14% of Dataset
- Ammount of Epochs: 1
- Metrics tracked: Training loss, Validation Accuracy, GPU memory usage and Training time per epoch

## Performance Metrics

| Epoch  | Training loss  | Validation Loss  | Accuracy |
| :------| :------------- | :--------------- | :------- |
| 1      | 1.743400	      | 1.507292         | 51%      |

- Training Time: 55 seconds

## Resource Usage

- GPU Type: Tesla T4
- GPU Memory Used: 11.2/15 GB
- Disk Memory Used: 49.8/112.6 GB
- RAM Memory Used: 4.2/12.7 GB
- GPU Utilization: 74.67%
- Driver/CUDA version: 550.54.15

## Data Visualization

![W B Chart 6_9_2025, 2_38_18 PM](https://github.com/user-attachments/assets/ecc67472-5a59-4fec-995c-76cdb5e377b6)

- Using 1 percent of the total dataset resulted in 51% accuraccy. The program executes the training loss seems to be lowering less.

![W B Chart 6_9_2025, 2_39_26 PM](https://github.com/user-attachments/assets/cc28eb44-86b9-4f47-92d4-431347c91339)

- There seems to be progess in learning trending on improvement, equal to previous testing.
