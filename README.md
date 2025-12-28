# Sentiment Analysis on SST-5 Dataset

## Problem Statement

This project addresses the task of fine-grained sentiment classification on the Stanford Sentiment Treebank (SST-5) dataset. The objective is to classify movie review sentences into five sentiment classes:
- 0: Very Negative
- 1: Negative
- 2: Neutral
- 3: Positive
- 4: Very Positive

## Dataset

The project uses the SST-5 dataset from Hugging Face, which contains 8,544 training samples, 1,101 validation samples, and 2,210 test samples. 
The dataset is available at: https://huggingface.co/datasets/SetFit/sst5

## Project Structure

### Baseline_model/
Contains traditional machine learning baseline models for sentiment classification:
- `train_svm.py`, `train_xgb.py`, `train_random_forest.py`, `train_logistic_regression.py`, `train_KNN.py`, `train_Naive_Bayes.py`: Training scripts for baseline models
- `evaluated_model_test.py`, `evaluated_model_val.py`: Evaluation scripts for validation and test sets
- `comparison_test/`, `comparison_val/`: Directories containing performance reports and comparison results for all baseline models

### bert-lstm/
Implements a BERT-LSTM hybrid architecture:
- `notebooks/`: Jupyter notebooks for data exploration and model prototyping
- `src/`: Source code including configuration, dataset handling, model definition, training pipeline, and utility functions
- `scripts/`: Shell scripts for training execution

### finetuned-bert/
Implements a fine-tuned BERT model specifically for SST-5 sentiment classification using Cross Entropy loss at first and Focal Loss for improved handling of class imbalance.

- `bert-sst5_cross_entropy.ipynb`: Training notebook for Bert with CE loss
- `bert-sst5_focal.ipynb`: Training notebook for Bert with Focal loss

### electra/
Implements an ELECTRA-based model with comprehensive training pipeline:
- `electra-sst5_cross_entropy.ipynb`: Complete training notebook for Electra with CE loss
- `electra-sst5_focal.ipynb`: Training notebook for Electra with Focal loss

## Models Overview

1. **Baseline Models**: Traditional approaches (SVM, XGBoost, Random Forest, Logistic Regression, KNN, Naive Bayes) using CountVectorizer for feature extraction

2. **BERT-LSTM**: Hybrid deep learning approach combining BERT embeddings with LSTM layers for sequential processing

3. **Fine-tuned BERT**: BERT model fine-tuned with Focal Loss to handle class imbalance in the SST-5 dataset

4. **ELECTRA**: Efficient pre-trained transformer model specifically adapted for sentiment classification with extensive performance analysis

## Key Features

- Comprehensive data preprocessing and analysis
- Multiple model architectures for comparison
- Detailed evaluation metrics (Accuracy, F1-score, Precision, Recall)
- Confusion matrices and error analysis
- Sentence length-based error analysis for understanding model behavior
- Training history visualization and logging