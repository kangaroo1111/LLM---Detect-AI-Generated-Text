# Import necessary libraries
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from datasets import Dataset

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer for the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Set the padding token to be the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load the training data from a CSV file
aug_txt_train = pd.read_csv("C:/llm-det/llm-detect-ai-generated-text/aug_txt_train.csv")

# Rename the 'generated' column to 'label' to match expected format
aug_txt_train.rename(columns={'generated': 'label'}, inplace=True)

# Convert the Pandas DataFrame to a Hugging Face Dataset
text_data = Dataset.from_pandas(aug_txt_train)

# Define a preprocessing function for tokenization
def preprocess_function(examples):
    # Tokenize the text with truncation
    return tokenizer(examples['text'], truncation=True)

# Apply the preprocessing function to the dataset
tokenized_txt = text_data.map(preprocess_function, batched=True)

# Split the dataset into training and testing sets (90% train, 10% test)
tokenized_data = tokenized_txt.train_test_split(test_size=0.1)

# Create a data collator that will dynamically pad the inputs
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define label mappings for the model
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Load the pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Move the model to the specified device (GPU or CPU)
model.to(device)

# Load the ROC AUC metric for evaluation
roc_auc = evaluate.load("roc_auc")

# Define a function to compute evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Convert predictions to probabilities using softmax
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)
    # Get probabilities for the positive class
    preds = probs[:, 1].numpy()
    # Compute ROC AUC score
    return roc_auc.compute(prediction_scores=preds, references=labels)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="C:/llm-det/base_model",     # Directory to save model checkpoints
    learning_rate=2e-5,                     # Learning rate
    per_device_train_batch_size=16,         # Batch size per device during training
    per_device_eval_batch_size=16,          # Batch size for evaluation
    num_train_epochs=2,                     # Total number of training epochs
    weight_decay=0.01,                      # Strength of weight decay
    evaluation_strategy="epoch",            # Evaluation is done at the end of each epoch
    save_strategy="epoch",                  # Model is saved at the end of each epoch
    load_best_model_at_end=True,            # Load the best model at the end of training
    push_to_hub=False,                      # Do not push the model to the Hugging Face Hub
    logging_dir="C:/llm-det/logs",          # Directory for storing logs
    logging_steps=10,                       # Log every 10 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                            # The pre-trained model
    args=training_args,                     # Training arguments
    train_dataset=tokenized_data["train"],  # Training dataset
    eval_dataset=tokenized_data["test"],    # Evaluation dataset
    tokenizer=tokenizer,                    # Tokenizer for data collator
    data_collator=data_collator,            # Data collator for dynamic padding
    compute_metrics=compute_metrics,        # Function to compute evaluation metrics
)

# Start the training process
trainer.train()
