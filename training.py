import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
from datasets import Dataset

device="cuda"

#aug_txt = {'text': [prompts_train['source_text'][text_train['prompt_id'][i]] + prompts_train['instructions'][text_train['prompt_id'][i]] #+ text_train['text'][i] for i in range(len(text_train['text']))],
#          'generated': [text_train['generated'][i] for i in range(len(text_train['text']))]}


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokenizer.pad_token = tokenizer.eos_token

aug_txt_train = pd.read_csv("C:/llm-det/llm-detect-ai-generated-text/aug_txt_train.csv")

aug_txt_train.rename(columns={'generated': 'label'}, inplace=True)

text_data = Dataset.from_pandas(aug_txt_train)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

tokenized_txt = text_data.map(preprocess_function, batched=True)

tokenized_data = tokenized_txt.train_test_split()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1", num_labels = 2, id2label=id2label, label2id=label2id)

roc_auc = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return roc_auc.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="C:/llm-det/base_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()