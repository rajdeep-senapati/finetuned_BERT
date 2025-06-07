# BERT Fine-Tuned for Phishing Site Classification

## Overview

This project fine-tunes the `bert-base-uncased` model from Google on the [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification) dataset to classify websites as "Safe" or "Not Safe". The workflow includes data loading, preprocessing, model setup, training, evaluation, and inference.

---

## Table of Contents

- [Dataset](#dataset)
- [Model and Labels](#model-and-labels)
- [Installation](#installation)
- [Data Loading](#data-loading)
- [Model Setup](#model-setup)
- [Freezing Layers](#freezing-layers)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Dataset

- **Name:** [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)
- **Features:** `text`, `labels`
- **Splits:**
  - Train: 2100 samples
  - Validation: 450 samples
  - Test: 450 samples

---

## Model and Labels

- **Base Model:** `google-bert/bert-base-uncased`
- **Task:** Sequence Classification (Binary)
- **Labels:** `{0: "Safe", 1: "Not Safe"}`

---

## Installation
pip install transformers datasets evaluate

---

## Data Loading

from datasets import load_dataset

dataset_dict = load_dataset("shawhin/phishing-site-classification")
print(dataset_dict)

Output:
DatasetDict({
train: Dataset({'features': ['text', 'labels'], 'num_rows': 2100}),
validation: Dataset({'features': ['text', 'labels'], 'num_rows': 450}),
test: Dataset({'features': ['text', 'labels'], 'num_rows': 450})
})

---

## Model Setup

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}
model = AutoModelForSequenceClassification.from_pretrained(
model_path,
num_labels=2,
id2label=id2label,
label2id=label2id,
)

---

## Freezing Layers

By default, all layers are trainable. To freeze the base BERT model and only train the classifier and pooler layers:

Freeze base model parameters
for name, param in model.base_model.named_parameters():
param.requires_grad = False

Unfreeze pooler layers
for name, param in model.base_model.named_parameters():
if "pooler" in name:
param.requires_grad = True

Optionally, print which layers are trainable
for name, param in model.named_parameters():
print(name, param.requires_grad)

---

## Preprocessing

from transformers import DataCollatorWithPadding

def preprocess_function(examples):
return tokenizer(examples["text"], truncation=True)

tokenized_data = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

---

## Training

from transformers import TrainingArguments, Trainer

lr = 2e-4
batch_size = 8
num_epochs = 10

training_args = TrainingArguments(
output_dir="bert-phishing-classifier_teacher",
learning_rate=lr,
per_device_train_batch_size=batch_size,
per_device_eval_batch_size=batch_size,
num_train_epochs=num_epochs,
logging_strategy="epoch",
evaluation_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
)

trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_data["train"],
eval_dataset=tokenized_data["test"],
tokenizer=tokenizer,
data_collator=data_collator,
)
trainer.train()

---

## Evaluation

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
predictions, labels = eval_pred
probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
positive_class_probs = probabilities[:, 1]
auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)
predicted_classes = np.argmax(predictions, axis=1)
acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)
return {"Accuracy": acc, "AUC": auc}

Apply model to validation set
predictions = trainer.predict(tokenized_data["validation"])
logits = predictions.predictions
labels = predictions.label_ids
metrics = compute_metrics((logits, labels))
print(metrics)

Example output: {'Accuracy': 0.893, 'AUC': 0.945}

---

## Inference

To run inference on new examples:

text = "Example website text to classify."
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
pred_label = outputs.logits.argmax(dim=1).item()
print(f"Prediction: {id2label[pred_label]}")

---

## Results

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.893  |
| AUC      | 0.945  |

---

## Acknowledgements

- Code authored by Shaw Talebi
- Based on Hugging Face Transformers and Datasets libraries
- [shawhin/phishing-site-classification](https://huggingface.co/datasets/shawhin/phishing-site-classification)

---

*For further details, refer to the original notebook.*
