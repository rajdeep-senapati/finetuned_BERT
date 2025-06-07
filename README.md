# DistilBERT Fine-Tuned Sequence Classifier

## Overview

This repository contains a fine-tuned version of `distilbert-base-uncased` for sequence classification using the Hugging Face Transformers library. The model was adapted to a custom dataset. This README provides instructions for data loading, preprocessing, training, evaluation, and inference.

---

## Dataset

- **Name:** *[Specify your dataset name or description here]*
- **Format:** Text classification (e.g., sentiment, spam, phishing, etc.)
- **Source:** *[Add dataset source or link if public]*

---

## Installation
pip install transformers datasets evaluate
---

## Data Loading
from datasets import load_dataset

dataset_dict = load_dataset("your-dataset-name") # Replace with actual dataset name
print(dataset_dict)
---

## Model Setup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
model_path,
num_labels=2, # Change as per your task
)
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

training_args = TrainingArguments(
output_dir="./distilbert-finetuned",
learning_rate=2e-4,
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
num_train_epochs=10,
evaluation_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
)

trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_data["train"],
eval_dataset=tokenized_data["validation"],
tokenizer=tokenizer,
data_collator=data_collator,
)
trainer.train()

text

---

## Evaluation

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
roc_auc = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
predictions, labels = eval_pred
probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
positive_class_probs = probabilities[:, 1]
auc = np.round(roc_auc.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'], 3)
predicted_classes = np.argmax(predictions, axis=1)
acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'], 3)
return {"Accuracy": acc, "AUC": auc}

trainer.compute_metrics = compute_metrics
eval_results = trainer.evaluate()
print(eval_results)

text

---

## Inference

from transformers import pipeline

classifier = pipeline("text-classification", model="./distilbert-finetuned")
result = classifier("Your text here")
print(result)

text

---

## Results

| Metric    | Value   |
|-----------|---------|
| Accuracy  | (0.893) |
| ROC AUC   | (0.945) |

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [Hugging Face Evaluate](https://github.com/huggingface/evaluate)





