import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets import Dataset
from project_evaluate import read_file
from transformers import AutoTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


file_en_tr, file_de_tr = read_file('data/mini_train.labeled')
file_en_val, file_de_val = read_file('data/mini_val.labeled')

max_length = max([len(x.split()) for x in
                  file_en_tr + file_de_tr + file_en_val + file_de_val])
# max_length = 512


class T5ForConditionalGenerationWithLoss(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        return {"loss": loss, "logits": logits}


# Load the tokenizer and T5 model
tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=max_length)
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Prepare the training data: Combine the sentences into input-output pairs
train_data = []
for german, english in zip(file_de_tr, file_en_tr):
    input_text = f"translate German to English: {german.strip()}"
    output_text = english.strip()
    train_data.append((input_text, output_text))

# Preprocess the data
# Tokenize the input and output text and add the EOS token
train_encodings = []
for input_text, output_text in train_data:
    input_encoding = tokenizer.encode_plus(
        input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    output_encoding = tokenizer.encode_plus(
        output_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = input_encoding['input_ids'].squeeze()
    attention_mask = input_encoding['attention_mask'].squeeze()
    decoder_input_ids = output_encoding['input_ids'].squeeze()
    decoder_attention_mask = output_encoding['attention_mask'].squeeze()
    train_encodings.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask})

eval_data = []
for german, english in zip(file_de_val, file_en_val):
    input_text = f"translate German to English: {german.strip()}"
    output_text = english.strip()
    eval_data.append((input_text, output_text))

eval_encodings = []
for input_text, output_text in eval_data:
    input_encoding = tokenizer.encode_plus(
        input_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    output_encoding = tokenizer.encode_plus(
        output_text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = input_encoding['input_ids'].squeeze()
    attention_mask = input_encoding['attention_mask'].squeeze()
    decoder_input_ids = output_encoding['input_ids'].squeeze()
    decoder_attention_mask = output_encoding['attention_mask'].squeeze()
    eval_encodings.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask})


# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define the trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_dict(train_encodings),
)

trainer.train()

eval_results = trainer.predict(eval_encodings)
