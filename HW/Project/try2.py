import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import evaluate
from tqdm import tqdm
import os
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import T5Tokenizer, T5ForConditionalGeneration
from project_evaluate import read_file, compute_metrics

file_en_tr, file_de_tr = read_file('data/train.labeled')    # TODO change to train without mini!
file_en_val, file_de_val = read_file('data/val.labeled')
output_dir = 'results'

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# the following 2 hyper parameters are task-specific
max_source_length = 512
max_target_length = 2000


# max_length = max([len(x.split()) for x in
#                   file_en_tr + file_de_tr + file_en_val + file_de_val])


####################### PREPARE DATA ########################

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)
print(f"Training on device: {device_str}")
model.to(device)


def get_dataset(input_sequences, target_sequences):
    # encode the inputs
    task_prefix = "translate German to English: "

    input_encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequences],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, input_attention_mask = input_encoding.input_ids, input_encoding.attention_mask

    # encode the targets
    target_encoding = tokenizer(
        target_sequences,
        padding="longest",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt",
    )
    target_ids, target_attention_mask = target_encoding.input_ids, target_encoding.attention_mask

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    target_ids[target_ids == tokenizer.pad_token_id] = -100

    dataset = {
        "input_ids": input_ids,
        "attention_mask": input_attention_mask,
        "decoder_input_ids": target_ids,
        "decoder_attention_mask": target_attention_mask
    }
    return Dataset.from_dict(dataset)


train_dataset = get_dataset(file_de_tr, file_en_tr)
eval_dataset = get_dataset(file_de_val, file_en_val)
data_collator = DataCollatorForSeq2Seq(tokenizer)

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


def compute_metrics_aux(predictions, labels):
    # predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_metrics(decoded_preds, decoded_labels)


####################### TRAIN ########################


epochs_n = 10
metric_name = 'bleu'
lr = 1e-4
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

for epoch in range(epochs_n):
    # ~~~~~~~~~~~~~~ train
    model.train()
    epoch_loss = 0
    epoch_bleu = 0

    for batch in tqdm(train_loader):
        input_ids, input_attention_mask, target_ids, _ = batch.values()
        input_ids, input_attention_mask, target_ids = \
            torch.tensor(input_ids).unsqueeze(0).to(device), \
            torch.tensor(input_attention_mask).unsqueeze(0).to(device), \
            torch.tensor(target_ids).unsqueeze(0).to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=input_attention_mask, labels=target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = model.generate(input_ids)
        bleu = compute_metrics_aux(predictions=preds, labels=target_ids.cpu())
        epoch_bleu += bleu

        input_ids, input_attention_mask = \
            input_ids.cpu(), input_attention_mask.cpu()

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_bleu = epoch_bleu / len(train_loader)

    print(f'\tTrain Loss: {avg_epoch_loss:.4f} | Train BLEU: {avg_epoch_bleu:7.3f}')

    # ~~~~~~~~~~~~~~ validate
    model.eval()
    val_epoch_loss = 0
    val_epoch_bleu = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, input_attention_mask, target_ids, _ = batch.values()
            input_ids, input_attention_mask, target_ids = \
                torch.tensor(input_ids).unsqueeze(0).to(device), \
                torch.tensor(input_attention_mask).unsqueeze(0).to(device), \
                torch.tensor(target_ids).unsqueeze(0).to(device),

            val_outputs = model(input_ids=input_ids, attention_mask=input_attention_mask, labels=target_ids)
            val_loss = val_outputs.loss

            val_epoch_loss += val_loss.item()
            val_preds = model.generate(input_ids)
            val_bleu = compute_metrics_aux(predictions=val_preds, labels=target_ids.cpu())
            val_epoch_bleu += val_bleu

            input_ids, input_attention_mask = \
                input_ids.cpu(), input_attention_mask.cpu()

    avg_val_epoch_loss = val_epoch_loss / len(eval_loader)
    avg_val_epoch_bleu = val_epoch_bleu / len(eval_loader)
    print(f'\t\tVal. Loss: {avg_val_epoch_loss:.4f}  |  Val. BLEU: {avg_val_epoch_bleu:7.3f}')

path = os.path.join(output_dir, "model_files")
model.save_pretrained(path)
tokenizer.save_pretrained(path)







# args = TrainingArguments(
#     evaluation_strategy="epoch",
#     output_dir=output_dir,
#     learning_rate=2e-5,
#     per_device_train_batch_size=10,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     load_best_model_at_end=False,
#     metric_for_best_model=metric_name,
#     logging_dir='logs',
# )
#
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics_aux,
# )
#
# trainer.train()
