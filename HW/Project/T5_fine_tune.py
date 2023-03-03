import transformers
from datasets import load_dataset, load_metric, Dataset
from project_evaluate import read_file, compute_metrics, postprocess_text
from transformers import AutoTokenizer, T5Tokenizer, MT5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np


class T5FineTune:
    def __init__(self):
        # ~~~~ params
        self.prefix = "translate German to English: "
        self.max_input_length = self.max_target_length = 128
        self.batch_size = 1
        self.epochs_n = 10
        self.source_lang = "de"
        self.target_lang = "en"

        # ~~~~ model stuff
        self.model_name = "t5-base"
        # self.model_name = "google/mt5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max=self.max_input_length,
            model_max_length=self.max_input_length
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.metric_name = "sacrebleu"
        self.metric = load_metric(self.metric_name)

        # ~~~~ paths
        self.train_path = 'data/train.labeled'
        self.eval_path = 'data/val.labeled'
        # self.train_path = 'data/mini_train.labeled'
        # self.eval_path = 'data/mini_train.labeled'
        self.fine_tuned_model_location = f"{self.model_name}_finetuned_{self.source_lang}_to_{self.target_lang}"

    def run_fine_tuning(self):
        file_en_tr, file_de_tr = read_file(self.train_path)
        file_en_val, file_de_val = read_file(self.eval_path)

        # TODO remove!!!!!!!!
        file_en_tr = file_en_tr[:1000]
        file_de_tr = file_de_tr[:1000]
        file_en_val = file_en_val[:25]
        file_de_val = file_de_val[:25]

        train_dataset =     self.get_dataset(input_sequences=file_de_tr, target_sequences=file_en_tr)
        val_dataset =       self.get_dataset(input_sequences=file_de_val, target_sequences=file_en_val)

        raw_datasets = {"train": train_dataset, "val": val_dataset}
        tokenized_datasets = {
            "train": raw_datasets["train"].map(self.preprocess_data, batched=True),
            "val": raw_datasets["val"].map(self.preprocess_data, batched=True)
        }

        args = Seq2SeqTrainingArguments(
            self.fine_tuned_model_location,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs_n,
            predict_with_generate=True,
            # fp16=True,
            report_to=["none"]
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["val"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_aux,
        )

        trainer.train()
        print(f"fine tuned model saved to: {self.fine_tuned_model_location}")

    def get_dataset(self, input_sequences, target_sequences):
        dataset = {
            "input_sequences": input_sequences,
            "target_sequences": target_sequences
        }
        return Dataset.from_dict(dataset)

    def preprocess_data(self, examples):
        inputs = [self.prefix + ex for ex in examples["input_sequences"]]
        targets = [ex for ex in examples["target_sequences"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # def compute_metrics(self, eval_preds):
    #     preds, labels = eval_preds
    #
        # if isinstance(preds, tuple):
        #     preds = preds[0]
    #     decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    #
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     # Some simple post-processing
    #     decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
    #
    #     result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     result = {"bleu": result["score"]}
    #
    #     prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
    #     result["gen_len"] = np.mean(prediction_lens)
    #     result = {k: round(v, 4) for k, v in result.items()}
    #     return result

    def compute_metrics_aux(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        res = compute_metrics(decoded_preds, decoded_labels)
        return res


if __name__ == '__main__':
    t5 = T5FineTune()
    t5.run_fine_tuning()
