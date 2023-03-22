import transformers
from datasets import load_dataset, load_metric, Dataset
from project_evaluate import read_file, compute_metrics, postprocess_text
from transformers import AutoTokenizer, T5Tokenizer, MT5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import datetime
from preprocess_dp import PreprocessDP
# os.listdir?
# conda activate azureml_py38


class T5FineTune:
    def __init__(self):
        # ~~~~ params
        self.prefix = "translate German to English: "
        self.max_input_length = self.max_target_length = 1024
        self.batch_size = 1
        self.epochs_n = 5
        self.source_lang = "de"
        self.target_lang = "en"

        # ~~~~ model stuff
        self.model_name = "t5-base"
        # self.model_name = "t5-base_finetuned_de_to_en/checkpoint-50000"

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
        self.fine_tuned_model_location = f"T5_base_finetuned"

        # self.true_english_list_val = None

    def run_fine_tuning(self):
        # get preprocessed data
        preprocess = PreprocessDP(self.tokenizer, self.max_input_length, self.max_target_length)
        tokenized_datasets = preprocess.get_preprocessed_datasets()

        args = Seq2SeqTrainingArguments(
            self.fine_tuned_model_location,
            evaluation_strategy="epoch",
            learning_rate=0.001,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.001,
            save_total_limit=3,
            num_train_epochs=self.epochs_n,
            predict_with_generate=True,
            generation_max_length=1500,
            generation_num_beams=2
            # fp16=True,
            # report_to='none'  # TODO ?
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
        logger(f"fine tuned model saved to: {self.fine_tuned_model_location}")

    def compute_metrics_aux(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        english_output = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        true_english_input = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        english_output, true_english_input = postprocess_text(english_output, true_english_input)

        result = self.metric.compute(predictions=english_output, references=true_english_input)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        logger(f"{datetime.datetime.now()} -------- {result}")
        return result


def logger(message: str):
    with open('log.txt', 'a') as f:
        log = f"{datetime.datetime.now()} ------ " + message + "\n"
        f.write(log)
        print(log)


if __name__ == '__main__':
    logger("++++++++++++++++ session starts ++++++++++++++++")
    t5 = T5FineTune()
    t5.run_fine_tuning()

    from huggingface_hub import login

    login("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")
    t5.model.push_to_hub('YaelHari/my-Finetuned-t5')


