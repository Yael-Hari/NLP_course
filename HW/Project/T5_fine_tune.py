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
        self.max_input_length = self.max_target_length = 1024
        self.batch_size = 1
        self.epochs_n = 5
        self.source_lang = "de"
        self.target_lang = "en"

        # ~~~~ model stuff
        self.model_name = "t5-base_finetuned_de_to_en/checkpoint-50000"
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

        # self.true_english_list_val = None

    def run_fine_tuning(self):
        file_en_tr, file_de_tr = read_file(self.train_path)
        file_en_val, file_de_val = read_file(self.eval_path)
        
        # TODO: remove!!!
        # file_en_tr = file_en_tr[:3]
        # file_de_tr = file_de_tr[:3]
        # file_en_val = file_en_val[:3]
        # file_de_val = file_de_val[:3]
        
        # self.true_english_list_val = file_en_val

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
            learning_rate=0.001,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=self.epochs_n,
            predict_with_generate=True,
            generation_max_length=1500,
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
        return result



if __name__ == '__main__':
    t5 = T5FineTune()
    # t5.run_fine_tuning()
    
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base_finetuned_de_to_en/checkpoint-50000_finetuned_de_to_en/checkpoint-50000")

    # predict
    de_input1 = r"Und weiterreichende Kürzungen wie die von der EU vorgeschlagenen – 20 Prozent " \
        r"unterhalb der Werte von 1990 innerhalb von zwölf Jahren – würden die globalen Temperaturen "\
        r"bis 2100 lediglich um ein Sechzigstel Grad Celsius (ein Dreißigstel Grad Fahrenheit) senken, " \
        r"und das bei Kosten von 10 Billionen Dollar. "\
        r"Für jeden ausgegebenen Dollar hätten wir nur eine Wertschöpfung von vier Cent erreicht. "
    de_input2 = "Offiziell bleibt China kommunistisch."\
        r"Doch in China sind Unternehmen mit weit weniger Regulierungen konfrontiert als in Taiwan, Korea,"\
        r" Japan, Deutschland, Frankreich und Schweden."\
        r"Selbst im Vergleich zu den USA ist China ein kapitalistisches Paradies - "\
        r"zumindest solange man sich von der Zentralregierung fernhält."\
        r"Unternehmen, die sich das chinesische System der regionalen Zollfreizonen und Steueroasen zunutze machen,"\
        r" zahlen beispielsweise gar keine oder extrem niedrige Zölle (die zwar von der Zentralregierung"\
        r" festgesetzt, aber von lokalen Behörden verwaltet werden)."

    input_ids = t5.tokenizer.encode(de_input1, return_tensors='pt', max_length=t5.max_input_length, truncation=True)
    output = model.generate(input_ids, max_length=2000, min_length=50, num_beams=4)
    #                   min_length=100      min_length=50       min_length=25
    # num_beams=100 |   27.02 @@        |   27.02           |   27.02
    # num_beams=64  |   27.02           |   27.02           |   27.02
    # num_bames=32  |   25.93
    # num_beams=24  |   25.77
    # num_beams=20  |   27.82 @@        |   27.82           |   27.82             min_length=50+earlystop 21.53  
    # num_beams=16  |   26.86           |   26.86           |   26.86
    # num_beams=8   |   27.34 @@        |   27.51           |   27.51
    # num_beams=4   |   24.43
    # num_beams=3   |   24.43
    # top_p=0.9     |   11.1
    # top_p=0.1     |   11.1
    # top_p=0.05    |   11.1
    # top_k=30      |   11.1
    # top_k=3       |   11.1

        #                   min_length=100      min_length=50       min_length=25
    # 20 15.92
    # 16 20.82
    # 8 20.82
    pred_en = t5.tokenizer.decode(output[0], skip_special_tokens=True)
    print(pred_en)
    true_en1 = r"And deeper emissions cuts like those proposed by the European Union – 20% below " \
        r"1990 levels within 12 years – would reduce global temperatures by only one-sixtieth of one " \
        r"degree Celsius (one-thirtieth of one degree Fahrenheit) by 2100, at a cost of $10 trillion. "\
        r"For every dollar spent, we would do just four cents worth of good."
    true_en2 = r"Officially, China remains Communist."\
        r"Yet companies in China face far fewer regulations than in Taiwan, Korea, Japan, Germany, France, and Sweden."\
        r"Even in comparison with the US, China is a capitalist paradise - so long as you steer clear of the central government."\
        r"For example, tariffs (which are set by the central government, but administered locally) are low or nonexistent for"\
        r" companies that take advantage of China's regional systems of tax-free zones and tax benefits."\

    score = compute_metrics([pred_en], [true_en1])
    print(score)


