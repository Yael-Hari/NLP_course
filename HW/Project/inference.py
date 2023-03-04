from datasets import load_dataset, load_metric, Dataset
from project_evaluate import read_file, compute_metrics, postprocess_text, calculate_score
from transformers import AutoTokenizer, T5Tokenizer, MT5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import datetime
from T5_fine_tune import T5FineTune


t5 = T5FineTune()

# load model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base_finetuned_de_to_en/checkpoint-50000")

# load data
file_en_val, file_de_val = read_file(t5.eval_path)

german_preds_pairs = []
for german_input in file_de_val:
    input_ids = t5.tokenizer.encode(german_input, return_tensors='pt', max_length=t5.max_input_length, truncation=True)
    output = model.generate(
        input_ids,
        max_length=t5.max_target_length,
        num_beams=8,
        min_length=100,
    )
    english_output = t5.tokenizer.decode(output[0], skip_special_tokens=True)
    german_preds_pairs.append((german_input, english_output))


# write to file
file_name = "data/inference_val.labeled"
with open(file_name, "w") as f:
    for ger, eng in german_preds_pairs:
        f.write("German:\n")
        f.write(ger + "\n")
        f.write("English:\n")
        f.write(eng + "\n")


score = calculate_score("data/val.labeled", file_name)

print("score:", score)
