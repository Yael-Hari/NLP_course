import datetime
import time

import numpy as np
from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, MT5Tokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5Tokenizer)

from project_evaluate import (calculate_score, compute_metrics,
                              postprocess_text, read_file)
from T5_fine_tune import T5FineTune


def print_time(start):
    now = time.time()
    elapsed_time = now - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

t5 = T5FineTune()

# load model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base_finetuned_de_to_en/checkpoint-50000_finetuned_de_to_en/checkpoint-50000")

# load data
file_en_val, file_de_val = read_file(t5.eval_path)

running_sum = 0
running_count = 0
german_preds_pairs = []
start = time.time()
for true_english_input, german_input in zip(file_en_val, file_de_val):
    input_ids = t5.tokenizer.encode(german_input, return_tensors='pt', max_length=t5.max_input_length, truncation=True)
    output = model.generate(
        input_ids,
        max_length=(int(len(german_input.split()) * 2)),
        num_beams=32,
        # top_p=0.3,
        min_length=(int(len(german_input.split()) * 2 / 3)),
    )
    english_output = t5.tokenizer.decode(output[0], skip_special_tokens=True)
    
    curr_score = compute_metrics([english_output], [true_english_input])
    running_sum += curr_score
    running_count += 1
    avg_BLEU = running_sum / running_count
    m, s = print_time(start)
    len_true = len(true_english_input.split())
    len_pred = len(english_output.split())
    msg = f"{running_count} | time: {m}:{s} | curr_score: {round(curr_score, 2)} | avg_BLEU: {round(avg_BLEU,3)} |"
    msg += f" l_true: {len_true} | l_pred: {len_pred} | l_diff: {len_true-len_pred}"
    print(msg)
    
    german_preds_pairs.append((german_input, english_output))

    # write to file
    file_name = "data/inference_val.labeled"
    with open(file_name, "a") as f:
        # for ger, eng in german_preds_pairs:
        f.write("German:\n")
        f.write(german_input + "\n")
        f.write("English:\n")
        f.write(english_output + "\n\n")
    
    file_name2 = "data/debug_inference_val.labeled"
    with open(file_name2, "a") as f:
        f.write(f"{running_count}\n")
        f.write("German:\n")
        f.write(german_input + "\n")
        f.write("Pred English:\n")
        f.write(english_output + "\n\n")
        f.write("True English:\n")
        f.write(true_english_input + "\n\n")
    


score = calculate_score("data/val.labeled", file_name)

print("score:", score)
