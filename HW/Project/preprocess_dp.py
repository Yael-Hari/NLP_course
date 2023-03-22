import torch
from torch.utils.data import DataLoader
import time
import spacy
from tqdm import tqdm
import pickle
from transformers import AutoModel, AutoTokenizer
from project_evaluate import read_file as read_labeled_file
from typing import List, Tuple
from datasets import load_dataset, load_metric, Dataset


class PreprocessDP:
    def __init__(self, tokenizer, max_input_length, max_target_length):
        # paths
        self.train_path = 'data/train.labeled'
        self.eval_path = 'data/val.labeled'
        self.eval_path_unlabeled = 'data/val.unlabeled'
        self.comp_path = 'data/comp.unlabeled'

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def get_preprocessed_datasets(self):
        train_dict = self.get_dependency_parsing_train_dict()
        eval_dict = self.get_dependency_parsing_eval_dict()
        comp_dict = self.get_dependency_parsing_unlabeled_dict(self.comp_path)
        eval_unlabeled_dict = self.get_dependency_parsing_unlabeled_dict(self.eval_path_unlabeled)

        train_dict = self.preprocess_german_input(train_dict)
        eval_dict = self.preprocess_german_input(eval_dict)
        comp_dict = self.preprocess_german_input(comp_dict)
        eval_unlabeled_dict = self.preprocess_german_input(eval_unlabeled_dict)

        train_dataset = Dataset.from_dict(train_dict)
        eval_dataset = Dataset.from_dict(eval_dict)
        comp_dataset = Dataset.from_dict(comp_dict)
        eval_unlabeled_dataset = Dataset.from_dict(eval_unlabeled_dict)

        tokenized_train_dataset = train_dataset.map(self.tokenize_labeled_data, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.tokenize_labeled_data, batched=True)
        tokenized_comp_dataset = comp_dataset.map(self.tokenize_unlabeled_data, batched=True)
        tokenized_eval_unlabeled_dataset = eval_unlabeled_dataset.map(self.tokenize_unlabeled_data, batched=True)

        tokenized_datasets = {
            "train": tokenized_train_dataset,
            "val": tokenized_eval_dataset,
            "comp": tokenized_comp_dataset,
            "val_unlabeled": tokenized_eval_unlabeled_dataset
        }

        return tokenized_datasets

    def tokenize_labeled_data(self, dataset):
        inputs = [de for de in dataset["DE"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        targets = [ex for ex in dataset["EN"]]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_unlabeled_data(self, dataset):
        inputs = [de for de in dataset["DE"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        return model_inputs

    def preprocess_german_input(self, dp_dict):
        processed_de = []
        for de, roots_list, mods_of_roots in zip(dp_dict["DE"], dp_dict["ROOTS"], dp_dict["ROOTS_MODIFIERS"]):
            de_input = "translate German to English with"
            for root, mods in zip(roots_list, mods_of_roots):
                de_input += f" {root} -> {mods}"

            de_input += f": {de}"
            processed_de.append(de_input)

        dp_dict["DE"] = processed_de
        dp_dict.pop("ROOTS")
        dp_dict.pop("ROOTS_MODIFIERS")
        return dp_dict

    def get_dependency_parsing_train_dict(self) -> dict:
        spacy.cli.download("en_core_web_sm")
        print("tagging dependencies ...")
        file_en, file_de = read_labeled_file(self.train_path)
        nlp_lm = spacy.load('en_core_web_sm')
        data_with_dp = {
            "DE": [],
            "EN": [],
            "ROOTS": [],
            "ROOTS_MODIFIERS": []
        }
        for en, de in tqdm(zip(file_en, file_de)):
            doc = nlp_lm(en)

            roots = []
            mods_of_roots = []
            for token in doc:
                if token.dep_ == 'ROOT':
                    roots.append(token.text)

                    # get 2 of the root's modiffiers
                    mods = [child.text for child in token.children]
                    if len(mods) >= 2:
                        mods_of_roots.append(tuple(mods[:2]))
                    elif len(mods) == 1:
                        mods_of_roots.append(tuple([mods[0], '--']))
                    else:
                        mods_of_roots.append(tuple(['--', '--']))
                else:
                    continue

            data_with_dp["DE"].append(de)
            data_with_dp["EN"].append(en)
            data_with_dp["ROOTS"].append(roots)
            data_with_dp["ROOTS_MODIFIERS"].append(mods_of_roots)

        # dp_dataset = Dataset.from_dict(data_with_dp)
        return data_with_dp

    def get_dependency_parsing_eval_dict(self):
        eval_dict = {
            "DE": [],
            "EN": [],
            "ROOTS": [],
            "ROOTS_MODIFIERS": []
        }
        file_en_val, file_de_val = read_labeled_file(self.eval_path)
        _, roots_list, mods_roots_list = self.read_unlabeled_file(self.eval_path_unlabeled)
        for en, de, roots, mods_of_roots in zip(file_en_val, file_de_val, roots_list, mods_roots_list):
            eval_dict["DE"].append(de)
            eval_dict["EN"].append(en)
            eval_dict["ROOTS"].append(roots)
            eval_dict["ROOTS_MODIFIERS"].append(mods_of_roots)
        return eval_dict

    def get_dependency_parsing_unlabeled_dict(self, file_path):
        file_de, roots_list, mods_roots_list = self.read_unlabeled_file(file_path)
        eval_dict = {
            "DE": [],
            "ROOTS": [],
            "ROOTS_MODIFIERS": []
        }
        _, roots_list, mods_roots_list = self.read_unlabeled_file(self.eval_path_unlabeled)
        for de, roots, mods_of_roots in zip(file_de, roots_list, mods_roots_list):
            eval_dict["DE"].append(de)
            eval_dict["ROOTS"].append(roots)
            eval_dict["ROOTS_MODIFIERS"].append(mods_of_roots)
        return eval_dict

    def read_unlabeled_file(self, file_path):
        de_list, roots_list, mods_roots_list = [], [], []
        with open(file_path, encoding='utf-8') as f:
            cur_str, cur_list = '', []
            for line in f.readlines():
                line = line.strip()
                if line == 'German:':
                    if len(cur_str) > 0:
                        cur_list.append(cur_str.strip())
                        cur_str = ''
                    cur_list = de_list
                    continue
                elif "Roots in English: " in line:
                    roots = line.replace("Roots in English: ", "").split(", ")
                    roots_list.append(roots)
                elif "Modifiers in English: " in line:
                    mods = line.replace("Modifiers in English: ", "").replace("–", "--").strip('()').split("), (")
                    mods = [tuple(x.split(', ')) for x in mods]
                    mods_roots_list.append(mods)
                else:
                    cur_str += line + ' '

        if len(cur_str) > 0:
            cur_list.append(cur_str)
        return de_list, roots_list, mods_roots_list


# if __name__ == '__main__':
#     preprocess = PreprocessDP()
#     preprocess.get_preprocessed_datasets()

"""



758 --------------------------------------------------
1 
['In all of these domains, as well as at the global level, where reform is equally urgent, there is no longer any room for complacency.']
3 
['Die Entscheidung der EZB vom Dezember 2011, ihr langfristiges Refinanzierungsgeschäft einzuführen, das Geschäftsbanken mit einer günstigen Dreijahresfinanzierung versorgt, erfüllt diese fünf Bedingungen', 
'Die Dauer der LRG war besonders angemessen angesichts der wachsenden Gefahr einer großen Störung im europäischen Bankensektor im Oktober, November und Anfang Dezember letzten Jahres', 
'Zudem hat EZB-Präsident Mario Draghi, mein Nachfolger, klar und deutlich gesagt, wie wichtig die Stützung der Bilanzen der Banken, die Anpassung der Strategien der einzelnen Länder und die Verbesserung der Steuerung in der Eurozone und in Europa als Ganzem sind.']



182 987 --------------------------------------------------
4 ['BRUSSELS – Ten years ago, Germany was considered the sick man of Europe', 
'Its economy was mired in recession, while the rest of Europe was recovering; its unemployment rate was higher than the eurozone average; it was violating the European budget rules by running excessive deficits; and its financial system was in crisis', 
'A decade later, Germany is considered a role model for everyone else', 
'But should it be?']

6 ['BRÜSSEL – Vor zehn Jahren galt Deutschland als kranker Mann Europas', 
'Während sich der Rest Europas erholte, steckte seine Wirtschaft in der Rezession', 
'Die Arbeitslosenrate lag höher als der Durchschnitt in der Eurozone', 
'Aufgrund übermäßiger Haushaltsdefizite brach man europäische Budgetregeln und das deutsche Finanzsystem befand sich in der Krise', 
'Ein Jahrzehnt später gilt Deutschland als Vorbild für alle anderen', 
'Zu Recht? &#160;']




"""
