from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from collections import OrderedDict

import tensorflow as tf
import numpy as np
import joblib
import os

class Transliteration:

    def __init__(self, HYPERPARAMS, tgt_lang):
        self.HYPERPARAMS = HYPERPARAMS
        self.dataset_configs = {}

        raw_dataset = self.fetch_dataset(tgt_lang)
        self.train_ds, self.val_ds, self.test_ds = self.process_raw_dataset(raw_dataset)
        
    def fetch_dataset(self, tgt_lang):

        raw_dataset = {'train':{},
                       'val': {},
                       'test': {},
                       }

        # Read train, val, test set from the dataset files
        for dataset_type in ['train', 'dev', 'test']:
            # Path to read data from
            dataset_path = os.path.join("/content/dakshina_dataset_v1.0/", tgt_lang, "lexicons", tgt_lang + ".translit.sampled." + dataset_type + ".tsv")

            with open(dataset_path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")

            src_wordlist, tgt_wordlist = [], []
            if dataset_type == 'dev':
                dataset_type = 'val'
            
            for line in lines:

                # Skip lines that do not meet the expected format
                if len(line.split("\t")) < 3:
                    continue

                # Get the input and target word
                tgt_wd, src_wd, _ = line.split("\t")

                # Add start and end token to the target word
                tgt_wd = "\t" + tgt_wd + "\n"

                src_wordlist.append(src_wd)
                tgt_wordlist.append(tgt_wd)

            raw_dataset[dataset_type]['src_wdlist'] = src_wordlist
            raw_dataset[dataset_type]['tgt_wdlist'] = tgt_wordlist
            self.dataset_configs["updated_"+dataset_type+"_tgts"] = self.update_targets(src_wordlist, tgt_wordlist)
            self.dataset_configs["n_"+dataset_type+"_egs"] = len(src_wordlist)

        return raw_dataset


    def update_targets(self, src_wdlist, tgt_wdlist):
        """
        Many target words could have the same romanized src word. To fix this,
        an updated dictionary is created.
        """
        valid_tgts = OrderedDict()
        for src_wd, tgt_wd in zip(src_wdlist, tgt_wdlist):
            if src_wd not in valid_tgts.keys():
                valid_tgts[src_wd] = [tgt_wd[1:-1]]
            else:
                valid_tgts[src_wd].append(tgt_wd[1:-1])
        return valid_tgts


    def tokenize_dataset(self, raw_dataset):

        # Character level tokenization for source and target language
        src_lang_tokenizer = Tokenizer(char_level=True)
        tgt_lang_tokenizer = Tokenizer(char_level=True)
        src_lang_tokenizer.fit_on_texts(raw_dataset['train']['src_wdlist'])
        tgt_lang_tokenizer.fit_on_texts(raw_dataset['train']['tgt_wdlist'])

        # Saving the tokenizers and vocab lengths
        self.dataset_configs["src_vocab_size"] = len(src_lang_tokenizer.word_index) + 1
        self.dataset_configs["tgt_vocab_size"] = len(tgt_lang_tokenizer.word_index) + 1
        self.dataset_configs["src_lang_tokenizer"] = src_lang_tokenizer
        self.dataset_configs["tgt_lang_tokenizer"] = tgt_lang_tokenizer

        # Transforms each text in texts to a sequence of integers.
        train_inputs = src_lang_tokenizer.texts_to_sequences(raw_dataset['train']['src_wdlist'])
        train_tgts = tgt_lang_tokenizer.texts_to_sequences(raw_dataset['train']['tgt_wdlist'])
        val_inputs = src_lang_tokenizer.texts_to_sequences(raw_dataset['val']['src_wdlist'])
        val_tgts = tgt_lang_tokenizer.texts_to_sequences(raw_dataset['val']['tgt_wdlist'])

        # Padding each word to ensure same word length
        train_inputs = pad_sequences(train_inputs, padding="post")
        train_tgts = pad_sequences(train_tgts, padding="post")
        val_inputs = pad_sequences(val_inputs, padding="post", maxlen=train_inputs.shape[1])
        val_tgts = pad_sequences(val_tgts, padding="post", maxlen=train_tgts.shape[1])

        # Saving max sequence lengths
        self.dataset_configs["max_src_len"] = train_inputs.shape[1]
        self.dataset_configs["max_tgt_len"] = train_tgts.shape[1]

        return train_inputs, train_tgts, val_inputs, val_tgts

    def process_raw_dataset(self, raw_dataset):

        train_inputs, train_tgts, val_inputs, val_tgts = self.tokenize_dataset(raw_dataset)

        # Creating a training dataset
        train_ds = Dataset.from_tensor_slices((train_inputs, train_tgts))
        train_ds = train_ds.shuffle(self.dataset_configs["n_train_egs"]).batch(self.HYPERPARAMS['batch_size'], drop_remainder=True)

        # Creating a validation dataset
        val_ds = Dataset.from_tensor_slices((val_inputs, val_tgts))
        val_ds = (val_ds.repeat().shuffle(self.dataset_configs["n_val_egs"]).batch(self.HYPERPARAMS['batch_size'], drop_remainder=True))

        # Creating a tensor with validation data for evaluation in the training loop
        val_X = self.dataset_configs["src_lang_tokenizer"].texts_to_sequences(list(self.dataset_configs["updated_val_tgts"].keys()))
        val_X = pad_sequences(val_X, padding="post", maxlen=train_inputs.shape[1])

        # Creating a tensor with test data for final evaluation
        test_X = self.dataset_configs["src_lang_tokenizer"].texts_to_sequences(list(self.dataset_configs["updated_test_tgts"].keys()))
        test_X = pad_sequences(test_X, padding="post", maxlen=train_inputs.shape[1])

        return (train_ds, (val_ds, val_X), test_X)