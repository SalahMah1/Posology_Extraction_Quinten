"Preprocessing input from doccano in a format readable by model."

import re
import numpy as np
import torch
import logging
from transformers import CamembertTokenizer, CamembertForTokenClassification, AdamW
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import trange
logger = logging.getLogger('main_logger')


def extract_labels(x, conf_labels):
    """Create columns for each label, extracting the words associated to the label.
    Args:
        x: The dataframe with the columns needed to create the output
    Returns:
        The dataframe with additional columns for each label.
    """
    labels = conf_labels["labels_name"]
    labels_common = []
    labels_dosage = []
    labels_drug = []
    labels_duration = []
    labels_form = []
    labels_frequency = []
    labels_route = []
    labels_treatment = []
    name_list_labels = [labels_common, labels_dosage, labels_drug, labels_duration, labels_form, labels_frequency, labels_route, labels_treatment]
    dict_labels_list = dict(zip(labels, name_list_labels))
    for elts in x[1]:
        sentence = x[0][elts[0]:elts[1]]
        sentence = re.sub("/j", " /j", sentence)
        sentence = re.sub("/24h", " /24h", sentence)
        sentence = re.sub("/12h", " /12h", sentence)
        sentence = re.sub("/48h", " /48h", sentence)
        sentence = re.sub("/72h", " /72h", sentence)
        sentence = re.sub("\s(?=ng|mU|pmol|nmol|mg|μg|ml|Gray|fois|UI|min|heures|heure|minutes|minute|g|grammes|gramme|Gy|semaines|jours|semaine|jour|mois|moi|séances)",
            "", sentence)
        sentence = re.sub("[^\w\/\d+,]", " ", sentence).split()
        dict_labels_list[elts[2]].append(sentence)
    x[conf_labels["LABEL0"]] = labels_common
    x[conf_labels["LABEL1"]] = labels_dosage
    x[conf_labels["LABEL2"]] = labels_drug
    x[conf_labels["LABEL3"]] = labels_duration
    x[conf_labels["LABEL4"]] = labels_form
    x[conf_labels["LABEL5"]] = labels_frequency
    x[conf_labels["LABEL6"]] = labels_route
    x[conf_labels["LABEL7"]] = labels_treatment
    return x[labels]


def create_tokenized_label(x):
    """Create a column with arrays of labels associated to each token in the text.    
    Args:
        x: The dataframe with the columns needed to create the output
    Returns:
        Array of the labels associated to each token in the text.
    """
    text = x[0]
    tokenized_label = np.zeros(len(text))
    for i in range(1, 9):
        for group_words in x[i]:
            nb_words = len(group_words)
            indx_start = [i for i, w in enumerate(text) if w == group_words[0]]
            for ind_start in indx_start:
                tokenized_label[ind_start:ind_start+nb_words] = i - 1
    return tokenized_label


def preprocess_post_docanno(df, conf_labels):
    """Tokenize input data and add labels column.    
    Args:
        df: The input dataframe with the text and labels from doccano
        labels: the labels
    Returns:
        A dataframe containing the tokenized text and a column
        with an array of labels corresponding to each tokenized word.
    """
    labels = conf_labels["labels_name"]
    df = df[df.label.astype(bool)]
    df['tokenized_text'] = df['text'].apply(lambda x: re.sub("/j", " /j", x))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x:
                                                      re.sub("/24h", " /24h", x)
                                                     )
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: re.sub("/48h", " /48h", x))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: re.sub("/72h", " /72h", x))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: re.sub("/12h", " /12h", x))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: re.sub("\s(?=ng|mU|pmol|nmol|mg|μg|ml|Gray|fois|UI|min|heures|heure|minutes|minute|g|grammes|gramme|Gy|semaines|jours|semaine|jour|mois|moi|séances)", "",  x))
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: re.sub("[^\w\/\d+,]", " ",  x).split())
    df[labels] = df[["text", "label"]].apply(lambda x: extract_labels(x, conf_labels),
                                             axis=1)
    df["tokenized_text_labels"] = df[
        ["tokenized_text"] + labels
        ].apply(lambda x: create_tokenized_label(x), axis=1)
    return df

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """Tokenize based on model tokenization and add labels.    
    Args:
        sentence: each row in the dataframe corresponding to tokenized text.
        text_labels: each array of labels corresponding to tokenized text.
    Returns:
        Sentences tokenized according to model tokenization and corresponding labels.
    """
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels

def padding(df, conf_padding):
    """Truncate and expand sentences/labels based on model sentence size.    
    Args:
        df: dataframe containing the tokenized text and associated labels.
        conf_padding: configured variables for padding.
    Returns:
        Sentences and labels padded and an attention mask.
    """
    # Initialize CamemBERT tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(conf_padding["tokenizer_model"])
    text = df['tokenized_text'].to_list()
    labels = df['tokenized_text_labels'].tolist()
    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, tokenizer)
        for sent, labs in zip(text, labels)
    ]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    # Transform token into ID and add a 0 where the sentence doesn't have enough word to fill the maximum sentence length
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=conf_padding["MAXLEN"], dtype=conf_padding["dtype"],
                              value=conf_padding["val_pad_text"],
                              truncating=conf_padding["truncating"],
                              padding=conf_padding["padding"])
    # Build arrays of labels corresponding to token label of input_ids arrays
    tags = pad_sequences(labels, maxlen=conf_padding["MAXLEN"],
                         value=conf_padding["val_pad_label"],
                         padding="post", dtype=conf_padding["dtype"],
                         truncating=conf_padding["truncating"])
    # Build binary arrays (0/1) to indicate where the sentence stops. 1 if the position corresponds to a token in input_ids, 0 if not
    attention_masks = [[float(i != conf_padding["val_pad_text"]) for i in ii] for ii in input_ids]
    return input_ids, tags, attention_masks

def train_test_split_data(df, conf):
    """Split the training set in train and validation set.    
    Args:
        df: dataframe containing the training data.
        conf: configured variables.
    Returns:
        Train and validation dataloader.
    """
    input_ids, tags, attention_masks = padding(df, conf["padding"])
    conf_split = conf["train_test_split"]
    # Train test split
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=conf_split["random_state"],
                                                                test_size=conf_split["test_size"])
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                                random_state=conf_split["random_state"],
                                                                test_size=conf_split["test_size"])
    tr_inputs_ts = torch.tensor(tr_inputs)
    val_inputs_ts = torch.tensor(val_inputs)
    tr_tags_ts = torch.tensor(tr_tags)
    val_tags_ts = torch.tensor(val_tags)
    tr_masks_ts = torch.tensor(tr_masks)
    val_masks_ts = torch.tensor(val_masks)
    train_data = TensorDataset(tr_inputs_ts, tr_masks_ts, tr_tags_ts)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=conf_split["batch_size"])
    valid_data = TensorDataset(val_inputs_ts, val_masks_ts, val_tags_ts)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=conf_split["batch_size"])
    return train_dataloader, valid_dataloader