"""Create functions for predicting and outputting predictions."""

import pandas as pd
import numpy as np
import torch
from transformers import CamembertTokenizer
from keras_preprocessing.sequence import pad_sequences


def tokenize_sentence(sentence, tokenizer):
    """Tokenize the sentence
    Args:
        Sentence: sentence to tokenize
        Tokenizer: tokenizer adapted to the dataset
    Returns:
        Tokenized sentence by t.
    """
    tokenized_sentence = []
    for word in sentence:
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
    return tokenized_sentence


def run_prediction_when_text_tokenied(texts, tokenizer):
    """Predict the outputs of a text
    Args:
        Texts: sentence to tokenize
        Tokenizer: tokenizer adapted to the dataset
    Returns:
        Tokenized text.
        Labels: labels corresponding to the tokenized text
    """
    val_pad_text = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Encode the comments
    texts = texts.to_list()
    lists = [[]]
    for i in texts:
        if i == '.':
            lists[-1].append(".")
            lists.append([])
        else:
            lists[-1].append(i)
    texts = lists[:-1]
    tokenized_texts = [tokenize_sentence(sent, tokenizer) for sent in texts]
    # Transform token into ID (with BERT tokenizer) and add a 0 where the sentence doesn't have enough word to fill the maximum sentence length
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                             maxlen=MAX_LEN, dtype="long", value=val_pad_text,
                             truncating="post", padding="post")
    # Build binary arrays (0/1) to indicate where the sentence stops. 1 if the position corresponds to a token in input_ids, 0 if not
    attention_masks = [[float(i != val_pad_text) for i in ii] for ii in input_ids]
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    predictions = []
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have not provided labels.
        outputs = model(prediction_inputs.to(device), token_type_ids=None,
                        attention_mask=prediction_masks.to(device))
        # Move logits and labels to CPU
        logits = outputs[0].detach().cpu().numpy()
        # Calculate the accuracy for this batch of test sentences.
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    return tokenized_texts, predictions


def store_old_word_and_label(label_word, word, sentence, labels):
    """Predict the outputs of a text
    Args:
        Word: Word that is being reconstructed
        Sentence: List of words once they've been regrouped
        Label_word: Label corresponding to the word that is being reconstructed
        Labels: List of labels
    Returns:
        Word: Word that is being reconstructed
        Sentence: List of words once they've been regrouped
        Label_word: Label corresponding to the word that is being reconstructed
        Labels: List of labels
    """
    labels_name = ["COMMON", "DOSAGE", "DRUG", "DURATION", "FORM", "FREQUENCY", "ROUTE", "TREATMENT", "PAD" ]
    dict_idx2word = dict(zip(np.arange(9), labels_name))
    COMMON_val = 8
    if label_word.count(label_word[0]) == len(label_word): # if the array has always the same value I only keep one value as label
        label_word = label_word[0]
    else:
        pass
    # Do the regex
    word = word.split(" ")
    if len(word) > 1:  #If it was necessary to separate the characters
        if isinstance(label_word, (int, np.integer)): # If the label for those characters is unique
            sentence.extend(word)
            label_word = [label_word] * len(word)
            labels.extend(label_word)
        elif len(word) == len(label_word): #If the label of those characters is the same size as the number of characters
            # ['C', "'", 'est'] [0, 8, 0]
            sentence.extend(word)
            labels.extend(label_word)
        else:
            # array but not enough labels, do smth rdm
            sentence.extend(word)
            diff = len(word) - len(label_word)
            if diff > 0:
                label_word.append([COMMON_val] * diff) # We label the missing parts as COMMON
                labels.extend(label_word)
            else:
                label_word = label_word[:len(word)]
                labels.extend(label_word)
    else: # it was not necessary to separate the characters, append word in sentence
        if isinstance(label_word, (int, np.integer)): # If the label for those characters is unique
            sentence.extend(word)
            label_word = [label_word] * len(word)
            labels.extend(label_word)
        elif len(word) == len(label_word): #If the label of those characters is the same size as the number of characters
            # ['C', "'", 'est'] [0, 8, 0]
            sentence.extend(word)
            labels.extend(label_word)
        else:
            # array but not enough labels, do smth rdm
            sentence.extend(word)
            diff = len(word) - len(label_word)
            if diff > 0:
                label_word.append([COMMON_val] * diff) # We label the missing parts as COMMON
                labels.extend(label_word)
            else:
                label_word = label_word[:len(word)]
                labels.extend(label_word)
    return label_word, word, sentence, labels


def reconstruct_sentence_and_labels(bert_tokenized, labels_tokenized):
    """Reformat the output of the model to match the tokenization of the input file.
    Args:
        bert_tokenized: tokenized text outputted from the model.
        labels_tokenized: tokenized labels.
    Return:
        The reformatted sentences and labels.
    """
    sentence = []
    labels = []
    word = bert_tokenized[0]
    label_word = [labels_tokenized[0]]
    for elt_word, elt_label in zip(bert_tokenized[1:], labels_tokenized[1:]):
        if elt_word[0] == '▁':
            label_word, word, sentence, labels = store_old_word_and_label(label_word, word, sentence, labels)
            word = elt_word
            label_word = [elt_label]
        else:
            word += elt_word
            label_word.append(elt_label)
    # Add the last word and label
    label_word, word, sentence, labels = store_old_word_and_label(label_word, word, sentence, labels)
    return sentence, labels


def run_save_predictions(input_sub, model):
    """Run and Save predictions in a csv file.
    Args:
        input_sub: the csv file that is to be predicted.
        model: the model used for predictions.
    Return:
        The predictions of the input in a dataframe and saves them in a csv file.
    """
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    tokenized_texts, predictions = run_prediction_when_text_tokenied(input_sub["token"], tokenizer)

    tokenized_sentences_and_labels = [
        reconstruct_sentence_and_labels(sent, labs)
        for sent, labs in zip(tokenized_texts, predictions)
    ]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_sentences_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_sentences_and_labels]
    flat_tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    flat_labels = [item for sublist in labels for item in sublist]
    results = pd.DataFrame(flat_labels, columns=['Predicted'])
    results.index.name = 'TokenId'
    results.to_csv("posology_extraction_gr8/Outputs/Predictions/predictions.csv")
    return results