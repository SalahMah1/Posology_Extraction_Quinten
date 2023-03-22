# Posology Extraction

## Introduction
Drug posology is a term to define all the conditions under which a patient gets administered a treatment.
Here, we focus on 7 attributes of posology:
- Treatment
- Drug
- Dosage
- Frequency
- Route
- Form
- Duration
Our objective is to train a NLP model for a Named-entity recognition (NER) task. More specifically, an information extraction subtask that aims at finding and grouping identified items in unstructured text into predetermined categories.

## Inputs
The inputs consist of two files.
- One data training label file with 567 texts and annotated inputs from doccano, containing a text and the labels positions and values. The text contains summaries of patients admission records in French speaking countries. Each text corresponds to a different patient, a different intervention.


## Outputs
The outputs are divided into five folders.
#### Evaluation
To evaluate our model we look at two metrics.
- The Macro Average F1-score grpah measures the model performance.
- The Macro Average F1 score, It is the harmonic mean of precision and the recall.
F1-score = 2 * {precison * recall} / {precision + recall}.

- We also make use of the loss function. With the loss fuction we compute the difference between the expected output and the obtained output. The closer to 0 the better. It is the absolute difference between the expected and predicted results.

#### Model
The model contains files with information on the models that were trained.

#### Preprocessed
We have one csv file with the tokenized annotations of our work on Doccano. We have a table with columns of the text, labels, tokenized text, a column for each of the posology attributes as well as the tokenized text label column.

## Source
The src is divided into three main folders.

## DataAugmented
This file contains the output of the data that is augmented.

## Predictions
This file contains the predictions for an input given in a csv file.


#### Modelling
Modelling contains a Model Class in which a Camembert ModelCreating. Several methods are created there to create the optimizer and scheduler, train, save and evaluate the model.

#### Preprocessing
Preprocessing input from Doccano in a format readable by the model.
We have fuctions that are used to extract the labels of the data, as well as to create tokenized labels.
A function for padding, to truncate and expand sentences/labels based on model sentence size.\
As well as a function to split the data into two : the train set and the testing set.

#### Data Augmentation
Three methods used to increase the number of input data in order to improve the results.
1. Using CamemBERT : replacing random words with synonyms. Up to three per text, each replaced by 3 synonyms, giving us 9 new texts for each text.
2. By creating mock sentences with gaps for the words of interest (route, drugs, dosage ...) and using a data base of possible words to fill in the gaps. Creating an infinity of possible sentences.
3. Create random labels from external and training data. Extract from the texts the sentences with labels, readjust position labels. Create a set of 125 (can be tuned) sentences for a random number of sentences and using the randomized label creator function.
All three methods are combined to give a final dataframe. The user can choose to tune the parameters (such as the number of sentences generated per augmentation and more).

<hr />
###### Requirements
The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using
    pip install -r requirements.txt
<hr />

###### Citations :
@inproceedings{martin2020camembert,
  title={CamemBERT: a Tasty French Language Model},
  author={Martin, Louis and Muller, Benjamin and Ortiz Su{\'a}rez, Pedro Javier and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}

