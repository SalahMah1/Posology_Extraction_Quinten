"""Main Pipeline. """

import os
import pandas as pd
import logging
from time import time
import sys
import json

logger = logging.getLogger('main_logger')

sys.path.insert(0,"Preprocessing/")
sys.path.insert(0,"Modelling/")
sys.path.insert(0,"Utils/")
sys.path.insert(0,"DataAugmentation/")

import preprocessing
import modelling
import utils
import dataaugmentation

def main(path_conf):
    
    START = time()
    conf = json.load(open(path_conf, 'r'))
    input_path = conf["paths"]["Inputs_path"]
    input_file = conf["paths"]["input_file"]
    output_path = conf["paths"]["Outputs_path"]
    folder_path = conf["paths"]["folder_preprocessed"]
    output_file = conf["paths"]["post_doccano_file"]
    input_df = pd.read_json(input_path + input_file,lines=True).set_index("id").drop(columns=["Comments"])
    # input_df = dataaugmentation.augment_data(num_sent_used=150, num_sample_per=3, merge_ds=True)
    post_docanno_df = preprocessing.preprocess_post_docanno(input_df, conf["labels"])
    post_docanno_df.to_csv(output_path+folder_path+output_file, index=False)
    train_dataloader, valid_dataloader = preprocessing.train_test_split_data(post_docanno_df, conf)
    model = modelling.Model(train_dataloader, valid_dataloader, conf)
    model.train()
    model.evaluation()
    model.save_model()
    #loaded_model = utils.load_model(conf, "20-10-2022-13:50")
    
    logger.debug("Time for total execution :" + str(time() - START))

if __name__ == '__main__':
    
    path_conf = '../params/config.json'
    try:
        main(path_conf=path_conf)
    
    except Exception as e:
        logger.error("Error during execution", exc_info=True)