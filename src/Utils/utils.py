"""Add utilities functions for the pipeline."""

import pickle
import logging
logger = logging.getLogger('main_logger')

def load_model(conf, date_time):
    """Save the model.
    Args:
        Trained model.
    Returns:
        None.
    """
    filename = conf["paths"]["Outputs_path"] + conf["paths"]["folder_models"] + conf["model"]["model_name"] + date_time +'.sav'
    clf = pickle.load(open(filename, 'rb'))
    logger.info('Modele chargé: ' + filename)
    return clf