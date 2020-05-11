import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import json
from preprocessing.TxtPreprocessor import run_txt_preprocessor
from embedding.YouTokenToMe.model import YouTokenToMe
import globals as globals_vars


if __name__ == "__main__":

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    assert os.path.isfile(config_path), f"missing embedding config.json: {config_path}"
    with open(config_path) as fd:
        config = json.load(fd)

    #############################################
    # FIT EMBEDDING (for English and French separated)
    #############################################

    emb_en = YouTokenToMe("en")
    print("Embedding : YouTokenToMe : training - english")
    emb_en.fit()

    emb_fr = YouTokenToMe("fr")
    print("Embedding : YouTokenToMe : training - french")
    emb_fr.fit()
