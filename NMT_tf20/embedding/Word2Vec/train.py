import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import json
from preprocessing.TxtPreprocessor import run_txt_preprocessor
from embedding.Word2Vec.model import Word2Vec
import globals as globals_vars


if __name__ == "__main__":

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    print(config_path)
    assert os.path.isfile(config_path), f"missing embedding config.json: {config_path}"
    with open(config_path) as fd:
        config = json.load(fd)

    #############################################
    # Preprocess datasets (aligned + unaligned) (for English and French separated)
    #############################################
    if config["preprocessing"]["raw_aligned"]["en"].get("split", 1) < 1:
        df_fr, _ = run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
        )
    else:
        df_fr = run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
        )
    df_fr = df_fr.append(
        run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_unaligned", "fr"
        )
    )
    # Preprocess EN datasets (aligned + unaligned)
    if config["preprocessing"]["raw_aligned"]["en"].get("split", 1) < 1:
        df_en, _ = run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
        )
    else:
        df_en = run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
        )

    df_en = df_en.append(
        run_txt_preprocessor(
            {"preprocessing": config["preprocessing"]}, "raw_unaligned", "en"
        )
    )
    #############################################
    # FIT EMBEDDING (for English and French separated)
    #############################################

    emb_en = Word2Vec("en")
    print("Embedding : Word2Vec : training - english")

    if config["embedding_model"]["en"]["fit_vocab_only"]:
        emb_en.fit_vocabulary(df_en, "en")
    else:
        emb_en.fit(df_en)

    emb_en.save_model()

    emb_fr = Word2Vec("fr")
    if config["embedding_model"]["fr"]["fit_vocab_only"]:
        emb_fr.fit_vocabulary(df_fr, "fr")
    else:
        emb_fr.fit(df_fr)
    print("Embedding : Word2Vec : training - french")
    emb_fr.save_model()
