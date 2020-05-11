import sys
import os

NUMPY_SEED = 24
TENSORFLOW_SEED = 42

from numpy.random import seed

seed(NUMPY_SEED)
# FORCING CPU (TEMPORARY) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# FORCING CPU (TEMPORARY) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import tensorflow as tf

tf.random.set_seed(TENSORFLOW_SEED)


import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from preprocessing.TxtPreprocessor import run_txt_preprocessor
import json

import tensorflow.keras as keras
import importlib
import numpy as np
import copy
import wandb
from tqdm import tqdm
import shutil
import pandas as pd
import fire

# from models.GRU_1to1_OneHot.model import GRU
from models.utils.train_utils import *
from models.utils.predict_utils import *

from evaluator import get_bleu
import time
import pathlib

model_name = os.path.dirname(os.path.abspath(__file__)).split("/")[
    -1
]  # e.g.: GRU_1to1_Onehot

################# UTILS FUNCTIONS ###################
from models.utils.train_utils import *
from evaluator import get_bleu


def run_predict(config, run_name):
    ################# SETUP  ###################
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    ################# PREPROCESSINGS  ###################
    _, df_en_valid, _, df_fr_valid = get_preprocessed_dfs(config)

    # Very useful print to always make sure we have all rows correctly
    print("Predict:  quick validation below (DO NOT DELETE) ##############")
    print("df_en_valid.shape: ", df_en_valid.shape)
    print("df_fr_valid.shape: ", df_fr_valid.shape)
    df_concat = pd.concat([df_en_valid, df_fr_valid])
    print(
        "max_seq across all 2 valid datas: {}\n".format(
            (df_concat.text.str.count(" ") + 1).max()
        )
    )

    ################# EMBEDDINGS ###################
    # Load embedding models
    embedding_module = importlib.import_module(
        "embedding." + config["embedding"] + "." + "model"
    )
    config_embedding_path = os.path.join(
        current_file_dir, run_name, "embedding", "config.json"
    )
    emb_en_obj = getattr(embedding_module, config["embedding"])(
        "en", config_embedding_path
    )
    emb_fr_obj = getattr(embedding_module, config["embedding"])(
        "fr", config_embedding_path
    )
    emb_fr_obj._load_model()

    # Generate embedding
    x_valid = tf.ragged.constant(emb_en_obj.generate(df_en_valid))
    y_valid = tf.ragged.constant(emb_fr_obj.generate(df_fr_valid))

    ################# PREPARE DATASETS ###################
    dataset = (
        tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        .batch(config["model"]["batch_size"] / 4)
        .shuffle(buffer_size=1024)
    )

    ################# MODEL SETUP  ###################
    model, _, _ = get_model(
        config["model"],
        model_name,
        [emb_fr_obj.vocab_size],
        config["model"]["model_kwargs"],
    )

    x_sequences = []
    pred_sequences = []
    true_sequences = []

    ################# VALID ###################
    for step, (x_batch, y_batch) in enumerate(dataset):
        # Persist the X and y_true for stats
        x_sequences.extend(tf.cast(x_batch, dtype=tf.int32).to_list())
        true_sequences.extend(tf.cast(y_batch, dtype=tf.int32).to_list())

        # Padding batch of sequences
        x_batch, y_batch = pad_batch_sequences(x_batch, y_batch)
        # Forward pass
        logits = model(x_batch)

        # Predictions
        predictions = tf.math.argmax(tf.nn.softmax(logits), axis=-1)
        pred_sequences.extend(predictions.numpy().tolist())

    left_trim = (
        "<BOS>" if emb_fr_obj.word_to_idx.get("<BOS>") in true_sequences[0] else None
    )
    right_trim = (
        "<EOS>" if emb_fr_obj.word_to_idx.get("<EOS>") in true_sequences[0] else None
    )

    _, x_sentences_path = tempfile.mkstemp()

    x_sentences = emb_en_obj.convert_idx_to_word(
        x_sequences,
        filepath=x_sentences_path,
        left_trim=left_trim,
        right_trim=right_trim,
    )
    _, pred_sentences_path = tempfile.mkstemp()
    pred_sentences = emb_fr_obj.convert_idx_to_word(
        pred_sequences,
        filepath=pred_sentences_path,
        left_trim=left_trim,
        right_trim=right_trim,
    )
    _, true_sentences_path = tempfile.mkstemp()
    true_sentences = emb_fr_obj.convert_idx_to_word(
        true_sequences,
        filepath=true_sentences_path,
        left_trim=left_trim,
        right_trim=right_trim,
    )
    _, x_sentences_path = tempfile.mkstemp()
    bleu = get_bleu(true_sentences_path, pred_sentences_path, True)

    df = pd.DataFrame(
        {
            "Input": x_sentences,
            "Prediction_out": pred_sentences,
            "True_out": true_sentences,
            "Bleu": bleu,
        }
    )

    ################# MODIFY HERE TO ADD STATS ################
    df = run_stats(df)

    ################# MODIFY HERE TO ADD STATS ################

    # Save raw data in run_name/stats
    stats_path = os.path.join(current_file_dir, run_name, "stats")
    os.makedirs(stats_path, exist_ok=True)
    df.to_csv(os.path.join(stats_path, "raw_data.csv"))


def main(run_name):
    ################# LOAD CONFIG  ###################
    print(f"Predict: Loading config/model from run {run_name}")
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), run_name, "config.json"
    )
    assert os.path.isfile(
        config_path
    ), f"Predict: Missing model config.json: {config_path}"
    with open(config_path, "r") as fd:
        config = json.load(fd)
    config["model"]["load_model"] = True
    config["model"]["wandb_run_to_load"] = run_name
    run_predict(config, run_name)


if __name__ == "__main__":
    fire.Fire(main)
