import argparse
import subprocess
import tempfile


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """

    import os
    import json
    import sys
    from preprocessing.TxtPreprocessor import run_txt_preprocessor
    import globals as globals_vars
    import importlib
    import tensorflow as tf
    from tqdm import tqdm
    import numpy as np
    import pandas as pd

    # TODO SPECIFY MODEL TO LOAD IN GLOBALS

    ################# LOAD BEST CONFIGS  ###################
    model_run_path = os.path.join(
        "models", globals_vars.TEST_MODEL_NAME, globals_vars.TEST_RUN_NAME
    )
    with open(
        os.path.join(globals_vars.TEST_PATH_SERVER, model_run_path, "config.json")
    ) as f:
        config_model = json.load(f)

    # TODO SET RUN PATH TO LOAD in <MODEL>/config.json OR HERE
    config_model["model"]["load_model"] = True
    config_model["model"]["wandb_run_to_load"] = globals_vars.TEST_RUN_NAME

    # Get embedding from specified model's config file
    embedding_path = os.path.join(
        globals_vars.TEST_PATH_SERVER, model_run_path, "embedding", "config.json"
    )
    print("embedding_path: ", embedding_path)
    ################# PREPROCESS INPUT FILE  ###################
    df_test = run_txt_preprocessor(
        {"preprocessing": config_model["preprocessing"]},
        "raw_aligned",
        "en",
        dst_path=globals_vars.TEST_PATH_SERVER,
        path_input_txt=input_file_path,
    )

    # Reindexing by sentence length to have efficient batch
    sorted_index = (
        np.maximum((df_test.text.str.count(" ") + 1), (df_test.text.str.count(" ") + 1))
        .sort_values()
        .index
    )
    df_test = df_test.reindex(sorted_index)

    ################# EMBEDDINGS ###################
    # Load embedding models
    embedding_module = importlib.import_module(
        "embedding." + config_model["embedding"] + "." + "model"
    )
    emb_en_obj = getattr(embedding_module, config_model["embedding"])(
        "en", embedding_path
    )
    emb_fr_obj = getattr(embedding_module, config_model["embedding"])(
        "fr", embedding_path
    )
    # FR: load model only to have convert_to_idx method
    emb_fr_obj._load_model()

    # EN: Generate embedding for english sentences to translate
    x_test = tf.ragged.constant(emb_en_obj.generate(df_test))

    ################# PREPARE DATASETS ###################
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(
        # config_model["model"]["batch_size"]
        32
    )

    ################# MODEL SETUP  ###################
    model_train_module = importlib.import_module(
        "models." + globals_vars.TEST_MODEL_NAME + "." + "train"
    )
    model_utils_module = importlib.import_module("models.utils.train_utils")

    # TODO SET POSITIONAL ARGS ACCORDING TO <MODEL>/TRAIN.py
    model = getattr(model_utils_module, "instantiate_load_model")(
        config_model["model"],
        globals_vars.TEST_MODEL_NAME,
        [emb_en_obj, emb_fr_obj],
        config_model["model"]["model_kwargs"],
    )

    pad_batch_sequences = getattr(model_utils_module, "pad_batch_sequences")
    predict = getattr(model_train_module, "predict")

    x_sequences = []
    pred_sequences = []
    for step, x_batch_test in tqdm(
        enumerate(test_dataset), desc="Generating predictions..."
    ):
        x_sequences.extend(tf.cast(x_batch_test, dtype=tf.int32).to_list())

        # Padding batch of sequences
        x_batch_test, _ = pad_batch_sequences(
            x_batch_test, x_batch_test, training=False, factor_pad=4
        )
        # Forward pass
        x_batch_test = tf.squeeze(x_batch_test)
        logits = model.evaluate(x_batch_test)
        # Predictions
        pred_sequences.extend(predict(logits).numpy().tolist())

    # replace data in original order
    df_test = pd.DataFrame({"text": pred_sequences}, index=sorted_index)
    df_test = df_test.sort_index()
    pred_sequences = df_test.text.values.tolist()

    # convert tokens id to token words
    left_trim = (
        "<BOS>" if emb_en_obj.word_to_idx.get("<BOS>") in x_sequences[0] else None
    )
    right_trim = (
        "<EOS>" if emb_en_obj.word_to_idx.get("<EOS>") in x_sequences[0] else None
    )

    pred_sentences = emb_fr_obj.convert_idx_to_word(
        pred_sequences,
        filepath=pred_file_path,
        left_trim=left_trim,
        right_trim=right_trim,
    )


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """
    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.
    Returns: None
    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def get_bleu(pred_file_path: str, target_file_path: str, return_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(
        [
            "sacrebleu",
            "--input",
            pred_file_path,
            target_file_path,
            "--tokenize",
            "none",
            "--sentence-level",
            "--score-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = out.stdout.split("\n")
    scores = [float(x) for x in lines[:-1]]

    print("final avg bleu score: {:.2f}".format(sum(scores) / len(scores)))
    if return_all_scores:
        return scores
    else:
        return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()
