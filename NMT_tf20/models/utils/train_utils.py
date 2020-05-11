import json
import pathlib
import sys
import os
import pandas as pd

current_file_dir = os.path.dirname(os.path.abspath(__file__))
import tensorflow.keras as keras
import tensorflow as tf
import importlib
import numpy as np
import wandb
import shutil

sys.path.insert(0, os.path.join(current_file_dir, "..", ".."))
from preprocessing.TxtPreprocessor import run_txt_preprocessor


class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    @classmethod
    def create_masks(cls, inputs, target):
        encoder_padding_mask = Mask.create_padding_mask(inputs)
        decoder_padding_mask = Mask.create_padding_mask(inputs)

        look_ahead_mask = tf.maximum(
            Mask.create_look_ahead_mask(tf.shape(target)[1]),
            Mask.create_padding_mask(target),
        )

        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


loss_object = tf.losses.CategoricalCrossentropy(from_logits=True, reduction="none")


def label_smoothing(target_data, depth, epsilon=0.1):
    target_data_one_hot = tf.one_hot(target_data, depth=depth)
    n = target_data_one_hot.get_shape().as_list()[-1]
    return ((1 - epsilon) * target_data_one_hot) + (epsilon / n)


def smoothed_sequence_softmax_cross_entropy_with_logits(real, pred, vocab_size):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real_one_hot = label_smoothing(real, depth=vocab_size)
    loss = loss_object(real_one_hot, pred)

    mask = tf.cast(mask, dtype=loss.dtype)

    loss *= mask
    return tf.reduce_mean(loss)


def sequence_softmax_cross_entropy_with_logits(output, target, mask_pad=True):
    assert len(output.shape) == 3
    assert len(target.shape) == 3
    assert target.shape[-1] == 1
    labels = tf.reshape(target, (-1,))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=tf.reshape(output, (-1, output.shape[2])),
    )
    if mask_pad:
        mask = tf.where(
            tf.greater(labels, 0),
            tf.cast(labels, dtype=tf.float32),
            tf.zeros_like(labels, dtype=tf.float32),
        )
        cross_entropy *= mask
    cross_entropy = tf.reshape(cross_entropy, (output.shape[0], output.shape[1]),)
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    return tf.reduce_mean(cross_entropy)


def masked_sequence_softmax_cross_entropy_with_logits(output, target):
    assert len(output.shape) == 3
    assert len(target.shape) == 3
    assert target.shape[-1] == 1
    labels = tf.reshape(target, (-1,))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=tf.reshape(output, (-1, output.shape[2])),
    )
    mask = tf.where(
        tf.greater(labels, 0),
        tf.cast(labels, dtype=tf.float32),
        tf.zeros_like(labels, dtype=tf.float32),
    )
    cross_entropy *= mask
    cross_entropy = tf.reshape(cross_entropy, (output.shape[0], output.shape[1]),)
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    return tf.reduce_mean(cross_entropy)


def masked_softmax_cross_entropy_with_logits(output, target):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target, logits=output,
    )
    mask = tf.where(
        tf.greater(target, 0),
        tf.cast(target, dtype=tf.float32),
        tf.zeros_like(target, dtype=tf.float32),
    )
    cross_entropy *= mask
    return tf.reduce_mean(cross_entropy)


def masked_smoothed_softmax_cross_entropy_with_logits(output, target, smoothing):
    mask = tf.where(
        tf.greater(target, 0),
        tf.cast(target, dtype=tf.float32),
        tf.zeros_like(target, dtype=tf.float32),
    )
    # smooth_target = (1 - smoothing) * target
    smooth_target = smoothing * tf.zeros_like(output, dtype=tf.float32)
    labels = tf.one_hot(target, output.shape[1], axis=-1)
    smooth_target = tf.where(
        tf.greater(labels, 0),
        (1 - smoothing) * tf.ones_like(labels, dtype=tf.float32),
        smoothing * tf.ones_like(labels, dtype=tf.float32),
    )
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=smooth_target, logits=output,
    )
    cross_entropy *= mask
    return tf.reduce_mean(cross_entropy)


def pad_batch_sequences(x_batch, y_batch, training=True, factor_pad=3):
    if training:
        if len(x_batch.shape) == 2:
            x_batch = tf.expand_dims(x_batch, -1)
        if len(y_batch.shape) == 2:
            y_batch = tf.expand_dims(y_batch, -1)
        batch_stack = x_batch.to_list() + y_batch.to_list()
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            batch_stack, padding="post",
        )
        x_batch = tf.cast(padded_inputs[: len(batch_stack) // 2], dtype=tf.float32)
        y_batch = tf.cast(padded_inputs[len(batch_stack) // 2 :], dtype=tf.int32)

    # This case is during inference where we don't know the shape of y so we pad with factor time longest lenght of x
    else:
        if len(x_batch.shape) == 2:
            x_batch = tf.expand_dims(x_batch, -1)
        if len(y_batch.shape) == 2:
            y_batch = tf.expand_dims(y_batch, -1)

        padded_inputs_x = tf.keras.preprocessing.sequence.pad_sequences(
            x_batch.to_list(), padding="post"
        )  # get obs in x_batch the same (the longest of x)

        padded_inputs_x = tf.keras.preprocessing.sequence.pad_sequences(
            padded_inputs_x.tolist(),
            padding="post",
            maxlen=round(padded_inputs_x.shape[1] * factor_pad),
        )  # pad x_batch with factor

        padded_inputs_y = tf.keras.preprocessing.sequence.pad_sequences(
            y_batch.to_list(), padding="post", maxlen=padded_inputs_x.shape[1]
        )  # pad y_batch as x_batch

        x_batch = tf.cast(padded_inputs_x, dtype=tf.float32)
        y_batch = tf.cast(padded_inputs_y, dtype=tf.int32)

    return (x_batch, y_batch)


def pad_batch_sequences_embedding(x_batch_emb, y_batch_emb, y_batch):
    batch_stack = x_batch_emb.to_list() + y_batch_emb.to_list() + y_batch.to_list()
    max_len = max([len(sequence) for sequence in batch_stack])
    emb_dim = len(x_batch_emb.to_list()[0][0])
    padded_sequences = []
    for seq in x_batch_emb.to_list():
        diff = int(max_len - len(seq))
        padded_sequences.append(tf.pad(tf.constant(seq), [[0, diff], [0, 0]]))
    x_batch_emb = tf.stack(padded_sequences)
    padded_sequences = []
    for seq in y_batch_emb.to_list():
        diff = int(max_len - len(seq))
        padded_sequences.append(tf.pad(tf.constant(seq), [[0, diff], [0, 0]]))
    y_batch_emb = tf.stack(padded_sequences)
    padded_sequences = []
    for seq in y_batch.to_list():
        diff = int(max_len - len(seq))
        padded_sequences.append(tf.pad(tf.constant(seq), [[0, diff]]))
    y_batch = tf.stack(padded_sequences)
    return (x_batch_emb, y_batch_emb, y_batch)


def get_preprocessed_dfs(config):
    # Preprocess aligned datasets
    df_en_train, df_en_valid = run_txt_preprocessor(
        {"preprocessing": config["preprocessing"]}, "raw_aligned", "en"
    )
    df_fr_train, df_fr_valid = run_txt_preprocessor(
        {"preprocessing": config["preprocessing"]}, "raw_aligned", "fr"
    )

    # Very useful print to always make sure we have all rows correctly
    print("Training:  quick validation below (DO NOT DELETE) ##############")
    print("df_en_train.shape: ", df_en_train.shape)
    print("df_en_valid.shape: ", df_en_valid.shape)
    print("df_fr_train.shape: ", df_fr_train.shape)
    print("df_fr_valid.shape: ", df_fr_valid.shape)
    df_concat = pd.concat([df_en_train, df_en_valid, df_fr_train, df_fr_valid])
    print(
        "max_seq across all 4 datas: {}\n".format(
            (df_concat.text.str.count(" ") + 1).max()
        )
    )

    # Reindexing by sentence length to have efficient batch
    sorted_index_train = (
        np.maximum(
            (df_en_train.text.str.count(" ") + 1), (df_fr_train.text.str.count(" ") + 1)
        )
        .sort_values()
        .index
    )
    sorted_index_valid = (
        np.maximum(
            (df_en_valid.text.str.count(" ") + 1), (df_fr_valid.text.str.count(" ") + 1)
        )
        .sort_values()
        .index
    )
    df_en_train = df_en_train.reindex(sorted_index_train)
    df_en_valid = df_en_valid.reindex(sorted_index_valid)
    df_fr_train = df_fr_train.reindex(sorted_index_train)
    df_fr_valid = df_fr_valid.reindex(sorted_index_valid)
    return (df_en_train, df_en_valid, df_fr_train, df_fr_valid)


# Load only if specified in config
def instantiate_load_model(model_config, model_name, args=[], kwargs={}):
    model_dir = os.path.abspath(os.path.join(current_file_dir, "..", model_name))
    module = importlib.import_module(
        f"...{os.path.basename(model_name)}." + "model", package=__name__
    )
    model = getattr(module, model_config["model_class"])(*args, **kwargs)
    if model_config["load_model"]:
        if type(model_config["wandb_run_to_load"]) == str and os.path.isfile(
            os.path.join(
                model_dir,
                model_config["wandb_run_to_load"],
                "best_model_weights.index",
            )
        ):
            print(
                f"Loading already trained model's weights from run: {model_config['wandb_run_to_load']}"
            )
            model.load_weights(
                os.path.join(
                    model_dir, model_config["wandb_run_to_load"], "best_model_weights",
                ),
            )

        else:
            raise ValueError(
                f"No trained weights found from run: {model_config['wandb_run_to_load']}"
            )
    return model


def get_model(model_config, model_name, args=[], kwargs={}):
    model_dir = os.path.join(current_file_dir, "..", model_name)
    model = instantiate_load_model(model_config, model_name, args, kwargs)
    best_valid_bleu = 0
    start_epoch = 0

    if model_config["load_model"]:
        if type(model_config["wandb_run_to_load"]) == str and os.path.isfile(
            os.path.join(
                model_dir,
                model_config["wandb_run_to_load"],
                "best_model_weights.index",
            )
        ):
            with open(
                os.path.join(
                    model_dir, model_config["wandb_run_to_load"], "run_logs.json",
                )
            ) as f:
                logs = json.load(f)
            best_valid_bleu, start_epoch = (logs["best_valid_bleu"], logs["epoch"] + 1)
            if wandb.run is not None:
                wandb.log(
                    {
                        "train_loss": logs["train_loss"],
                        "valid_loss": logs["valid_loss"],
                        "epoch": logs["epoch"],
                    }
                )
        else:
            raise ValueError(
                f"No trained weights found from run: {model_config['wandb_run_to_load']}"
            )
    return (model, best_valid_bleu, start_epoch)


def save_model(config, model_file_dir, logs, model):
    wandb_run_dir_name = os.path.split(wandb.run._dir)[-1]
    model.save_weights(
        os.path.join(wandb_run_dir_name, "best_model_weights"), save_format="tf"
    )
    with open(os.path.join(wandb_run_dir_name, "run_logs.json"), "w") as f:
        json.dump(logs, f)
    with open(os.path.join(wandb_run_dir_name, "config.json"), "w") as f:
        json.dump(config, f)

    # Save model files
    shutil.copy(
        os.path.join(model_file_dir, "train.py"),
        os.path.join(model_file_dir, wandb_run_dir_name, "train.py"),
    )
    shutil.copy(
        os.path.join(model_file_dir, "model.py"),
        os.path.join(model_file_dir, wandb_run_dir_name, "model.py"),
    )

    # Save embedding files
    embedding_path = os.path.join(
        model_file_dir, "..", "..", "embedding", config["embedding"],
    )
    shutil.copy(
        os.path.join(embedding_path, "config.json"),
        os.path.join(model_file_dir, wandb_run_dir_name, "embedding"),
    )


def save_models(config, model_file_dir, logs, named_models_dict):
    wandb_run_dir_name = os.path.split(wandb.run._dir)[-1]
    for name, model in named_models_dict.items():
        model.save_weights(
            os.path.join(wandb_run_dir_name, f"best_{name}_weights"), save_format="tf"
        )
    with open(os.path.join(wandb_run_dir_name, "run_logs.json"), "w") as f:
        json.dump(logs, f)
    with open(os.path.join(wandb_run_dir_name, "config.json"), "w") as f:
        json.dump(config, f)

    # Save model files
    shutil.copy(
        os.path.join(model_file_dir, "train.py"),
        os.path.join(model_file_dir, wandb_run_dir_name, "train.py"),
    )
    shutil.copy(
        os.path.join(model_file_dir, "model.py"),
        os.path.join(model_file_dir, wandb_run_dir_name, "model.py"),
    )

    # Save embedding files
    embedding_path = os.path.join(
        model_file_dir, "..", "..", "embedding", config["embedding"],
    )
    shutil.copy(
        os.path.join(embedding_path, "config.json"),
        os.path.join(model_file_dir, wandb_run_dir_name, "embedding"),
    )
