import sys
import os

NUMPY_SEED = 24
TENSORFLOW_SEED = 42
from numpy.random import seed

seed(NUMPY_SEED)
import tensorflow as tf

tf.random.set_seed(TENSORFLOW_SEED)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import json
import importlib
import wandb
from tqdm import tqdm
from os.path import relpath
import time
from numba import cuda

model_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[
    -1
]  # e.g.: GRU_1to1_Onehot

################# UTILS FUNCTIONS ###################
from models.utils.train_utils import *
from evaluator import get_bleu


def predict(logits):
    return tf.math.argmax(tf.nn.softmax(logits), axis=-1)


def run_train(config):
    ################# SETUP  ###################
    wandb_run_dir_name = os.path.split(wandb.run._dir)[-1]
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_file_dir, wandb_run_dir_name))
    os.mkdir(os.path.join(current_file_dir, wandb_run_dir_name, "embedding"))

    ################# PREPROCESSINGS  ###################
    df_en_train, df_en_valid, df_fr_train, df_fr_valid = get_preprocessed_dfs(config)
    num_obs_train = df_en_train.shape[0]
    num_obs_valid = df_en_valid.shape[0]

    ################# EMBEDDINGS ###################
    # Load embedding models
    embedding_module = importlib.import_module(
        "embedding." + config["embedding"] + "." + "model"
    )
    emb_en_obj = getattr(embedding_module, config["embedding"])("en")
    emb_fr_obj = getattr(embedding_module, config["embedding"])("fr")

    # Generate embedding
    start_time_ = time.perf_counter()
    x_train = tf.ragged.constant(emb_en_obj.generate(df_en_train))
    x_valid = tf.ragged.constant(emb_en_obj.generate(df_en_valid))
    y_train = tf.ragged.constant(emb_fr_obj.generate(df_fr_train))
    y_valid = tf.ragged.constant(emb_fr_obj.generate(df_fr_valid))
    print("time to generate: ", time.perf_counter() - start_time_)

    ################# PREPARE DATASETS ###################
    y_train_true = tf.ragged.constant(df_fr_train.text.values.tolist())
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train, y_train_true))
        .batch(config["model"]["batch_size"])
        .shuffle(buffer_size=1024)
    )

    y_valid_true = tf.ragged.constant(df_fr_valid.text.values.tolist())
    valid_dataset = (
        tf.data.Dataset.from_tensor_slices((x_valid, y_valid, y_valid_true))
        .batch(config["model"]["batch_size_valid"])
        .shuffle(buffer_size=1024)
    )

    ################# MODEL SETUP  ###################
    model, best_valid_bleu, start_epoch = get_model(
        config["model"],
        model_name,
        [emb_en_obj, emb_fr_obj],
        config["model"]["model_kwargs"],
    )

    # Instantiate optimizer given config
    learning_rate = CustomSchedule(config["model"]["model_kwargs"]["d_model"])
    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    ################# TRAINING LOOP ###################
    for epoch in range(start_epoch, config["model"]["epochs"]):

        epoch_train_loss, epoch_valid_loss = (0, 0)
        pred_train_sequences, pred_valid_sequences = [], []
        true_train_sequences, true_valid_sequences = [], []

        ################# TRAIN ###################
        for step, (x_batch_train, y_batch_train, y_true_train) in tqdm(
            enumerate(train_dataset),
            total=int(num_obs_train / config["model"]["batch_size"]),
            desc="Epoch {d}: Training on batch".format(d=epoch),
        ):

            ### Add Padding ###
            x_batch_train, y_batch_train = pad_batch_sequences(
                x_batch_train, y_batch_train
            )

            x_batch_train = tf.squeeze(x_batch_train)
            y_batch_train = tf.squeeze(y_batch_train)
            target_with_BOS = y_batch_train[:, :-1]
            target_with_EOS = y_batch_train[:, 1:]

            (
                encoder_padding_mask,
                look_ahead_mask,
                decoder_padding_mask,
            ) = Mask.create_masks(x_batch_train, target_with_BOS)

            # Do forward pass and record gradient
            with tf.GradientTape() as tape:

                logits = model.call(
                    inputs=x_batch_train,
                    target=target_with_BOS,
                    inputs_padding_mask=encoder_padding_mask,
                    look_ahead_mask=look_ahead_mask,
                    target_padding_mask=decoder_padding_mask,
                    training=True,
                )

                # Logits for this minibatch
                loss_value = smoothed_sequence_softmax_cross_entropy_with_logits(
                    target_with_EOS, logits, emb_fr_obj.vocab_size
                )
                epoch_train_loss += loss_value

            # Take a step with optimizer
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Predictions
            predictions = tf.math.argmax(tf.nn.softmax(logits), axis=-1)
            pred_train_sequences.extend(predictions.numpy().tolist())

            true_train_sequences.extend(
                [x.decode("utf-8") for x in y_true_train.numpy().tolist()]
            )
        # average loss over number of batches
        epoch_train_loss /= step + 1

        ################# VALID ###################
        for step, (x_batch_valid, y_batch_valid, y_true_valid) in tqdm(
            enumerate(valid_dataset),
            total=int(num_obs_valid / config["model"]["batch_size_valid"]),
            desc="Epoch {d}: Validation on batch".format(d=epoch),
        ):
            # Padding batch of sequences
            x_batch_valid, y_batch_valid = pad_batch_sequences(
                x_batch_valid,
                y_batch_valid,
                training=False,
                factor_pad=config["model"]["factor_pad"],
            )

            x_batch_valid = tf.squeeze(x_batch_valid)
            y_batch_valid = tf.squeeze(y_batch_valid)

            logits = model.evaluate(x_batch_valid)

            loss_value = smoothed_sequence_softmax_cross_entropy_with_logits(
                y_batch_valid, logits, emb_fr_obj.vocab_size
            )
            epoch_valid_loss += loss_value
            # Predictions
            predictions = tf.math.argmax(tf.nn.softmax(logits), axis=-1)
            pred_valid_sequences.extend(predictions.numpy().tolist())
            true_valid_sequences.extend(
                [x.decode("utf-8") for x in y_true_valid.numpy().tolist()]
            )

        epoch_valid_loss /= step + 1

        print(
            "Train loss= {}   ||   Validation loss= {}".format(
                epoch_train_loss, epoch_valid_loss
            )
        )

        true_train_sentences_path = os.path.join(
            current_file_dir, wandb_run_dir_name, "true_train_sentences"
        )
        pred_train_sentences_path = os.path.join(
            current_file_dir, wandb_run_dir_name, "pred_train_sentences"
        )
        true_valid_sentences_path = os.path.join(
            current_file_dir, wandb_run_dir_name, "true_valid_sentences"
        )
        pred_valid_sentences_path = os.path.join(
            current_file_dir, wandb_run_dir_name, "pred_valid_sentences"
        )

        left_trim = "<BOS>" if "<BOS>" in true_train_sequences[0] else None
        right_trim = "<EOS>" if "<EOS>" in true_train_sequences[0] else None

        emb_fr_obj.convert_idx_to_word(
            true_train_sequences,
            filepath=true_train_sentences_path,
            left_trim=left_trim,
            right_trim=right_trim,
            only_write_sent_to_file=True,
        )
        emb_fr_obj.convert_idx_to_word(
            pred_train_sequences,
            filepath=pred_train_sentences_path,
            left_trim=None,  ## model does not predict BOS is training
            right_trim=right_trim,
        )
        emb_fr_obj.convert_idx_to_word(
            true_valid_sequences,
            filepath=true_valid_sentences_path,
            left_trim=left_trim,
            right_trim=right_trim,
            only_write_sent_to_file=True,
        )
        emb_fr_obj.convert_idx_to_word(
            pred_valid_sequences,
            filepath=pred_valid_sentences_path,
            left_trim=left_trim,
            right_trim=right_trim,
        )

        bleu_train = get_bleu(
            relpath(true_train_sentences_path, os.getcwd()),
            relpath(pred_train_sentences_path, os.getcwd()),
            False,
        )
        bleu_valid = get_bleu(
            relpath(true_valid_sentences_path, os.getcwd()),
            relpath(pred_valid_sentences_path, os.getcwd()),
            False,
        )
        ################# LOGGING & SAVING ###################
        wandb.log(
            {
                "train_loss": epoch_train_loss.numpy(),
                "valid_loss": epoch_valid_loss.numpy(),
                "bleu_train": bleu_train,
                "bleu_valid": bleu_valid,
                "epoch": epoch,
            }
        )

        # Save best model on valid & run logs
        if bleu_valid > best_valid_bleu:
            best_valid_bleu = bleu_valid
            logs = {
                "train_loss": epoch_train_loss.numpy().item(),
                "valid_loss": epoch_valid_loss.numpy().item(),
                "bleu_train": bleu_train,
                "bleu_valid": bleu_valid,
                "epoch": epoch,
                "best_valid_bleu": best_valid_bleu,
            }
            save_model(config, current_file_dir, logs, model)

    ################# PREDICT + STATS ###################
    if config["model"]["output_stats"]:
        # Reset GPU memory before calling predict.py
        try:
            device = cuda.get_current_device()
            print("Following device will be reset:", device)
            device.reset()
        except:
            print("running on cpu, no reset required")

        # Call predict.py with <wandb_run_dir_name> as argument
        predict_cmd = f"python predict.py {wandb_run_dir_name}"
        return_value = os.system(predict_cmd)
        if return_value != 0:
            print(f"Predictions: {predict_cmd} failed")


if __name__ == "__main__":
    ################# LOAD CONFIG  ###################
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    assert os.path.isfile(config_path), f"missing embedding config.json: {config_path}"
    with open(config_path, "r") as fd:
        config = json.load(fd)
    os.environ["WANDB_MODE"] = "dryrun"
    run = wandb.init(
        project="elcheapo-translator",
        entity="ift6759",
        config=config,
        dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )
    wandb.run.summary["tensorflow_seed"] = TENSORFLOW_SEED
    wandb.run.summary["numpy_seed"] = NUMPY_SEED

    for k, v in config.items():
        wandb.run.summary["model_config_" + k] = v
    run_train(config)
