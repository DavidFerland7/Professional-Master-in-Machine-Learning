# TODO:
# 1) create dummy validation set + monitor training on validation set (for early stopping and log the metrics on valiation set)
# 2) connect training runs to weights and biaises
# 3) define the hyperparameter search workflow
# 4) create dummy test set + test prediction function to be evaluated by TA's evaluation function
# 5) add callback: checkpoints
# 6) Create a .sh file (bash file) to launch training script on compute note + try the whole training process on gpu with dummy small dataset


# TODO: make sure master branch is seeded  ref: https://machinelearningmastery.com/reproducible-results-neural-networks-keras/

NUMPY_SEED = 42
TENSORFLOW_SEED = 24

from numpy.random import seed
seed(NUMPY_SEED)
import tensorflow as tf
tf.random.set_seed(TENSORFLOW_SEED)

import argparse
import datetime
import json
import typing
import sys
import os
import re
import pandas as pd
import numpy as np
from models.utils.metrics import *
from models.utils.losses import *
import importlib
import copy
from dataloader import prepare_dataloader
from datetime import timedelta
import wandb
from wandb.tensorflow import WandbHook
from wandb.keras import WandbCallback
import utils as util
import pickle
import shutil
import hashlib


def build_model(data, config):
    config_full = copy.deepcopy(config)
    config = config['train_model']

    # import module and instantiate object
    module = importlib.import_module('models.' + config_full['globals']['model'] + '.' + config_full['globals']['model'])
    model_obj = getattr(module, config_full['globals']['model'])()

    # print model in table form
    if config_full['globals']['prints']:
        print(model_obj.build_graph(data).summary())

    # save model architecture to file - table form (with # of parameters)
    orig_stdout = sys.stdout
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/' + config_full['globals']['model'] + '/model.txt', 'w')
    sys.stdout = f
    print(model_obj.build_graph(data).summary())
    sys.stdout = orig_stdout
    f.close()

    # save model architecture to file - graph form
    tf.keras.utils.plot_model(model_obj.build_graph(data),
                              to_file=os.path.dirname(os.path.abspath(__file__)) + '/models/' + config_full['globals']['model'] + '/model.png',
                              show_shapes=True)
    return model_obj


def compile_model(model_obj, config):
    config_full = copy.deepcopy(config)
    config = config['train_model']
    if config['optim'] == 'adam':
        model_obj.compile(
            loss=WeightedMeanSquaredError(),
            optimizer=tf.optimizers.Adam(lr=config['init_lr'])
        )
    elif config['optim'] == 'sgd':
        model_obj.compile(
            loss=WeightedMeanSquaredError(),
            optimizer=tf.optimizers.SGD(lr=config['init_lr'])
        )
    return model_obj


def train_model(dataset_train, dataset_valid, model_obj, config, wandb_filename):
    config_full = copy.deepcopy(config)
    config = config['train_model']

    # initialize the optimizer and compile the model
    model_obj = compile_model(model_obj, config_full)

    # Define path to save checkpoints
    model_path = os.path.dirname(os.path.abspath(__file__)) + '/models/' + config_full['globals']['model'] + '/' + wandb_filename + '/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # save model file(.py) in model/wandbname folder
    shutil.copy(
        os.path.dirname(os.path.abspath(__file__)) + '/models/' + config_full['globals']['model'] + '/' + config_full['globals']['model'] + '.py',
        model_path + '/' + config_full['globals']['model'] + '.py'
    )

    # DUMP data dummy for reloading at test time
    dataset_train2 = dataset_train
    with open(model_path + 'single_batch.pickle', 'wb') as f:
        data = (next(iter(dataset_train2)), tf.compat.v1.data.get_output_types(dataset_train2), tf.compat.v1.data.get_output_shapes(dataset_train2))
        pickle.dump(data, f)

    # Setup tensorboard callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # callbacks definition
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', min_delta=0, mode='min', restore_best_weights=True)
    callbacks = [
        early_stopping,
        tf.keras.callbacks.ModelCheckpoint(model_path + 'best-weights.{epoch:02d}-{val_loss:.2f}.tf', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True),
        util.WandbExtraLogs(),
        WandbCallback(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch'),
    ]

    print("[INFO] training network...")
    model_obj.fit_generator(dataset_train, validation_data=dataset_valid, epochs=config['num_epochs'], shuffle=False, verbose=1, callbacks=callbacks)

    stopped_epoch = config['num_epochs'] if early_stopping.stopped_epoch == 0 else early_stopping.stopped_epoch - early_stopping.patience + 1
    print("best MSE score={}; from epoch={}".format(early_stopping.best, stopped_epoch))

    return model_obj


def instantiate_model(config):
    config_full = copy.deepcopy(config)
    module = importlib.import_module('models.' + config_full['globals']['model'] + '.' + config_full['load_model']['load_wandb_dir'] + '.' + config_full['globals']['model'])
    return getattr(module, config_full['globals']['model'])()


def get_weights_filename(path, config):
    config_full = copy.deepcopy(config)
    config = config['load_model']

    if config["load_best_inside_run"]:
        filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) if 'best-weights' in f]
        scores = [re.search(r'(.*-)(.*)(\.tf.*)', f).group(2) for f in filenames]
        best_score = str(min(float(sub) for sub in scores))
        weights_filename = [re.search(r'(.*-)({0})(\.tf.*)'.format(best_score), f) for f in filenames]
        weights_filename = list(set([f.group(1) + f.group(2) for f in weights_filename if f is not None]))[0]
    else:
        weights_filename = config['load_filename']

    return weights_filename


def load_model(config):
    config_full = copy.deepcopy(config)

    model_path = os.path.dirname(os.path.abspath(__file__)) + '/models/' + config_full['globals']['model'] + '/' + config_full['load_model']['load_wandb_dir'] + '/'

    # Recreate the model
    model_obj = instantiate_model(config_full)
    model_obj = compile_model(model_obj, config_full)

    with open(model_path + 'single_batch.pickle', 'rb') as f:
        data = pickle.load(f)
    batch, output_types, output_shapes = data

    def generator():
        yield batch

    data = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

    # Also save the loss on the first batch
    # to later assert that the optimizer state was preserved
    # model_obj.train_on_batch(dataset.take(1))
    model_obj.train_on_batch(data)

    # Define file name to load
    weights_filename = get_weights_filename(model_path, config_full)
    model_obj.load_weights(model_path + weights_filename + '.tf')

    return model_obj


def main(
        preds_output_path: typing.AnyStr,
        admin_config_path: typing.AnyStr,
        user_config_path: typing.Optional[typing.AnyStr] = None,
        stats_output_path: typing.Optional[typing.AnyStr] = None,
) -> None:
    """Extracts predictions from a user model/data loader combo and saves them to a CSV file."""

    user_config = {}
    if user_config_path:
        assert os.path.isfile(user_config_path), f"invalid user config file: {user_config_path}"
        with open(user_config_path, "r") as fd:
            user_config = json.load(fd)

    assert os.path.isfile(admin_config_path), f"invalid admin config file: {admin_config_path}"
    with open(admin_config_path, "r") as fd:
        admin_config = json.load(fd)

    assert user_config['globals']['training'], f"This file should be run in training only (and globals/training is false in user_config)"

    path_cache = f'/./data/cache/' + str(os.environ['USER']) + '/'
    dict_to_hash = {
        'data': user_config['data'],
        'train_start_bound': user_config['train_model']['train_start_bound'],
        'train_end_bound': user_config['train_model']['train_end_bound'],
        'valid_start_bound': user_config['train_model']['valid_start_bound'],
        'valid_end_bound': user_config['train_model']['valid_end_bound']
    }
    config_hash = hashlib.md5(json.dumps(dict_to_hash).encode("utf-8")).hexdigest()
    if not os.path.isdir(path_cache + config_hash):
        print("Creating cache directory with hash: ", str(config_hash))
        os.mkdir(path_cache + config_hash)
    else:
        print("Cache directory already exists with hash: ", str(config_hash))

    # Initialize Wandb experiment tracking MUST BE LOGGED IN (wandb login "API_KEY")
    os.environ['WANDB_MODE'] = 'dryrun'
    run = wandb.init(project="spaceeye", entity='david', sync_tensorboard=True, config=user_config)
    wandb.run.summary["tensorflow_seed"] = TENSORFLOW_SEED
    wandb.run.summary["numpy_seed"] = NUMPY_SEED
    for k, v in admin_config.items():
        wandb.run.summary["admin_config_" + k] = v

    dataframe_path = admin_config["dataframe_path"]
    assert os.path.isfile(dataframe_path), f"invalid dataframe path: {dataframe_path}"
    dataframe = pd.read_pickle(dataframe_path)

    if "start_bound" in admin_config:
        dataframe = dataframe[dataframe.index >= datetime.datetime.fromisoformat(admin_config["start_bound"])]
    if "end_bound" in admin_config:
        dataframe = dataframe[dataframe.index < datetime.datetime.fromisoformat(admin_config["end_bound"])]

    target_datetimes = [datetime.datetime.fromisoformat(d) for d in admin_config["target_datetimes"]]
    assert target_datetimes and all([d in dataframe.index for d in target_datetimes])
    target_stations = admin_config["stations"]
    target_time_offsets = [pd.Timedelta(d).to_pytimedelta() for d in admin_config["target_time_offsets"]]

    # Get all images if they don't exist in cache
    region_size = user_config['data']['imgs']['region_size']
    cache_folder = f'/./data/imgs_cropped/{region_size}/'
    if not os.path.isdir(cache_folder):
        util.get_all_imgs(target_stations, dataframe.dropna(subset=['ncdf_path']), user_config['data']['imgs']['region_size'], user_config['data']['imgs']['n_channels'], user_config['globals']['training'], user_config['globals']['prints'])

    ############################### TRAINING #################################
    #df_paths, df = preprocess_dataframe(dataframe, user_config)
    dataset_valid = None
    df = dataframe

    df_train = df[df.index >= datetime.datetime.fromisoformat(user_config['train_model']["train_start_bound"])]
    df_train = df_train[df_train.index < datetime.datetime.fromisoformat(user_config['train_model']["train_end_bound"])]

    dataset_train = prepare_dataloader(df_train, df_train.index.to_pydatetime(), target_stations, target_time_offsets, user_config)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE).batch(user_config['data']['imgs']['batch_size'], drop_remainder=True).cache(path_cache + f'{config_hash}/train_cache')  # loss calculated is wrong on last minibatch when uneven size

    if user_config['train_model']['split_valid']:
        df_valid = df[df.index >= datetime.datetime.fromisoformat(user_config['train_model']["valid_start_bound"])]
        df_valid = df_valid[df_valid.index < datetime.datetime.fromisoformat(user_config['train_model']["valid_end_bound"])]

        dataset_valid = prepare_dataloader(df_valid, df_valid.index.to_pydatetime(), target_stations, target_time_offsets, user_config)
        dataset_valid = dataset_valid.prefetch(tf.data.experimental.AUTOTUNE).batch(user_config['data']['imgs']['batch_size'], drop_remainder=True).cache(path_cache + f'{config_hash}/valid_cache')  # loss calculated is wrong on last minibatch when uneven size

    model_obj = build_model(dataset_train, user_config)
    model_obj = train_model(dataset_train, dataset_valid, model_obj, user_config, run.dir.split('/')[-1])

    # Code below to load back the model and print evaluate on chosen dataset (here its valid dataset just to validate its the same loss as the best one during training)
    #loaded_model = load_model(user_config)
    #print("\nevaluate with LOADED model:\n{}".format(loaded_model.evaluate(dataset_valid)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_out_path", type=str,
                        help="path where the raw model predictions should be saved (for visualization purposes)")
    parser.add_argument("admin_cfg_path", type=str,
                        help="path to the JSON config file used to store test set/evaluation parameters")
    parser.add_argument("-u", "--user_cfg_path", type=str, default=None,
                        help="path to the JSON config file used to store user model/dataloader parameters")
    parser.add_argument("-s", "--stats_output_path", type=str, default=None,
                        help="path where the prediction stats should be saved (for benchmarking)")
    args = parser.parse_args()
    main(
        preds_output_path=args.preds_out_path,
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        stats_output_path=args.stats_output_path,
    )
