import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tensorflow as tf
from tensorflow.keras import layers
from models.BaseModel import BaseModel
from models.utils.TimeDistributedCustom import TimeDistributedCustom as TD
import numpy as np


class encoder(BaseModel):
    def __init__(self, emb=None, fine_tune_emb_input=False, rnn_size=1024, dropout=0.5):
        super().__init__()

        self.embedding_layer = self.build_pre_emb_layer(
            emb, fine_tune_emb_input=fine_tune_emb_input
        )

        ## encoder
        self.encoder_rnn1 = layers.GRU(
            rnn_size,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
            dropout=dropout,
        )
        self.encoder_rnn2 = layers.GRU(
            rnn_size, return_state=True, recurrent_initializer="glorot_uniform"
        )

    def call(self, x, training=False):

        x = tf.squeeze(x)

        x = self.embedding_layer(x)
        x = self.encoder_rnn1(x)
        encoder_outputs, enc_state = self.encoder_rnn2(x, training=training)

        return encoder_outputs, enc_state


class decoder(BaseModel):
    def __init__(self, emb=None, fine_tune_emb_input=False, rnn_size=1024, dropout=0.5):
        super().__init__()

        self.embedding_layer = self.build_pre_emb_layer(
            emb, fine_tune_emb_input=fine_tune_emb_input
        )
        self.out_vocab_size = emb.vocab_size
        self.word_to_idx = emb.word_to_idx
        self.decoder_rnn1 = layers.GRU(
            rnn_size,
            return_sequences=True,
            recurrent_initializer="glorot_uniform",
            dropout=dropout,
        )
        self.decoder_rnn2 = layers.GRU(
            rnn_size,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.decoder_dense1 = layers.Dense(rnn_size)
        self.dropout = layers.Dropout(dropout)
        self.decoder_dense2 = layers.Dense(self.out_vocab_size)

    def call(self, x, state, training=False):

        x = tf.cast(x, dtype=tf.float32)

        if training:
            x = tf.squeeze(x)
            start = tf.constant(
                value=self.word_to_idx["BOS"], shape=(x.shape[0], 1), dtype=tf.float32
            )
            x = tf.concat([start, x], axis=1)

        x = self.embedding_layer(x)
        x = self.decoder_rnn1(x)
        decoder_outputs, decoder_states = self.decoder_rnn2(
            x, initial_state=state, training=training
        )

        decoder_outputs = self.dropout(decoder_outputs)
        decoder_outputs = self.decoder_dense1(decoder_outputs)
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        if training:
            return decoder_outputs[:, :-1, :]
        else:
            return decoder_outputs, decoder_states


class EncDec(BaseModel):
    def __init__(
        self, emb_en=None, emb_fr=None, fine_tune_emb_input=False, rnn_size=1, dropout=0
    ):
        super().__init__()
        self.encoder = encoder(emb_en)
        self.decoder = decoder(emb_fr)

    def generate_sequence(self, x, training=False):

        MAX_LENGTH = x.shape[1]

        # ENCODER PORTION

        _, state = self.encoder(x, training=training)

        decoded_sequences = np.zeros(shape=(x.shape[0], MAX_LENGTH), dtype=np.int32)

        logits = np.zeros(shape=(x.shape[0], MAX_LENGTH, self.decoder.out_vocab_size))
        # logits[:,0,self.word_to_idx["BOS"]] = 1  # we force logits for first token to be BOS since here we input BOS to the model

        y = tf.constant(value=self.decoder.word_to_idx["BOS"], shape=(x.shape[0], 1))

        for i in range(
            MAX_LENGTH
        ):  ## We start at one because first token is set to BOS

            decoder_outputs, decod_state = self.decoder(y, state, training=training)

            ## need to generate logits for the rest of train.py/predict.py (softmax cross entropy with logits) eventhough it is not used here #####
            logits[:, i, :] = tf.squeeze(decoder_outputs.numpy()[:, -1, :])
            ####################################################

            pred = tf.math.argmax(
                tf.nn.softmax(decoder_outputs[:, -1, :]),
                axis=-1,
                output_type=tf.dtypes.int32,
            )
            decoded_sequences[:, i] = tf.squeeze(pred).numpy()

            y = tf.concat([y, tf.expand_dims(pred, axis=-1)], axis=-1)
            state = decod_state
            # Exit condition: either hit max length
            # or find stop character.

        with tf.device(
            "/CPU:0"
        ):  # place tensor on CPU due to memory limit, this will slow down loss computation
            logits = tf.convert_to_tensor(logits, dtype=tf.float32)

        return logits
