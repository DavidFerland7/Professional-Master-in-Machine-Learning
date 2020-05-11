import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tensorflow as tf
from tensorflow.keras import layers
from models.BaseModel import BaseModel
from models.utils.TimeDistributedCustom import TimeDistributedCustom as TD
from tensorflow.keras.layers import Bidirectional


class GRU(BaseModel):
    def __init__(
        self,
        emb_en=None,
        emb_fr=None,
        fine_tune_emb_input=False,
        rnn_size=1,
        dropout=0.5,
        recurrent_dropout=0.0,
    ):
        super().__init__()

        if emb_en:
            self.embedding = self.build_pre_emb_layer(
                emb_en, fine_tune_emb_input=fine_tune_emb_input
            )

        self.rnn1 = Bidirectional(
            layers.GRU(
                rnn_size,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )
        self.rnn2 = Bidirectional(
            layers.GRU(
                rnn_size,
                return_sequences=True,
                dropout=0.0,
                recurrent_dropout=recurrent_dropout,
            )
        )

        self.out = TD(layers.Dense(emb_fr.vocab_size))

    def call(self, x, training=False):
        x = self.embedding(tf.squeeze(x))
        x = self.rnn1(x, training=training)
        x = self.rnn2(x, training=training)
        return self.out(x)
