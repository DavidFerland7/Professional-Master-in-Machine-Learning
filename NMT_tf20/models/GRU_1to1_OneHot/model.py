import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tensorflow as tf
from tensorflow.keras import layers
from models.BaseModel import BaseModel
from models.utils.TimeDistributedCustom import TimeDistributedCustom as TD


class GRU(BaseModel):
    def __init__(self, out_vocab_size, rnn_size=1, dropout=0.5):
        super().__init__()
        self.rnn = layers.GRU(rnn_size, return_sequences=True, dropout=dropout)
        self.out = TD(layers.Dense(out_vocab_size))

    def call(self, x, training=False):
        x = self.rnn(x, training=training)
        return self.out(x)
