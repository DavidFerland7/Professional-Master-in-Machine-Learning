import tensorflow as tf
from models.BaseModel import BaseModel

class ConvLSTM(BaseModel):

        def __init__(self):
            super(ConvLSTM, self).__init__()

            self.batchnorm0 = tf.keras.layers.BatchNormalization()

            self.convltsm1 = tf.keras.layers.ConvLSTM2D(filters=64,
                   kernel_size=(2, 2),
                   strides=(1, 1),
                   return_sequences=True)
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

            self.convlstm2 = tf.keras.layers.ConvLSTM2D(filters=128,
                           kernel_size=(2, 2),
                           strides=(1, 1),
                           return_sequences=True)

            self.batchnorm2 = tf.keras.layers.BatchNormalization()
            #
            # self.convlstm3 = tf.keras.layers.ConvLSTM2D(filters=16,
            #                                             kernel_size=(1, 1),
            #                                             padding='same',
            #                                             strides=(1, 1),
            #                                             return_sequences=True)
            #
            # self.batchnorm3 = tf.keras.layers.BatchNormalization()
            self.conv1A = tf.keras.layers.Conv3D(1, (1, 1, 1))

            self.flatten = tf.keras.layers.Flatten()
            self.concat_flat = tf.keras.layers.Concatenate(axis=-1)
            self.batchnorm4 = tf.keras.layers.BatchNormalization()

            self.dense2A = tf.keras.layers.Dense(512, activation=tf.nn.relu)
            self.dense2B = tf.keras.layers.Dense(16, activation=tf.nn.relu)

            self.denseOutput = tf.keras.layers.Dense(4, name='output')

        def call(self, inputs):

            x = inputs['imgs']

            x = self.convltsm1(x)
            x = self.batchnorm1(x)
            x = self.convlstm2(x)
            x = self.batchnorm2(x)
            #
            # x = self.convlstm3(x)
            # x = self.batchnorm3(x)

            x = self.conv1A(x)

            #x = self.concat_flat([x, inputs['clearsky_ghi']])
            #x = self.batchnorm4(x)
            x = self.dense2A(x)
            x = self.flatten(x)
            x = self.dense2B(x)
            x = self.denseOutput(x)

            return x
