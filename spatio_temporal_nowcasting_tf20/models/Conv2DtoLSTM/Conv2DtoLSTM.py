import tensorflow as tf
from models.BaseModel import BaseModel

from models.utils.TimeDistributedCustom import TimeDistributedCustom as TimeDistributed

class Conv2DtoLSTM(BaseModel):

    def __init__(self):
        super(Conv2DtoLSTM, self).__init__()

        self.batchnorm0 = tf.keras.layers.BatchNormalization()

        #self.concat_time_axis = tf.keras.layers.Concatenate(axis=0)

        self.conv1A = TimeDistributed(tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.conv1B = TimeDistributed(tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.maxpool1A = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.conv2A = TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.conv2B = TimeDistributed(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.maxpool2A = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.conv3A = TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm5 = tf.keras.layers.BatchNormalization()
        self.conv3B = TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu))
        self.batchnorm6 = tf.keras.layers.BatchNormalization()
        self.maxpool3A = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.flatten1 = TimeDistributed(tf.keras.layers.Flatten())

        self.dense1A = tf.keras.layers.Dense(24)

        self.lstm = tf.keras.layers.LSTM(units=512)
        self.batchnorm7 = tf.keras.layers.BatchNormalization()

        #self.concat_flat = tf.keras.layers.Concatenate(axis=-1)
        #self.batchnorm8 = tf.keras.layers.BatchNormalization()



        #self.flattend = tf.keras.layers.Flatten()



        self.dense2A = tf.keras.layers.Dense(12)
        self.dense2B = tf.keras.layers.Dense(8)

        self.denseOutput = tf.keras.layers.Dense(4, name='output')

    def call(self, inputs):

        x = inputs['imgs']

        x = self.batchnorm0(x)

        x = self.conv1A(x)
        x = self.batchnorm1(x)
        x = self.conv1B(x)
        x = self.batchnorm2(x)
        x = self.maxpool1A(x)

        x = self.conv2A(x)
        x = self.batchnorm3(x)
        x = self.conv2B(x)
        x = self.batchnorm4(x)
        x = self.maxpool2A(x)

        x = self.conv3A(x)
        x = self.batchnorm5(x)
        x = self.conv3B(x)
        x = self.batchnorm6(x)
        x = self.maxpool3A(x)

        x = self.dense1A(x)

        x = self.flatten1(x)
        x = self.lstm(x)
        x = self.batchnorm7(x)


        #x = self.concat_flat([x, inputs['CLEARSKY_GHI']])
        #x = self.batchnorm8(x)

        x = self.dense2A(x)
        x = self.dense2B(x)

        x = self.denseOutput(x)
        return x
