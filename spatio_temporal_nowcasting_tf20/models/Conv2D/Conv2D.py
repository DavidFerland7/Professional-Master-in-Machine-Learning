import tensorflow as tf
from models.BaseModel import BaseModel

class Conv2D(BaseModel):

    def __init__(self):
        super(Conv2D, self).__init__()

        self.concat_time_axis = tf.keras.layers.Concatenate(axis=-1)

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.conv1A = tf.keras.layers.Conv2D(8, (2,2), padding='same', activation=tf.nn.relu)
        self.conv1B = tf.keras.layers.Conv2D(4, (2,2), padding='same', activation=tf.nn.relu)
        self.maxpool1A = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.do1A = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.concat_flat = tf.keras.layers.Concatenate(axis=-1)

        self.dense2A = tf.keras.layers.Dense(12, activation=tf.nn.relu)
        self.dense2B = tf.keras.layers.Dense(12, activation=tf.nn.relu)
        self.dense2C = tf.keras.layers.Dense(8, activation=tf.nn.relu)

        self.denseOutput = tf.keras.layers.Dense(4, name='output')

    def call(self, inputs):

        x = self.concat_time_axis([inputs['imgs'][:,t,:,:,:] for t in range(inputs['imgs'].shape[1])])
        x = self.batch_norm1(x)
        x = self.conv1A(x)
        x = self.batch_norm2(x)
        x = self.conv1B(x)
        x = self.batch_norm3(x)
        x = self.maxpool1A(x)
        x = self.do1A(x)

        x = self.flatten(x)
        x_clearsky_ghi = inputs['CLEARSKY_GHI']
        x_azimuth = inputs['azimuth']
        x_timeofday = inputs['timeofday']
        x_biweekly = inputs['biweekly']
        x = self.concat_flat([x, x_clearsky_ghi, x_azimuth, x_timeofday, x_biweekly])

        x = self.dense2A(x)
        x = self.dense2B(x)
        x = self.dense2C(x)

        x = self.denseOutput(x)
        return x
