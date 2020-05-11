import tensorflow as tf
from tensorflow.python.ops.losses import util as tf_losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops


#@tf.function
class WeightedMeanSquaredErrorMetric(tf.keras.metrics.Metric):

    def __init__(self, name='weighted_mean_squared_error', **kwargs):
        #super(WeightedMeanSquaredErrorMetric, self).__init__(name=name, dtype=dtype,  **kwargs)
        #super(WeightedMeanSquaredErrorMetric, self).__init__(name=name,  **kwargs)
        super(WeightedMeanSquaredErrorMetric, self).__init__(name=name,  **kwargs)
        self.metric = self.add_weight(name="weighted_mse", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # tf.print("before: \n", y_true)
        weights = y_true[:,4:]
        y_true = y_true[:,:4]
        # tf.print("after: \n", y_true)
        # tf.print("weights: \n", weights)
        # tf.print("pred: \n", y_pred)

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        #loss = K.sum(tf.math.multiply(math_ops.squared_difference(y_pred, y_true), weights))/K.sum(weights)
        loss = K.mean(K.sum(tf.math.multiply(math_ops.squared_difference(y_pred, y_true), weights), axis=-1)/K.sum(weights, axis=-1), axis=-1)

        # tf.print("loss: \n", loss)

        #loss = K.mean(tf.math.multiply(math_ops.squared_difference(y_pred, y_true), weights))  #,axis=-1)
        return self.metric.assign_add(loss)

    def result(self):
        return self.metric

    def reset_states(self):
        self.metric.assign(0.)