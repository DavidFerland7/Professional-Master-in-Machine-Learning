import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


class WeightedMeanSquaredError(tf.keras.losses.Loss):
    """
    Args:
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='weighted_mean_squared_error'
    ):
        super(WeightedMeanSquaredError, self).__init__(reduction=reduction, name=name)

    #TODO: remove comments at the end
    # @tf.function
    def weighted_mean_squared_error(self, y_true, y_pred):


        # print("\nbefore True: \n", y_true)
        # print("\nbefore  True shape: \n", y_true.shape)
        if y_true.shape[1] != None:
            n_target = int(y_true.shape[1]/2)
        else:
            n_target = 1
        weights = y_true[:,n_target:]
        y_true = y_true[:,:n_target]
        # tf.print("after: \n", y_true)
        # tf.print("weights: \n", weights)
        # tf.print("pred: \n", y_pred)

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)

        ### calculate weighted avg (weights for an obs(rows) of a batch are the # of 1 in its row mask values)
        # ex: batch size = 2
        # [[1 1 0 0]    -> weights in the batch sum = 2  [l1 = sum of (squared loss * mask) / 2]
        #  [1 1 1 0]]   -> weights in the batch sum = 3  [l2 = sum of (squared loss * mask) / 3]
        # Then finally the loss outputted for the batch is the arithmetic average of the weighted loss of all obs -> L = (l1+l2)/2

        loss = tf.math.divide_no_nan(
            K.sum(tf.math.multiply(math_ops.squared_difference(y_pred, y_true), weights)),
            K.sum(weights)
        )

        # loss_per_target = tf.math.divide_no_nan(K.sum(tf.math.multiply(math_ops.squared_difference(y_pred, y_true), weights), axis=0), K.sum(weights, axis=0))
        # loss = tf.math.divide_no_nan(K.sum(tf.math.multiply(loss_per_target, K.sum(weights, axis=0))), K.sum(weights))

        # tf.print("loss: \n", loss)
        return loss

    def call(self, y_true, y_pred):
        return self.weighted_mean_squared_error(y_true, y_pred)