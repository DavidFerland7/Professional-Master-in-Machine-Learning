import tensorflow as tf

class BaseModel(tf.keras.Model):

    def __init__(self):
        super(BaseModel, self).__init__()

    def build_graph(self, inputs):
        """
        This function call the model with tf.keras functional api
        The main purposes of this function are:
         (1) Have the layer shapes properly displayed in the model.Summary() output
            ** reason: shapes are displayed as 'multiple' when using the subclassing api
         (2) Be able to output the model from tf.keras.misc.plot_model function
        :param inputs: data_loader
        :return:
        """
        #note: get_output_shapes(inputs)[0] is the tensor, position [1] would be the targets (this is from the generator)
        x = {k: tf.keras.layers.Input(tuple(v[1:]), name=k) for k, v in tf.compat.v1.data.get_output_shapes(inputs)[0].items()}
        return tf.keras.Model(inputs=x, outputs=self.call(x))