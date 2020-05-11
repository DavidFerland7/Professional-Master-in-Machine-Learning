import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


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
        # note: get_output_shapes(inputs)[0] is the tensor, position [1] would be the targets (this is from the generator)
        # x = {k: tf.keras.layers.Input(tuple(v[1:]), name=k) for k, v in tf.compat.v1.data.get_output_shapes(inputs)[0].items()}
        # x = tf.keras.layers.Input(inputs)
        x = tf.keras.layers.Input(inputs.shape[1:], batch_size=inputs.shape[0])
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def build_pre_emb_layer(self, emb_obj, fine_tune_emb_input=False):
        # WARNING emb_obj has to be a gemsim model
        # create gensim compatible layer for keras, english voc

        embedding_matrix = np.zeros((emb_obj.vocab_size, emb_obj.emb_size))
        embedding_matrix[1:] = emb_obj.model.wv[emb_obj.word_to_idx.keys()]
        return layers.Embedding(
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            input_dim=emb_obj.vocab_size,
            output_dim=emb_obj.emb_size,
            trainable=fine_tune_emb_input,  ## Change to FALSE if we don't want to fine-tuned embedding layer
            mask_zero=True,
        )
