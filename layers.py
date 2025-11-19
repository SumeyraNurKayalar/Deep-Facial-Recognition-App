import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        # Inputs bir tuple olarak geliyor: (input_embedding, validation_embedding)
        input_embedding, validation_embedding = inputs
        
        # Tensor'ları doğrudan al
        return tf.math.abs(input_embedding - validation_embedding)
    
    def compute_output_shape(self, input_shape):
        # Input shape bir tuple: (shape1, shape2)
        return input_shape[0]
    
    def get_config(self):
        return super().get_config()