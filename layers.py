import tensorflow as tf
from tensorflow.keras.layers import Layer

def make_siamese_model():
    input_image = tf.keras.layers.Input(name='input_img', shape=(100, 100, 3))
    validation_image = tf.keras.layers.Input(name='validation_img', shape=(100, 100, 3))

    embedding_model = make_embedding()

    inp_embedding = embedding_model(input_image)
    val_embedding = embedding_model(validation_image)
    
    distances = tf.keras.layers.Lambda(
        lambda x: tf.abs(x[0] - x[1]), 
        name='l1_distance'
    )([inp_embedding, val_embedding])

    classifier = tf.keras.layers.Dense(1, activation='sigmoid')(distances)
    
    return tf.keras.models.Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
