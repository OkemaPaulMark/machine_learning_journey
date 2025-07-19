import tensorflow as tf

model = tf.keras.models.load_model("crop-disease_model.h5")
print("Model input shape:", model.input_shape)
