import numpy as np
import tensorflow as tf


class GradCam:

    def __init__(
            self, model, preprocess_function,
            decode_prediction_function, last_layer, classifier_layers, size):
        self.model = model(weights="imagenet")
        self.preprocess_function = preprocess_function
        self.decode_prediction_function = decode_prediction_function
        self.last_layer = last_layer
        self.classifier_layers = classifier_layers
        self.size = size

    def get_tensor(self, image_path):
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.size
        )
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def get_prediction(self, image_path):
        image = self.get_tensor(image_path)
        image = self.preprocess_function(image)
        prediction = self.model.predict(image)
        prediction = self.decode_prediction_function(prediction)
        return prediction
