import numpy as np
import tensorflow as tf
from matplotlib import cm


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

    def get_tensor(self, image):
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def get_prediction(self, image):
        image = self.get_tensor(image)
        image = self.preprocess_function(image)
        prediction = self.model.predict(image)
        prediction = self.decode_prediction_function(prediction)
        return prediction

    def apply_gradcam(self, image_tensor):
        last_conv_layer = self.model.get_layer(self.last_layer)
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in self.classifier_layers:
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(image_tensor)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        heatmap = np.mean(last_conv_layer_output, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    @staticmethod
    def get_overlayed_image(image_path, heatmap):
        img = tf.keras.preprocessing.image.load_img(image_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        overlayed_img = jet_heatmap * 0.7 + img
        overlayed_img = tf.keras.preprocessing.image.array_to_img(overlayed_img)
        return overlayed_img
