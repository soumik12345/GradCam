import numpy as np
import tensorflow as tf
from matplotlib import cm


def get_overlayed_image(image_path, heatmap, weightage):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    overlayed_img = jet_heatmap * weightage + img
    overlayed_img = tf.keras.preprocessing.image.array_to_img(overlayed_img)
    return overlayed_img
