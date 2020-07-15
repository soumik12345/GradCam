import cv2
import numpy as np
import tensorflow as tf
from matplotlib import cm


def get_overlayed_image(image_pil, heatmap, original_size, weightage):
    image_tensor = tf.keras.preprocessing.image.img_to_array(image_pil)
    image_tensor = np.squeeze(image_tensor)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_tensor.shape[1], image_tensor.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    overlayed_img = jet_heatmap * weightage + image_tensor
    overlayed_img = cv2.resize(
        overlayed_img, (
            int(overlayed_img.shape[0] * (original_size[0] / original_size[1])),
            int(overlayed_img.shape[1] * (original_size[0] / original_size[1]))
        )
    )
    overlayed_img = tf.keras.preprocessing.image.array_to_img(overlayed_img)
    return overlayed_img
