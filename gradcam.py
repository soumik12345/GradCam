import cv2
import numpy as np
import tensorflow as tf
from utils import read_image


def get_grad_model(model, layer_name):
    grad_model = model(weights='imagenet', include_top=True)
    grad_model = tf.keras.models.Model(
        [grad_model.inputs],
        [
            grad_model.get_layer(layer_name).output,
            grad_model.output
        ]
    )
    return grad_model


def generate_cam(image_file, grad_model, class_index, image_size=224):
    image, original_image = read_image(image_file, image_size=image_size)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    del grad_model
    return original_image, cam


def generate_heatmap(original_image, cam, image_size=224):
    cam = cv2.resize(cam.numpy(), (image_size, image_size))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_image = cv2.resize(
        np.array(original_image),
        (image_size, image_size)
    )
    cam_overlayed = cv2.addWeighted(
        cv2.cvtColor(
            original_image,
            cv2.COLOR_RGB2BGR
        ), 0.4, cam, 0.6, 0
    )
    return cam, cam_overlayed
