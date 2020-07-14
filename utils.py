import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt


def read_image(file_path, image_size):
    original_image = np.array(Image.open(file_path))
    image = keras.preprocessing.image.load_img(
        file_path, target_size=(image_size, image_size)
    )
    image = np.array([keras.preprocessing.image.img_to_array(image)])
    return image, original_image


def visualize(image, cam, cam_overlayed):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))
    plt.setp(axes.flat, xticks=[], yticks=[])
    for i, ax in enumerate(axes.flat):
        if i % 3 == 0:
            ax.imshow(image)
            ax.set_xlabel('Image')
        elif i % 3 == 1:
            ax.imshow(cam)
            ax.set_xlabel('Cam')
        else:
            ax.imshow(cam_overlayed)
            ax.set_xlabel('Cam Overlayed')
    plt.show()


def get_prediction(image_path: str, model_dict: dict, model_option: str) -> list:
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = cv2.resize(image, (224, 224))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    model, preprocessing_function, decoding_function = model_dict[model_option]
    image = preprocessing_function(image)
    model = model(weights='imagenet')
    prediction = model.predict(image)
    prediction = decoding_function(prediction)
    return prediction
