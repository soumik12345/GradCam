import numpy as np
from PIL import Image
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
