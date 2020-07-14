import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from utils import get_prediction
from imagenet_labels import IMAGENET_LABELS

st.markdown('## GradCam')

image_path = st.sidebar.text_input('Enter Absolute Path of Image', './assets/cat_1.jpg')
try:
    image_pil = Image.open(image_path).resize((224, 224))
    st.image(image_pil, caption='Input Image (File: {})'.format(image_path))
except:
    st.error('Invalid File')

model_option = st.sidebar.selectbox(
    'Please Select the Model',
    ('VGG16', 'VGG19', 'ResNet50', 'ResNet101')
)

model_dict = {
    'VGG16': [
        tf.keras.applications.vgg16.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
        tf.keras.applications.vgg16.decode_predictions
    ],
    'VGG19': [
        tf.keras.applications.vgg19.VGG19,
        tf.keras.applications.vgg19.preprocess_input,
        tf.keras.applications.vgg19.decode_predictions
    ]
}

if model_option not in list(model_dict.keys()):
    st.error('Invalid Model')
else:
    classify_button = st.button('Classify')
    if classify_button:
        prediction = get_prediction(image_path, model_dict, model_option)
        st.sidebar.text(prediction)
