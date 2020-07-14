import cv2
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from gradcam import GradCam
from plotly import express as px
from imagenet_labels import IMAGENET_LABELS


def visualize_classification_prediction(prediction):
    ids, class_names, probabilities = [], [], []
    for pred in prediction[0]:
        ids.append(pred[0])
        class_names.append(pred[1])
        probabilities.append(pred[2])
    df = pd.DataFrame(data={
        'id': ids,
        'class': class_names,
        'probability': probabilities
    })
    figure = px.bar(
        df, x='class', y='probability',
        hover_data=['id'], color='probability'
    )
    return figure


def get_imagenet_id(class_name: str) -> int:
    for _id in range(1000):
        if IMAGENET_LABELS[_id] == class_name.replace('_', ' '):
            return _id


def run_app():
    st.markdown(
        '<h1>GradCam</h1><hr><br>',
        unsafe_allow_html=True
    )

    # ./assets/cat_1.jpg
    image_path = st.sidebar.text_input('Enter Absolute Path of Image', '')
    if image_path != '':
        try:
            image_pil = Image.open(image_path).resize((224, 224))
            st.image(image_pil, caption='Input Image (File: {})'.format(image_path))
        except:
            st.error('Invalid File')

        model_option = st.sidebar.selectbox(
            'Please Select the Model',
            ('', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101')
        )

        model_dict = {
            'VGG16': [
                tf.keras.applications.vgg16.VGG16,
                tf.keras.applications.vgg16.preprocess_input,
                tf.keras.applications.vgg16.decode_predictions,
                'block5_conv3', (224, 224),
                ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
            ],
            'VGG19': [
                tf.keras.applications.vgg19.VGG19,
                tf.keras.applications.vgg19.preprocess_input,
                tf.keras.applications.vgg19.decode_predictions
            ],
            'ResNet50': [
                tf.keras.applications.resnet50.ResNet50,
                tf.keras.applications.resnet50.preprocess_input,
                tf.keras.applications.resnet50.decode_predictions
            ]
        }

        if model_option in list(model_dict.keys()):

            classify_button = st.button('Classify')

            if classify_button:
                (
                    model, preprocess_function, decode_function,
                    last_layer, size, classifier_layers
                ) = model_dict[model_option]
                grad_cam = GradCam(
                    model, preprocess_function, decode_function, last_layer=last_layer,
                    classifier_layers=classifier_layers, size=size
                )

                prediction = grad_cam.get_prediction(image_path)
                st.markdown(
                    '<hr><h3>Prediction Probablities</h3><br>',
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    visualize_classification_prediction(prediction),
                    use_container_width=True
                )

                gradcam_heatmap = grad_cam.apply_gradcam(grad_cam.get_tensor(image_path))
                overlayed_image = grad_cam.get_overlayed_image(image_path, gradcam_heatmap)

                st.image(overlayed_image.resize((300, 300)))


run_app()
