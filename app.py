import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from gradcam import GradCam
from plotly import express as px
from utils import get_overlayed_image


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


def run_app():
    st.markdown(
        '<h1>GradCam Application</h1><hr><br>',
        unsafe_allow_html=True
    )

    # ./assets/cat_1.jpg
    image_path = st.sidebar.file_uploader('Please Select a File')
    if image_path is not None:
        try:
            image_pil = Image.open(image_path)
            original_size = image_pil.size
            height, width = original_size
            image_pil = image_pil.resize((224, 224))
            st.image(image_pil, caption='Input Image')
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

            classify_button = st.sidebar.button('Classify')

            if classify_button:
                (
                    model, preprocess_function, decode_function,
                    last_layer, size, classifier_layers
                ) = model_dict[model_option]
                grad_cam = GradCam(
                    model, preprocess_function, decode_function, last_layer=last_layer,
                    classifier_layers=classifier_layers, size=size
                )
                prediction = grad_cam.get_prediction(image_pil)
                st.markdown(
                    '<hr><h3>Prediction Probablities</h3><br>',
                    unsafe_allow_html=True
                )
                st.plotly_chart(
                    visualize_classification_prediction(prediction),
                    use_container_width=True
                )

                st.markdown(
                    '<hr><h3>GradCam</h3><br>',
                    unsafe_allow_html=True
                )

                image_tensor = grad_cam.get_tensor(image_pil)
                gradcam_heatmap = grad_cam.apply_gradcam(image_tensor)
                overlayed_image = get_overlayed_image(
                    image_pil, gradcam_heatmap, original_size, weightage=0.8
                )

                st.image(overlayed_image)


run_app()
