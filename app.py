from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_clahe
import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    feature_extractor_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2", input_shape=(300, 300, 3), trainable=False)
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])   
    model.load_weights("model_weight/CL_EffNetv2-B3_weights.h5")

    x_model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b3/classification/2")
    ])
    x_model.build([None, 300, 300, 3])
    return model, x_model

def load_imagenet_label():
    labels_file = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
    downloaded_file = tf.keras.utils.get_file("labels21.txt", origin=labels_file)
    classes = []
    with open(downloaded_file) as f:
        labels = f.readlines()
        classes = [l.strip() for l in labels]
    classes = np.array(classes)
    return classes

if __name__ == '__main__':

    model, x_model = load_model()
    IMGNET_CLASSES = load_imagenet_label()
    st.image("https://cdn-icons-png.flaticon.com/512/4193/4193294.png", width=100)
    st.title('COVID-19 Pneumonia Classifier')
    st.markdown('##### An Efficient Deep Learning Algorithm For COVID-19 Pneumonia Detection Using Chest Radiography Imaging')
    st.caption('The algorithm has been trained to distinguish between normal, COVID-19 pneumonia, and other viral or bacterial pneumonia patients. The image will be enhanced using CLAHE technique prior to being passed onto the prediction model.')
    CLASS_NAMES = ['normal', 'pneumonia', 'COVID-19']

    file = st.file_uploader("Upload a chest x-ray", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text('Waiting for upload....')
    else:

        test_image = Image.open(file).convert('RGB')
        image_arr = np.asarray(test_image).astype('float32')
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.resize(image, [300, 300], antialias=True)

        # check x-ray image
        image_check = np.expand_dims(image, axis = 0)/255
        check_xray = x_model(image_check)
        check_index = np.argmax(check_xray)

        if check_index == 9647:
            image_clahe = tf_clahe.clahe(image, tile_grid_size=(7, 7), clip_limit=3.)/255

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Uploaded image")
                st.image(test_image, use_column_width=True)
            with col2:
                st.caption("CLAHE")
                st.image(image_clahe.numpy(), use_column_width=True)

            image_clahe = np.expand_dims(image_clahe, axis = 0)

            predictions = model.predict(image_clahe)
            result = CLASS_NAMES[np.argmax(predictions)]
            confidence = predictions[0][np.argmax(predictions)]*100

            st.text('The chest x-ray belongs to a ' + result + ' patient.')
            st.text('Confidence : {:.2f}%'.format(confidence))
        else:
            st.caption("Uploaded image")
            st.image(test_image, use_column_width=True)
            st.error("Please upload x-ray image. This is " + IMGNET_CLASSES[check_index], icon="🚨")