from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
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
    return model


if __name__ == '__main__':

    model = load_model()
    st.title('COVID-19 Classifier')
    CLASS_NAMES = ['normal', 'pneumonia', 'COVID-19']

    file = st.file_uploader("Upload a chest x-ray to test", type=["jpg", "png"])

    if file is None:
        st.text('Waiting for upload....')
    else:

        test_image = Image.open(file).convert('RGB')
        st.image(test_image)

        image_arr = np.asarray(test_image).astype('float32')
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.resize(image, [300, 300], antialias=True)/255
        image = np.expand_dims(image, axis = 0)

        pred = model.predict(image)
        result = CLASS_NAMES[np.argmax(pred)] 

        output = 'This is ' + result + ' patient chest x-ray.'

        st.text(output)