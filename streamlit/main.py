import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


def load_model():
    model = tf.keras.models.load_model('CNN_Fruits_Classification.h5')
    return model

model = load_model()

Classes=['apple','banana','orange']

st.title('Fruits Classification Application')

st.write("Upload an image of a fruit to identify if it is an **orange**,**apple**, or a **banana**")

uploaded_file=st.file_uploader('Upload an image',type=['jpg','jpeg','png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img,caption='Uploaded image',use_column_width=True)

    img_resized=img.resize((32,32))
    img_array=np.array(img_resized)
    img_array=np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)
    predicted_class=np.argmax(prediction)
    confidence=np.max(prediction)

    st.subheader('Prediction')
    st.success(f'**the predicted class is** \t:{Classes[predicted_class]}')
    st.write(f'Confidence is {confidence * 100:.2f}%')
    st.write("### Class probabilities")
    for i,label in enumerate(Classes):
        st.write(f"{label}:{prediction[0][i]*100:.2f}%")

else :
    st.info(" Please Upload an image of a fruit ")

