from ast import arg
from turtle import onclick
from PIL import Image
from keras.models import model_from_json
import numpy as np
import pandas as pd
import streamlit as st
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from config import MODEL_DIR, WEIGHTS_FILE, IMAGE_REPO_URL
from utils import reformat_img, test_model

subdir = MODEL_DIR
filename = WEIGHTS_FILE
checkpoint_dir = os.path.dirname('./model_data_h5/model_es.h5')



st.set_page_config(page_title="Wildlife Image Classifier App",
                  page_icon="🧊",layout="wide", initial_sidebar_state="expanded"
 )

def columns():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        image = Image.open('./images/Loss.png')
        st.image(image)
        
    with col2:
        image = Image.open('./images/Accuracy.png')
        st.image(image)
        
    with col3:
        image = Image.open('./images/Recall.png')
        st.image(image)
        
    with col4:
        image = Image.open('./images/Precision.png')
        st.image(image)
    
        
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('./model_data_h5/final_model')

def description():
    st.write('''
                Welcome to our classifier app! 
             
                On the left side you can download an image from our server that contains an image that the model has never seen before.
                Afer you download that image, you can upload it and have it classified.
                
                Our overall model performance on training and validation data is found below. 
                You will see benchmarks on loss, accuracy, recall and precision.
             ''')

def main():
    prediction = None
    animal_perc = None
    model = load_model()
    with st.sidebar:
        st.subheader('Utilities')
        st.markdown(f"Download an image [here]({IMAGE_REPO_URL})")
        image_file = st.file_uploader("Upload image here:", type=["jpg","jpeg"])
        
        if st.button('Run Model'):
            with st.spinner('Wait for it...'):
                if image_file is not None:

                    image_reform = reformat_img(image_file)
                    prediction, animal_perc = test_model(model, image_reform)
                    
                    time.sleep(5)    
                    
    st.title('Wildlife Image Classifier')
    if prediction is not None and animal_perc is not None:
        tags = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']
        st.subheader("Class probabilities")
        st.dataframe(pd.DataFrame(animal_perc.round(2), columns=tags))
        st.markdown(f'Predicted class: **{prediction}**')
    st.write('''---''')
    description()
    st.write('''---''')
    columns()
    
if __name__ == '__main__':
    main()