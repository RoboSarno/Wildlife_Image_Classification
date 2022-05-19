
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

subdir = './model_data'
filename = 'saved_model.pb'
checkpoint_dir = os.path.dirname('./model_data_h5/model_es')



st.set_page_config(page_title="Wild Image Classifier App",
                  page_icon="🧊",layout="wide", initial_sidebar_state="expanded"
 )

def columns():
    col1, col2, col3 = st.columns(3)
    with col1:
        image = Image.open('./test.png')
        st.image(image)
        
    with col2:
        image = Image.open('./test.png')
        st.image(image)
        
    with col3:
        image = Image.open('./test.png')
        st.image(image)
        
def reformat_img(image_file):
    img = Image.open(image_file)
    # width, height = img.size
    # print(width, height)
    # st.image(img)
    img = img.resize((256, 256))
    
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape(256, 256, 3)

    # img_arr = img_arr / 256
    return img_arr
    

def test_model(loaded_model, img):
    print('-----------------------------------')
    return loaded_model.predict(img)
        
def sidebar():
    model = load_model()
    with st.sidebar:
        st.subheader('Test our Model:')
        st.write('Select An Image')
        image_file = st.file_uploader("Upload Images", type=["jpg","jpeg"])

        if st.button('Run Model'):
            print('fdsafdsfdasfssa')
            with st.spinner('Wait for it...'):
                if image_file is not None:

                    image_reform = reformat_img(image_file)
                    prediction = test_model(model, image_reform)
                    print(prediction.shape)
                    
                    # print(prediction)
                    time.sleep(5)     
        #         st.success('Done!')
            return prediction

    # return None
# @st.cache
def load_model():
    return tf.keras.models.load_model('./model_data_h5/model_es.h5')


    

def description():
    st.write('Hello welcome to our application: ')

def main():
    prediction = sidebar()
    st.title('Wild Image Classifier')
    st.write('''---''')
    description()
    st.write('''---''')
    columns()
    st.subheader('The models prediction is...')
    if prediction is not None:
        print(prediction)
        # st.write(prediction)
        # st.dataframe(prediction, columns=['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent'])
    
    

if __name__ == '__main__':
    main()