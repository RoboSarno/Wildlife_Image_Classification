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

subdir = '../model_data'
filename = 'saved_model.pb'
checkpoint_dir = os.path.dirname('../model_data_h5/model_es.h5')



st.set_page_config(page_title="Wild Image Classifier App",
                  page_icon="🧊",layout="wide", initial_sidebar_state="expanded"
 )

def columns():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        image = Image.open('../images/Loss.png')
        st.image(image)
        
    with col2:
        image = Image.open('../images/Accuracy.png')
        st.image(image)
        
    with col3:
        image = Image.open('../images/Recall.png')
        st.image(image)
        
    with col4:
        image = Image.open('../images/Precision.png')
        st.image(image)
        
def reformat_img(image_file):
    img = Image.open(image_file)
    # width, height = img.size
    # print(width, height)
    # st.image(img)
    img = img.resize((256, 256))
    
    #img_arr = img_to_array(img)
    img_arr = np.reshape(img, (256, 256, 3))
    img_arr = np.expand_dims(img_arr, axis=0) #required structured for preds
    # img_arr = img_arr / 256
    return img_arr
    

def test_model(loaded_model, img):
    print('-----------------------------------')
    class_labels = ['antelope_duiker','bird','blank','civet_genet', 'hog',
            'leopard','monkey_prosimian','rodent']
    prediction_probs = loaded_model.predict(img)
    predicted_label = class_labels[np.argmax(prediction_probs)]
    return predicted_label, prediction_probs
        


@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('../model_data_h5/final_model')


    

def description():
    st.write('''
                Hello, welcome to our application: 
             
                Below you can see Our Models Preformance for train and validation set for Loss, Accuracy, Recall, and Precision.
                Go ahead and Test Our Models Preformance on the Sidebar.
             ''')

def main():
    prediction = None
    animal_perc = None
    model = load_model()
    with st.sidebar:
        st.subheader('Test our Model:')
        image_file = st.file_uploader("Upload Images", type=["jpg","jpeg"])

        if st.button('Run Model'):
            with st.spinner('Wait for it...'):
                if image_file is not None:

                    image_reform = reformat_img(image_file)
                    prediction, animal_perc = test_model(model, image_reform)
                    
                    time.sleep(5)     
                    
    st.title('Wild Image Classifier')
    st.write('''---''')
    description()
    st.write('''---''')
    columns()
    if prediction is not None and animal_perc is not None:
        
        st.dataframe(pd.DataFrame(animal_perc.round(2), columns=['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']))
        st.write(f'The models prediction is...{prediction}')
    
if __name__ == '__main__':
    main()