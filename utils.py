import numpy as np
from PIL import Image
import pandas as pd
from config import TEST_IMG_DIR

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

def lookup(filename):
    classes_df = pd.read_csv(TEST_IMG_DIR, index_col="id")
    row = classes_df.loc(f'{filename}')
    try:
        idx = list(row).index(1)
        label = str(classes_df.columns[idx])
        return label
    except:
        return "Image not found. Try again."