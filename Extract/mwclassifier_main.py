'''
This is the main function of the malware classification of this program
'''

import sys
import os
import pickle
from math import log
import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

## This function allows us to process our files into png images##
def covert_to_png(fpath: str, img_size: tuple =(64,64,3)) -> Image:
    with open(fpath, 'rb') as file:
        binary_data  = file.read()
    file.close()

    # Convert the bytes to a numpy array
    file_array = np.frombuffer(binary_data, dtype=np.uint8)

    # Resize the array to (64, 64)
    resized_array = np.resize(file_array, img_size)

    # Create a grayscale PIL Image from the resized array
    image = Image.fromarray(resized_array, mode='RGB')

    return image


# Registere functions with @keras.utils.register_keras_serializable to load model
from keras import backend as K
def recall_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_test, y_pred):
    precision = precision_m(y_test, y_pred)
    recall = recall_m(y_test, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


if __name__ == '__main__':
    # # Get file name input
    # file_name = str(input("Input file name: "))
    
    # # Check if file name exist
    # TestFileDir = os.getcwd() +'\\TestFile'
    # dir_list = os.listdir(TestFileDir)
    # if file_name not in dir_list:
    #     print("Bad input\nExiting...")
    #     exit()
    
    # Convert file to 
    fpath = os.getcwd() + f'\\TestFile\\{sys.argv[1]}'
    png = covert_to_png(fpath)

    # Loading the Malware_Classifier_model.h5
    with open('Classifier\Malware_Classifier\pickel_malware_classifier.pkl','rb') as file:
        pickle_model = pickle.load(file)
    file.close()

    # Loading list of malware family classes
    with open('Classifier\Malware_Classifier\Malware_classes.pkl','rb') as file1:
        class_names = pickle.load(file1)
    file1.close()

    # Convert png to image array
    img_array = keras.utils.img_to_array(png)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = pickle_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {}."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )    


