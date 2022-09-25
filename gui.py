

import streamlit as st
import time
from keras.models import load_model
import cv2
from PIL import Image
from PIL import Image
import numpy as np 
import cv2
import pandas as pd 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow_io as tfio
from skimage.io import imsave
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpg')  
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write("")
with col2:
    st.title("Optic Nerve Segmentation")    
with col3:
    st.write("")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write("")
with col2:
    st.image("ons.png")
with col3:
    st.write("")

IMAGE_SIZE = 512
BATCH_SIZE = 8

def read_files(image_path, mask=False):
    image = tf.io.read_file(image_path)
    #image = io.imread(image_path)
    if mask:
        image = tf.io.decode_gif(image) 
        image = tf.squeeze(image)
        #image=clahe_equalized(image)
        #image=adjust_gamma(image,1.2)
        image = tf.image.rgb_to_grayscale(image) 
        image = tf.divide(image, 128)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.cast(image, tf.int32)
        
        
    else:
        try:
            image = tfio.experimental.image.decode_tiff(image)
        except: 
            image = tf.io.decode_image(image)
        image = image[:,:,:3] # out: (h, w, 3)
        image.set_shape([None, None, 3])
        #image=clahe_equalized(image)
        #image=adjust_gamma(image,1.2)
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255.
    return image

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def create_mask(pred_mask):
    pred_mask = (pred_mask > 0.5).astype("int32")
    return pred_mask[0]

unet1 = load_model("./unet1/unetscratch",custom_objects={'dice_coef':dice_coef,'iou':iou})
unet2 = load_model("./unet2/unetsm",custom_objects={'dice_coef':dice_coef,'iou':iou})

input_image = st.file_uploader("Select an Image to feed into the model: ")
print(input_image)
if st.button("Submit"):
    name=input_image.name
    print(name)
    st.success("Image recieved successfully: "+str(input_image.name))
    img = Image.open(name)
    img.save('image.tiff')
    x=read_files("image.tiff")
    x=x.reshape([1,512,512,3])  
    pred=unet1.predict(x)
    pred2 = unet2.predict(x)
    pred_masked=create_mask(pred)
    pred_masked2 = create_mask(pred2)
    imsave('output.jpg',pred_masked)
    imsave('output2.jpg',pred_masked2)
    
    
    st.title('Input Image')

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.write("")
    with col2:
        st.image("1.jpg")
    with col3:
        st.write("")
    
    st.title("Output Image")
    
    col1,col2 = st.columns([1,1])
    with col1:
        st.write("Model 1 Prediction")
    with col2:
        st.write("Model 2 Prediction") 
    col1, col2 = st.columns([1,1])
    with col1:
        st.image('output.jpg')
    with col2:
        st.image('output2.jpg')
