# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This Notebook shows implementaion of Google Deep Dream algorithm
#
# **CONTENTS**
#
# 1. IMPORT MODEL WITH PRE-TRAINED WEIGHTS
# 2.  VISUALIZATING IMAGE AND PRE-PROCESS IT!
# 3. RUN THE PRETRAINED MODEL, SELECTING LAYERS AND EXPLOREING ACTIVATIONS
# 4. intitution: UNDERSTANDING DEEPDREAM
# 5. IMPLEMENT DEEP DREAM ALGORITHM USING INCEPTIONNET
# 6. (VIDEO) APPLY DEEPDREAM TO GENERATE A SERIES OF IMAGES
# 7. DEEPDREAM USING DENSENET

# %% [markdown]
# # 1: IMPORT MODEL WITH PRE-TRAINED WEIGHTS

# %% _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from IPython import display

# %% _kg_hide-output=true
# loading pre-trained model and weights
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# %% _kg_hide-output=true
base_model.summary()

# %% [markdown]
# # 2: VISUALIZATING IMAGE AND PRE-PROCESS IT!

# %%
# Open the first image
img_1 = Image.open('../input/mars-eiffel-deepdream/mars.jpg')
img_1

# %%
# opening second image
img_2 = Image.open('../input/mars-eiffel-deepdream/eiffel.jpg')
img_2

# %%
# Blend the two images
img_0 = Image.blend(img_1, img_2, alpha=0.5)# alpha --> The interpolation alpha factor. If alpha is 0.0, a copy of the first image is returned.
                                                # If alpha is 1.0, a copy of the second image is returned.
img_0

# %%
Image.blend(img_1, img_2, 0.1)

# %%
np.shape(img_1)

# %%
type(img_1)

# %%
# Convert to numpy array
sample_image = tf.keras.preprocessing.image.img_to_array(img_0)
# Confirm that the image is converted to Numpy array
type(sample_image)

# %%
# Obtain the max and min values
print('Maximum value of pixel {} & minimum value of pixel {}'.format(sample_image.max(), sample_image.min()))

# %%
# Normalize the input image
sample_image = np.array(sample_image)/255.
# verify normalized images values!
print('Maximum value of pixel {} & minimum value of pixel {}'.format(sample_image.max(), sample_image.min()))

# %%
sample_image = tf.expand_dims(sample_image, axis=0)
np.shape(sample_image)

# %% [markdown]
# # 3: RUN THE PRETRAINED MODEL, SELECTING LAYERS AND EXPLOREING ACTIVATIONS
#
# * Inception network has multiple concatenated layers named `mixed` (refer model summary)
# * we can select any layer depending on the feature we want such as edges, shapes
# * deep layers generate more compleex features such as entire face, building or tree

# %% _kg_hide-output=true
names = ['mixed3', 'mixed5', 'mixed8', 'mixed9']

layers = [base_model.get_layer(name).output for name in names]

#create feature extracion model
deepdream_model = tf.keras.Model(base_model.input, layers)
deepdream_model.summary()

# %% _kg_hide-output=true
# Let's run the model by feeding in our input image and taking a look at the activations "Neuron outputs"
activations = deepdream_model(sample_image)
activations

# %%
len(activations) # equal to the number of layers we selected

# %% [markdown]
# # 4: UNDERSTANDING DEEPDREAM

# %% [markdown]
# Deep Dream is a computer vision program created by Google.
# Uses a convolutional neural network to find and enhance patterns in images
# with powerful AI algorithms.
# Creating a dreamlike hallucinogenic appearance in the deliberately
# over-processed images.
#
# When you look for shapes in the clouds, you’ll often find things you see every day: Dogs, people, cars. It turns out that artificial “brains” do the same thing. Google calls this phenomenon “Inceptionism,” and it’s a shocking look into how advanced artificial neural networks really are. ON trained upon, each CNN layer learns filters of increasing complexity. The first layers learn basic feature detection filters such as edges and corners. The middle layers learn filters that detect parts of objects — for faces, they might learn to respond to eyes and noses. The last layers have higher representations: they learn to recognize full objects, in different shapes and positions.
#
# * If we feed an image to CNN model it it starts detecting various featue in a image, the initial layers will detect low-level features like edges and shape
# * AS deepper as we go high level features will be detected like color, face , tree, buliding...
# * It is expermented that is a network is trained over say dog then it try to detect a dog in the image, or a plant or any face.
# * From here the idea of deep dream rises that what is we try to magnify what the network is detecting.
# * When we feed an image in to CNN, the neurons will fire up generate results, we call them activations
# * The deep dream algorithm works in the way that it will try to change the input image by making soome of these neurons fire more
# * We can select layers that we want to keep in the network & bhy the combination of different layers, intersting patterns will drawn in image
# * For Ex. if input a image of water to network which is trained on fish images, then network will to identify fishes in the image and will generate output of we magnify these activations. More patterns will apper in image.
#
#
# "The final few layers assemble those into complete interpretations—these neurons activate in response to very complex things such as entire buildings or trees,” Google’s engineers explain.
#
#
# Normally when an image is feed to network depending on the problem we select a loss function and try to minimize it which is called `gradient decend` and feed the image again, but in deep dream we try to maximize this loss (gradient ascend). Instead of changing model weights we are changinf the input
#
#
# **DEEP DREAM STEPS**
# 1. First we train a deep netwok
# 2. We select the layers we want ot keep in network
# 3. Calculate the activations coming from them
# 4. Calculte the gradient and loss of these activations 
# 5. Modify the input image by incresing these activation, thus enhancing the pattern
# 6. Feed the obtained image to network again
# 7. This process is repeated number of times
#
# ![alt text](https://drive.google.com/uc?id=1R7_C4r4vy2tqIB5Pi-ltyY2N_WC6jSYF)

# %% [markdown]
# **BELOW ARE SOME BEAUTIFUL WORKS**
# ![A](https://i.kinja-img.com/gawker-media/image/upload/c_fit,f_auto,g_center,pg_1,q_60,w_1465/1302697548279136403.png)
# ![aa](https://i.kinja-img.com/gawker-media/image/upload/c_fit,f_auto,g_center,pg_1,q_60,w_1465/1302697548325886611.png)
#
# ![vv](https://i.kinja-img.com/gawker-media/image/upload/c_fit,f_auto,g_center,pg_1,q_60,w_1315/1302697548458347155.png)
# ![aa](https://i.kinja-img.com/gawker-media/image/upload/c_fit,f_auto,g_center,pg_1,q_60,w_1465/1302697548523125139.png)

# %% [markdown]
# # 5: IMPLEMENT DEEP DREAM ALGORITHM

# %%
sample_image.shape

# %%
# Since the cal_closs function includes expand dimension, let's squeeze the image (reduce_dims)
sample_image = tf.squeeze(sample_image, axis=0)
sample_image.shape


# %% _kg_hide-output=true
#LOSS CALCULATION

# REFERANCE: https://www.tensorflow.org/tutorials/generative/deepdream
def calc_loss(image, model):
    '''
    Function will calculate loss function
    it works by feedforwarding the input image through the network and generate activations
    the obtain the average sum
    '''
    img = tf.expand_dims(image, axis=0) # converting image to bach format
    layer_activation = model(img) #extracting activation results from model

    print('ACTIVATION VALUES (LAYER OUTPUT) =\n', layer_activation)
    
    losses = []
    for act in layer_activation:
        l = tf.math.reduce_mean(act) #calculate mean of each activation
        losses.append(l)
    
    print('LOSSES (FROM MULTIPLE ACTIVAION LAYERS)= ',losses)
    print('LOSSES SHAPE (FROM ALL ACTIVATION LAYERS)= ',np.shape(losses))
    print('SUM OF ALL LOSSES (OF ALL LAYER)= ',tf.reduce_sum(losses))
    
    return tf.reduce_sum(losses)  #calculate sum

calc_loss(tf.Variable(sample_image), deepdream_model)


# %%
#CALCULATE THE GRADIENT

##loss that has been calculated in the previous step and calculate the gradient with respect to the given input image and then
   #add it to the input original image.
##Doing so iteratively will result in feeding images that continiously and increasingly excite the neurons and generate more 
   #dreamy like images!
    
# When you annotate a function with tf.function, the function can be called like any other python defined function. 
# The benefit is that it will be compiled into a graph so it will be much faster and could be executed over TPU/GPU
@tf.function
def deepdream(model, image, step_size):
    with tf.GradientTape() as tape:
        # this need gradient relative to 'img'
        # 'GradientTape' on;y watches 'tf.Variable' by default
        tape.watch(image)
        loss = calc_loss(image, model)   #calling function to caluclate loss
        
    # Calculate the gradient of the loss with respect to the pixels of the input image.
    gradients = tape.gradient(loss, image)
    print('GRADIENT =\n', gradients)
    print('GRADIENTS SHAPE =\n', np.shape(gradients))
    
    # tf.math.reduce_std computes the standard deviation of elements across dimensions of a tensor
    gradients /= tf.math.reduce_std(gradients)
    
    # In `gradient ascent`, the "loss" is maximized so that the input image increasingly "excites" the layers.
    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1,1) #normalize the image as addition may icrease the pixels values 
    
    return loss, image


# %%
def deprocess(image):
    image = 255*(image +1)/2.
    return tf.cast(image, tf.uint8)

def run_deepdream(model, image, steps=100, step_size=0.01):
    # convert from unit8 to range expected by model
    
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    
    var = image
    alpha = 0.5
    for s in range(steps):
        loss, image = deepdream(model, image, step_size)
        
        if s%100==0:
            plt.figure(figsize=(8,8))
            plt.imshow(deprocess(image))
            plt.title("STEP {}, LOSS {}".format(s, loss))
            plt.show()
            #plt.title("STEP {}, LOSS {}".format(s, loss))
        
    plt.figure(figsize=(8,8))
    plt.imshow(deprocess(image))
    plt.show()
    
    return deprocess(image)


# %%
img_0.save("img_0.jpg", "JPEG", quality=80, optimize=True, progressive=True)

# %%

Sample_Image = np.array(load_img('./img_0.jpg'))

dream_img = run_deepdream(model=deepdream_model,
                          image=Sample_Image,
                          steps=1000,
                          step_size=0.02
                         )

# %% [markdown]
# # 6: (VIDEO) APPLY DEEPDREAM TO GENERATE A SERIES OF IMAGES

# %%
# name of folder
dream_name = 'inception_dream'
path = './' + dream_name

if not os.path.exists(path):
    os.makedirs(path)

# saving image in one dir to make video
img_0.save(path + '/img_0.jpg', "JPEG", quality=80, optimize=True, progressive=True)


# %%
# defining function again by removing imshow satatement
def run_deepdream(model, image, steps=100, step_size=0.01):
    # convert from unit8 to range expected by model
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    
    for s in range(steps):
        loss, image = deepdream(model, image, step_size)
        
        if s%100==0:
            #plt.figure(figsize=(8,8))
            #plt.imshow(deprocess(image))
            #plt.show()
            print("STEP {}, LOSS {}".format(s, loss))
            
    #plt.figure(figsize=(8,8))
    #plt.imshow(deprocess(image))
    #plt.show()
    
    return deprocess(image)


# This helper function loads an image and returns it as a numpy array of floating points

def load_image(filename):
    image = Image.open(filename)
    return np.float32(image)

def save_dream(model, path):
    '''
    this function will create the image and save it direcory of 
    creating video
    '''
    
    # Blended image dimension
    x_size = 910 # larger the image longer is going to take to fetch the frames 
    y_size = 605

    #zoom the image
    x_zoom = 3
    y_zoom = 2
        
    for i in range(60):
        img = load_image(path + '/img_{}.jpg'.format(i))
        
        #chop off the edges of the image and resize it back to original shape
        img = img[0+x_zoom:y_size-y_zoom, 0+y_zoom:x_size-x_zoom]
        img = cv2.resize(img, (x_size, y_size))
        
        # adjust RGB values (not necessary just experimental)
        img[:, :, 0] +=2 #red
        img[:, :, 1] +=2 #green
        img[:, :, 2] +=2 #blue
        
        # DEEP DREAM MODEL
        img = run_deepdream(model=deepdream_model, image=img, steps=500, step_size=0.02)
        
        #clip the image
        img = np.clip(img, 0., 255.)
        img = img.astype(np.uint8)
        res = Image.fromarray(img, mode='RGB')
        
        # the save generated image
        res.save(path + '/img_{}.jpg'.format(i+1))


# %% _kg_hide-output=true
save_dream(deepdream_model, path)


# %%
# creating dream video

def get_video(path, fname):
    out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'MP4V'), 8,(910, 605))
    
    # The frames per second value is depends on few important things
    # 1. The number of frames we have created. Less number of frames brings small fps
    # 2. The larger the image the bigger the fps value. For example, 1080 pixel image can bring 60 fps 
    i = 0
    while True:
        if os.path.isfile(path + '/img_{}.jpg'.format(i+1)):
            i +=1
        # Figure out how long the dream is 
        else:
            dream_length = i
            break

    for i in range(dream_length):
        try:
            img = os.path.join(path, 'img_{}.jpg'.format(i))
            print(img)
            frame = cv2.imread(img)
            out.write(frame)
        except Exception as e:
            print(e)
    out.release()


# %% _kg_hide-output=true
get_video('./inception_dream', 'inceptiondream.mp4')

# %%
display.YouTubeVideo(id='a_9dF6UUPbI', height=605, width=910)

# %% [markdown]
# I was unable to display video using `IPython.display.Video or HTML`, i had no other choice but to put the video on youtube. If any one of you figure it out please do comment

# %% [markdown]
# # 7: DEEPDREAM USING DENSENET

# %% _kg_hide-output=true
base_model =  tf.keras.applications.DenseNet201(include_top=False, weights='imagenet')
base_model.summary()

# %% _kg_hide-output=true
names = ['conv3_block8_concat', 'conv5_block26_concat', 'conv5_block28_concat']

layers = [base_model.get_layer(name).output for name in names]

#create feature extracion model
dense_model = tf.keras.Model(base_model.input, layers)
dense_model.summary()


# %%
def run_deepdream(model, image, steps=100, step_size=0.01):
    # convert from unit8 to range expected by model
    image = tf.keras.applications.densenet.preprocess_input(image)
    
    for s in range(steps):
        loss, image = deepdream(model, image, step_size)
        
        if s%100==0:
            # plt.figure(figsize=(8,8))
            # plt.imshow(deprocess(image))
            # plt.title("STEP {}, LOSS {}".format(s, loss))
            # plt.show()
            print("STEP {}, LOSS {}".format(s, loss))
            
   # plt.figure(figsize=(8,8))
   # plt.imshow(deprocess(image))
   # plt.show()
    
    return deprocess(image)



# %% _kg_hide-output=true
# name of folder
dream_name = 'dense_dream'
path = './' + dream_name

if not os.path.exists(path):
    os.makedirs(path)

# saving image in one dir to make video
img_0.save(path + '/img_0.jpg', "JPEG", quality=80, optimize=True, progressive=True)

save_dream(dense_model, path)

# %%
get_video('./dense_dream', 'densedream.mp4')

# %%
display.YouTubeVideo(id='S2z_UO0L3Aw', height=605, width=910)

# %% [markdown]
# Final words i creted this dream using Inception and Dense Nets you can try any other model like VGG or MobileNet, or use your own model and train it on a dataset og your chosing and create awesome dreams. Can't wait to see other Notebooks
#
# * Your feedback in comments is much appreciated, Comment if you have any doubts or for inprovement
# * Please **UPVOTE if you LIKE this notebook**, it will keep me motivated
#
# **REFRENCES**
# * https://www3.cs.stonybrook.edu/~cse352/T12talk.pdf
# * https://www.topbots.com/advanced-topics-deep-convolutional-neural-networks/
# * https://wccftech.com/nvidia-demo-skynet-gtc-2014-neural-net-based-machine-learning-intelligence/
# * https://gizmodo.com/these-are-the-incredible-day-dreams-of-artificial-neura-1712226908
# * https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
# * SuperDataScience
