#!/usr/bin/env python
# coding: utf-8

# # Face Recognition for the Happy House
# 
# Welcome to the first assignment of week 4! Here you will build a face recognition system. Many of the ideas presented here are from [FaceNet](https://arxiv.org/pdf/1503.03832.pdf). In lecture, we also talked about [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf). 
# 
# Face recognition problems commonly fall into two categories: 
# 
# - **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem. 
# - **Face Recognition** - "who is this person?". For example, the video lecture showed a face recognition video (https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem. 
# 
# FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.
#     
# **In this assignment, you will:**
# - Implement the triplet loss function
# - Use a pretrained model to map face images into 128-dimensional encodings
# - Use these encodings to perform face verification and face recognition
# 
# In this exercise, we will be using a pre-trained model which represents ConvNet activations using a "channels first" convention, as opposed to the "channels last" convention used in lecture and previous programming assignments. In other words, a batch of images will be of shape $(m, n_C, n_H, n_W)$ instead of $(m, n_H, n_W, n_C)$. Both of these conventions have a reasonable amount of traction among open-source implementations; there isn't a uniform standard yet within the deep learning community. 
# 
# Let's load the required packages. 
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import sys
import keras
from imutils import paths
import imutils
import pickle
from face_rec_model import who_is_it
from face_rec_model import who_is_it_image
from face_rec_model import verify
from face_rec_model import img_to_encoding
from face_rec_model import img_to_encoding_1
import face_recognition
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.set_printoptions(threshold=sys.maxsize)


# In[ ]:





# ## 0 - Naive Face Verification
# 
# In Face Verification, you're given two images and you have to tell if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person! 
# 
# <img src="images/pixel_comparison.png" style="width:380px;height:150px;">
# <caption><center> <u> <font color='purple'> **Figure 1** </u></center></caption>

# Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on. 
# 
# You'll see that rather than using the raw image, you can learn an encoding $f(img)$ so that element-wise comparisons of this encoding gives more accurate judgements as to whether two pictures are of the same person.

# ## 1 - Encoding face images into a 128-dimensional vector 
# 
# ### 1.1 - Using an ConvNet  to compute encodings
# 
# The FaceNet model takes a lot of data and a long time to train. So following common practice in applied deep learning settings, let's just load weights that someone else has already trained. The network architecture follows the Inception model from [Szegedy *et al.*](https://arxiv.org/abs/1409.4842). We have provided an inception network implementation. You can look in the file `inception_blocks.py` to see how it is implemented (do so by going to "File->Open..." at the top of the Jupyter notebook).  
# 

# The key things you need to know are:
# 
# - This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of $m$ face images) as a tensor of shape $(m, n_C, n_H, n_W) = (m, 3, 96, 96)$ 
# - It outputs a matrix of shape $(m, 128)$ that encodes each input face image into a 128-dimensional vector
# 
# Run the cell below to create the model for face images.

# In[2]:


print(keras.__version__)
FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# In[3]:


print("Total Params:", FRmodel.count_params())


# ** Expected Output **
# <table>
# <center>
# Total Params: 3743280
# </center>
# </table>
# 

# By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings the compare two face images as follows:
# 
# <img src="images/distance_kiank.png" style="width:680px;height:250px;">
# <caption><center> <u> <font color='purple'> **Figure 2**: <br> </u> <font color='purple'> By computing a distance between two encodings and thresholding, you can determine if the two pictures represent the same person</center></caption>
# 
# So, an encoding is a good one if: 
# - The encodings of two images of the same person are quite similar to each other 
# - The encodings of two images of different persons are very different
# 
# The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart. 
# 
# <img src="images/triplet_comparison.png" style="width:280px;height:150px;">
# <br>
# <caption><center> <u> <font color='purple'> **Figure 3**: <br> </u> <font color='purple'> In the next part, we will call the pictures from left to right: Anchor (A), Positive (P), Negative (N)  </center></caption>

# 
# 
# ### 1.2 - The Triplet Loss
# 
# For an image $x$, we denote its encoding $f(x)$, where $f$ is the function computed by the neural network.
# 
# <img src="images/f_x.png" style="width:380px;height:150px;">
# 
# <!--
# We will also add a normalization step at the end of our model so that $\mid \mid f(x) \mid \mid_2 = 1$ (means the vector of encoding should be of norm 1).
# !-->
# 
# Training will use triplets of images $(A, P, N)$:  
# 
# - A is an "Anchor" image--a picture of a person. 
# - P is a "Positive" image--a picture of the same person as the Anchor image.
# - N is a "Negative" image--a picture of a different person than the Anchor image.
# 
# These triplets are picked from our training dataset. We will write $(A^{(i)}, P^{(i)}, N^{(i)})$ to denote the $i$-th training example. 
# 
# You'd like to make sure that an image $A^{(i)}$ of an individual is closer to the Positive $P^{(i)}$ than to the Negative image $N^{(i)}$) by at least a margin $\alpha$:
# 
# $$\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2 + \alpha < \mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2$$
# 
# You would thus like to minimize the following "triplet cost":
# 
# $$\mathcal{J} = \sum^{m}_{i=1} \large[ \small \underbrace{\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2}_\text{(1)} - \underbrace{\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2}_\text{(2)} + \alpha \large ] \small_+ \tag{3}$$
# 
# Here, we are using the notation "$[z]_+$" to denote $max(z,0)$.  
# 
# Notes:
# - The term (1) is the squared distance between the anchor "A" and the positive "P" for a given triplet; you want this to be small. 
# - The term (2) is the squared distance between the anchor "A" and the negative "N" for a given triplet, you want this to be relatively large, so it thus makes sense to have a minus sign preceding it. 
# - $\alpha$ is called the margin. It is a hyperparameter that you should pick manually. We will use $\alpha = 0.2$. 
# 
# Most implementations also normalize the encoding vectors  to have norm equal one (i.e., $\mid \mid f(img)\mid \mid_2$=1); you won't have to worry about that here.
# 
# **Exercise**: Implement the triplet loss as defined by formula (3). Here are the 4 steps:
# 1. Compute the distance between the encodings of "anchor" and "positive": $\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2$
# 2. Compute the distance between the encodings of "anchor" and "negative": $\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2$
# 3. Compute the formula per training example: $ \mid \mid f(A^{(i)}) - f(P^{(i)}) \mid - \mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2 + \alpha$
# 3. Compute the full formula by taking the max with zero and summing over the training examples:
# $$\mathcal{J} = \sum^{m}_{i=1} \large[ \small \mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2 - \mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2+ \alpha \large ] \small_+ \tag{3}$$
# 
# Useful functions: `tf.reduce_sum()`, `tf.square()`, `tf.subtract()`, `tf.add()`, `tf.maximum()`.
# For steps 1 and 2, you will need to sum over the entries of $\mid \mid f(A^{(i)}) - f(P^{(i)}) \mid \mid_2^2$ and $\mid \mid f(A^{(i)}) - f(N^{(i)}) \mid \mid_2^2$ while for step 4 you will need to sum over the training examples.

# In[4]:


# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),axis=-1, keep_dims=True)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),axis=-1, keep_dims=True)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **loss**
#         </td>
#         <td>
#            528.143
#         </td>
#     </tr>
# 
# </table>

# ## 2 - Loading the trained model
# 
# FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data and a lot of computation, we won't train it from scratch here. Instead, we load a previously trained model. Load a model using the following cell; this might take a couple of minutes to run. 

# In[5]:


print('Loading model')
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
print('Model ready')


# Here're some examples of distances between the encodings between three individuals:
# 
# <img src="images/distance_matrix.png" style="width:380px;height:200px;">
# <br>
# <caption><center> <u> <font color='purple'> **Figure 4**:</u> <br>  <font color='purple'> Example of distance outputs between three individuals' encodings</center></caption>
# 
# Let's now use this model to perform face verification and face recognition! 

# ## 3 - Applying the model

# Back to the Happy House! Residents are living blissfully since you implemented happiness recognition for the house in an earlier assignment.  
# 
# However, several issues keep coming up: The Happy House became so happy that every happy person in the neighborhood is coming to hang out in your living room. It is getting really crowded, which is having a negative impact on the residents of the house. All these random happy people are also eating all your food. 
# 
# So, you decide to change the door entry policy, and not just let random happy people enter anymore, even if they are happy! Instead, you'd like to build a **Face verification** system so as to only let people from a specified list come in. To get admitted, each person has to swipe an ID card (identification card) to identify themselves at the door. The face recognition system then checks that they are who they claim to be. 

# ### 3.1 - Face Verification
# 
# Let's build a database containing one encoding vector for each person allowed to enter the happy house. To generate the encoding we use `img_to_encoding(image_path, model)` which basically runs the forward propagation of the model on the specified image. 
# 
# Run the following code to build the database (represented as a python dictionary). This database maps each person's name to a 128-dimensional encoding of their face.

# In[6]:


database = {}
#face_encodings(face_image, known_face_locations=None, num_jitters=1):
#レオン

#Diego
database["diego"] = img_to_encoding("images/diego/diego_1.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_2.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_3.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_4.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_5.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_6.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_7.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_8.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_9.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_10.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_11.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_12.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_13.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_14.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_15.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_16.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_17.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_18.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_19.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_20.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_21.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_22.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_23.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_24.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_25.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_26.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_27.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_28.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_29.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_30.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_31.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_32.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_33.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_34.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_35.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_36.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_37.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_38.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_39.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_40.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_41.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_42.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_43.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_44.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_45.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_46.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_47.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_48.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_49.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_50.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_51.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_52.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_53.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_54.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_55.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_56.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_57.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_58.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_59.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_60.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_61.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_62.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_63.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_64.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_65.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_66.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_67.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_68.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_69.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_70.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_71.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_72.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_73.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_74.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_75.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_76.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_77.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_78.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_79.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_80.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_81.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_82.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_83.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_84.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_85.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_86.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_87.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_88.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_89.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_90.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_91.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_92.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_93.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_94.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_95.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_96.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_97.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_98.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_99.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_100.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_101.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_102.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_103.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_104.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_105.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_106.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_107.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_108.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_109.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_110.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_111.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_112.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_113.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_114.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_115.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_116.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_117.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_118.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_119.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_120.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_121.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_122.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_123.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_124.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_125.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_126.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_127.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_128.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_129.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_130.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_131.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_132.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_133.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_134.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_135.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_136.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_137.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_138.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_139.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_140.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_141.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_142.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_143.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_144.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_145.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_146.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_147.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_148.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_149.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_150.jpg", FRmodel)


#Greg
database["Toyota"] = img_to_encoding("images/gregory/gregory_1.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_2.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_3.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_4.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_5.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_6.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_7.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_8.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_9.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_10.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_11.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_12.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_13.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_14.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_15.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_16.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_17.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_18.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_19.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_20.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_21.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_22.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_23.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_24.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_25.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_26.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_27.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_28.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_29.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_30.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_31.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_32.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_33.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_34.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_35.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_36.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_37.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_38.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_39.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_40.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_41.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_42.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_43.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_44.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_45.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_46.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_47.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_48.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_49.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_50.jpg", FRmodel)
#daniel
database["daniel"] = img_to_encoding("images/daniel/daniel_1.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_2.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_3.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_4.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_5.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_6.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_7.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_8.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_9.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_10.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_11.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_12.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_13.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_14.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_15.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_16.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_17.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_18.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_19.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_20.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_21.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_22.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_23.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_24.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_25.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_26.jpg", FRmodel)
'''
#Yamagishi
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_1.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_2.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_3.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_4.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_5.jpg", FRmodel)

database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_6.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_7.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_8.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_9.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_10.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_11.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_12.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_13.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_14.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_15.jpg", FRmodel)
database["yamagishi"] = img_to_encoding("images/yamagishi/yamagishi_16.jpg", FRmodel)
'''


# Now, when someone shows up at your front door and swipes their ID card (thus giving you their name), you can look up their encoding in the database, and use it to check if the person standing at the front door matches the name on the ID.
# 
# **Exercise**: Implement the verify() function which checks if the front-door camera picture (`image_path`) is actually the person called "identity". You will have to go through the following steps:
# 1. Compute the encoding of the image from image_path
# 2. Compute the distance about this encoding and the encoding of the identity image stored in the database
# 3. Open the door if the distance is less than 0.7, else do not open.
# 
# As presented above, you should use the L2 distance (np.linalg.norm). (Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.) 

# Diego is trying to enter the Happy House and the camera takes a picture of him ("images/test_2.jpg"). Let's run your verification algorithm on this picture:
# 
# <img src="images/test_2.jpg" style="width:100px;height:100px;">

# In[7]:


#verify("images/leon/leon_34.jpg", "Leon", database, FRmodel)
print('Hola mundo')


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **It's Diego, welcome home!**
#         </td>
#         <td>
#            (0.65939283, True)
#         </td>
#     </tr>
# 
# </table>

# bertrand, who broke the aquarium last weekend, has been banned from the house and removed from the database. He stole Andrew's ID card and came back to the house to try to present himself as Kian. The front-door camera took a picture of Benoit ("images/camera_1.jpg). Let's run the verification algorithm to check if benoit can enter.
# <img src="images/diego/diego_29.jpg" style="width:100px;height:100px;">

# In[8]:


verify("images/diego/diego_29.jpg", "diego", database, FRmodel)


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **It's not Andrew, please go away puto**
#         </td>
#         <td>
#            (0.86224014, False)
#         </td>
#     </tr>
# 
# </table>

# ### 3.2 - Face Recognition
# 
# Your face verification system is mostly working well. But since Kian got his ID card stolen, when he came back to the house that evening he couldn't get in! 
# 
# To reduce such shenanigans, you'd like to change your face verification system to a face recognition system. This way, no one has to carry an ID card anymore. An authorized person can just walk up to the house, and the front door will unlock for them! 
# 
# You'll implement a face recognition system that takes as input an image, and figures out if it is one of the authorized persons (and if so, who). Unlike the previous face verification system, we will no longer get a person's name as another input. 
# 
# **Exercise**: Implement `who_is_it()`. You will have to go through the following steps:
# 1. Compute the target encoding of the image from image_path
# 2. Find the encoding from the database that has smallest distance with the target encoding. 
#     - Initialize the `min_dist` variable to a large enough number (100). It will help you keep track of what is the closest encoding to the input's encoding.
#     - Loop over the database dictionary's names and encodings. To loop use `for (name, db_enc) in database.items()`.
#         - Compute L2 distance between the target "encoding" and the current "encoding" from the database.
#         - If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

# In[11]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #face_locations = face_recognition.face_locations(frame)    
    
    for (x, y, w, h) in faceRects:
        roi = frame[y:y+h,x:x+w]
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi,(96, 96))
        #min_dist = 1000
        faces = list(database.keys())
        detected  = False
        
        for face in range(len(faces)):
            person = faces[face]
                            #verify("roi", "diego", database, FRmodel)
                            #verify(image_path, identity, database, model): dist, door_open
                            #who_is_it(image_path, database, model): min_dist, identity
            #dist, detected = verify(roi, person, database[person], FRmodel)
            min_dist, identity = who_is_it(roi, database, FRmodel)
        #print(identity)
        if min_dist > 0.8:
            name = 'Unknown'
        else:
            name = identity
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
        #cv2.putText(frame, identity, (x+ (w//2),y-2), font, 1.0, (0, 0, 255), 1)
        '''else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.putText(frame, 'nO', (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)'''
    print('Distancia minima: ',min_dist)
    cv2.imshow('frame', frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Your Happy House is running well. It only lets in authorized persons, and people don't need to carry an ID card around anymore! 
# 
# You've now seen how a state-of-the-art face recognition system works.
# 
# Although we won't implement it here, here're some ways to further improve the algorithm:
# - Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then given a new image, compare the new face to multiple pictures of the person. This would increae accuracy.
# - Crop the images to just contain the face, and less of the "border" region around the face. This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.
# 

# <font color='blue'>
# **What you should remember**:
# - Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
# - The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
# - The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person. 

# Congrats on finishing this assignment! 
# 

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
