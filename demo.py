import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# used for the model created by Wang et al.
from networks.resnet import resnet50
import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
# from PIL import Image
from pytorch2keras import pytorch_to_keras

'''
input:
    img_path: e.g., 'examples/real.png'
    trained: use the weights that are only pretrained with imagenet (0=default), our model (1), or trained in the original paper (2)
'''
def demo(img_path, trained=1, model_path='weights/blur_jpg_prob0.5.pth'):
    if (trained==0):    # load in ResNet50 with pre-trained weights with imagenet
        base_model = ResNet50(weights='imagenet', include_top=False, classes=1, pooling='max')
        # add a global spatial average pooling layer
        x = base_model.output
        # add a fully-connected layer
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

        # transform images
        img = image.load_img(img_path)  #target_size=(256,256)
        x = image.img_to_array(img)     #(256, 256, 3)
        x = np.expand_dims(x, axis=0)   #(1, 256, 256, 3)
        x = preprocess_input(x)         #(1, 256, 256, 3)
        predictions = model.predict(x)
        print(predictions)
        return predictions

    elif (trained==1):      # use the model we trained
        # get demo images
        demo_data = readData1(img_path)
        # get model
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(demo_data)
        print(predictions)
        return predictions

    # currently with error
    elif (trained==2):   # convert pytorch model to keras
        base_model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu')
        base_model.load_state_dict(state_dict['model'])
        base_model
        base_model.eval()

        # create a dummy variable with correct shape, use it to trace the model
        input_np = np.random.uniform(0,256,(1,3,256,256))
        input_var = Variable(torch.FloatTensor(input_np))
        model = pytorch_to_keras(base_model,input_var,[(3,None,None)],verbose=True)    # converted to keras

        # transform images
        img = image.load_img(img_path)  #target_size=(256,256)
        x = image.img_to_array(img)     #(256, 256, 3)
        x = np.expand_dims(x, axis=0)   #(1, 256, 256, 3)
        x = preprocess_input(x)         #(1, 256, 256, 3)
        x = tf.transpose(x, [0, 3, 1, 2])   # reshape to (1, 3, 256, 256)
        accuracy = model(x)

        return accuracy


print(demo('examples/real.png', trained=0))
# print('probability of being synthetic: {:.2f}%'.format(prob * 100))

'''
error when trained==1:
    tensorflow.python.framework.errors_impl.UnimplementedError: The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW [Op:Conv2D]
'''
