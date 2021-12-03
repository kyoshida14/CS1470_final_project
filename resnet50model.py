import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

'''class ResNet50Modded(tf.keras.Model):
    def __init__(self, block=100, num_classes=2):
        super(ResNet50Modded, self).__init__()

        #self.resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False)
        #self.outputLayer = tf.keras.layers.Dense(2, activation='softmax', name="classifier")'''

        


def createModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, pooling = "max"))
    model.add(tf.keras.layers.Dense(2, activation='softmax', name="classifier"))

    #m = model()
    model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

