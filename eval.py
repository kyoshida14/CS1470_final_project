import os
import csv
import numpy as np
import tensorflow as tf
from readData import readData

def eval(model_path, x_test, y_test):
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(x_test, y_test, batch_size=128)
    loss = results[0]
    accuracy = results[1]
    
    return (loss, accuracy)


def main():
    realDir = "deepfake/0_real"
    fakeDir = "deepfake/1_fake"
    (test_R, test_F) = readData(realDir, fakeDir, 0)
    testTot = tf.concat([test_R, test_F], axis = 0)
    testLabels = tf.concat([tf.zeros(tf.shape(test_R)[0]), tf.ones(tf.shape(test_F)[0])], axis = 0)

    print("Evaluate on test data")
    loss, accuracy = eval('CS1470_final_project/model_part4', testTot, testLabels)
    print("test loss:", loss, " test accuracy:", accuracy)


if __name__ == '__main__':
    main()
