import os
import csv
import numpy as np
import tensorflow as tf

def eval(model_path, test_path):
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(x_test, y_test, batch_size=128)
    loss = results[0]
    accuracy = results[1]
    
    return (loss, accuracy)


def main():
    print("Evaluate on test data")
    loss, accuracy = eval('CS1470_final_project/model_weights', ???)
    print("test loss:", loss, " test accuracy:", accuracy)


if __name__ == '__main__':
    main()
