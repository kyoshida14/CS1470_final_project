import os
import csv
import numpy as np
import tensorflow as tf

def eval(model_path):
    model = tf.keras.models.load_model(model_path)

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss:", results[0], " test accuracy:", results[1])

    print("Generate predictions for 5 samples")
    predictions = model.predict(x_test[:5])
    # print("predictions shape:", predictions.shape)
