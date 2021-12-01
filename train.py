import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from readData import readData




def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    
    '''missed_trials = train_inputs.size % model.window_size
    training_samples = np.reshape(train_inputs[0:train_inputs.size-missed_trials],(-1,20))
    training_labels = np.reshape(train_labels[0:train_labels.size-missed_trials],(-1,20))
    
        #print(training_samples)
    training_samples = training_samples.astype(int)
    training_labels = training_labels.astype(int)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) #?
    batch_count = np.int(np.floor(training_labels.shape[0] / model.batch_size))
    
    indices = tf.random.shuffle(range(training_samples.shape[0])) 
    randomized_inputs =tf.gather(training_samples, indices)
    randomized_labels = tf.gather(training_labels, indices)
    
    for batch in range(batch_count):
        cur_batch = randomized_inputs[batch *model.batch_size : batch  *model.batch_size + model.batch_size,:]
        cur_labels = randomized_labels[batch *model.batch_size:batch *model.batch_size + model.batch_size,:]
        #print(cur_batch)
        with tf.GradientTape() as tape:
            predictions, restB = model.call(cur_batch, None) # this calls the call function conveniently
            loss = model.loss(predictions, cur_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))'''

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for batch in range(batch_count):
        cur_batch = randomized_inputs[batch *model.batch_size : batch  *model.batch_size + model.batch_size,:]
        cur_labels = randomized_labels[batch *model.batch_size:batch *model.batch_size + model.batch_size,:]
        #print(cur_batch)
        with tf.GradientTape() as tape:
            predictions, restB = model.call(cur_batch, None) # this calls the call function conveniently
            loss = model.loss(predictions, cur_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, logits))

        return loss

def main():
    
    


    (train_Rdog, train_Fdog) = readData('progan_train/dog/0_real', 'progan_train/dog/1_fake')
    (train_Rcat, train_Fcat) = readData('progan_train/dog/0_real', 'progan_train/dog/1_fake')


    dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
    catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
    dogLabelsHot = tf.one_hot(dogLabels, 2)
    catLabelsHot = tf.one_hot(catLabels, 2)
    
    labelsHot = tf.concat([dogLabelsHot, catLabelsHot])
    trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat])
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False,  classes=2)

   
    pass

if __name__ == '__main__':
    main()
