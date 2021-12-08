import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from readData import readData
from resnet50model import createModel
from matplotlib import pyplot as plt
import csv



def train(model, train_inputs, train_labels, batch_size):
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

    batch_count = np.int(np.floor(train_inputs.shape[0] / batch_size))
    indices = tf.random.shuffle(range(train_inputs.shape[0]))
    randomized_inputs =tf.gather(train_inputs, indices)
    randomized_labels = tf.gather(train_labels, indices)
    print(tf.shape(randomized_inputs))
    for batch in range(batch_count):
        cur_batch = randomized_inputs[batch *batch_size : batch  *batch_size + batch_size,:,:,:]
        cur_labels = randomized_labels[batch *batch_size:batch *batch_size + batch_size,:]
        print( tf.shape(cur_batch))
        #print(cur_batch)
        with tf.GradientTape() as tape:
            predictions = model.call(cur_batch) # this calls the call function conveniently
            model_loss = loss(predictions, cur_labels)
        gradients = tape.gradient(model_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def loss(logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        print(logits)
        print(labels)
        model_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, logits))

        return(model_loss)

def visualize_results(losses, accuracies):
    # losses
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch (1000 images)')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('CS1470_final_project/loss.png')
    plt.show()

    # accuracies
    plt.plot(x2, accuracies)
    plt.title('Accuracy per batch (1000 images)')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.savefig('CS1470_final_project/accuracy.png')
    plt.show()

    # save the data in csv
    heading = ['Batch','Loss','Accuracy']
    rows = [[i, losses[i], accuracy[i]] for i in range(len(losses))]
    with open('CS1470_final_project/results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(heading)
        csvwriter.writerows(rows)


def main():

    '''(train_Rdog, train_Fdog) = readData('data/progan/dog/0_real', 'data/progan/dog/1_fake')
    (train_Rcat, train_Fcat) = readData('data/progan/cat/0_real', 'data/progan/cat/1_fake')
    dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
    catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
    labelsHot = tf.concat([dogLabels, catLabels], axis = 0)
    trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)

    #(train_Real, train_Fake) = readData('trainData/fake/', 'trainData/real/')
    print('finishRead')
    #dogLabelsHot = tf.zeros(tf.shape(train_Real)[0], dtype=tf.dtypes.int32)
    #catLabelsHot = tf.ones(tf.shape(train_Fake)[0], dtype=tf.dtypes.int32)
    #labelsHot = tf.concat([dogLabelsHot, catLabelsHot], axis = 0)

    #trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)
    #trainTot = tf.concat([train_Real, train_Fake], axis = 0)

    #modelcreator = ResNet50Modded()#tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False,  classes=1, pooling = "max")
    print(tf.shape(labelsHot))
    print(tf.shape(trainTot))

    model = createModel()
    model.fit(trainTot, labelsHot, batch_size=100, epochs=10)


    model.save('CS1470_final_project/model_part1')
    model.save_weights('CS1470_final_project/model_weightspart1')

    del (train_Rdog, train_Fdog,train_Rcat, train_Fcat, dogLabels, catLabels, labelsHot,trainTot)'''
    model = tf.keras.models.load_model('CS1470_final_project/model_part2')
    print('model loaded')
    epochNum = 1
    losses = []
    accuracies = []

    for ep in range(5):
        for photoStart in range(20):

            index = photoStart * 1000
            print( "Index start in training: " + str(index) )
            (train_Rdog, train_Fdog) = readData('data/progan/dog/0_storage', 'data/progan/dog/1_storage', index) #readData('data/progan/dog/0_storage', 'data/progan/dog/1_storage', index)
            (train_Rcat, train_Fcat) = readData('data/progan/cat/0_storage', 'data/progan/cat/1_storage', index) #readData('data/progan/cat/0_storage', 'data/progan/cat/1_storage', index)
            dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
            catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
            labelsHot = tf.concat([dogLabels, catLabels], axis = 0)
            trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)
            print('finishRead')
            print(tf.shape(labelsHot))
            print(tf.shape(trainTot))



            model.fit(trainTot, labelsHot, batch_size=100, epochs=epochNum)


            model.save('CS1470_final_project/model_part2')
            model.save_weights('CS1470_final_project/model_weightspart2')
            del train_Rdog, train_Fdog,train_Rcat, train_Fcat, dogLabels, catLabels, labelsHot,trainTot

            # test after training 1 epoch
            (test_Rdog, test_Fdog) = readData('data/testprogan/dog/0_real', 'data/testprogan/dog/1_fake', None)
            (test_Rcat, test_Fcat) = readData('data/testprogan/cat/0_real', 'data/testprogan/cat/1_fake', None)
            dogLabels = tf.concat([tf.zeros(tf.shape(test_Rdog)[0]), tf.ones(tf.shape(test_Fdog)[0])], axis = 0)
            catLabels = tf.concat([tf.zeros(tf.shape(test_Rcat)[0]), tf.ones(tf.shape(test_Fcat)[0])], axis = 0)
            labelsHot = tf.concat([dogLabels, catLabels], axis = 0)
            trainTot = tf.concat([test_Rdog, test_Fdog, test_Rcat, test_Fcat], axis = 0)

            print("Evaluate on test data")
            results = model.evaluate(trainTot,labelsHot, batch_size=100)
            loss = results[0]
            accuracy = results[1]
            losses.append(loss)
            accuracies.append(accuracy)
            print("test loss:", loss, " test accuracy:", accuracy)

            del test_Rdog, test_Fdog,test_Rcat, test_Fcat, dogLabels, catLabels, labelsHot,trainTot

    model.save('CS1470_final_project/model_Final')
    model.save_weights('CS1470_final_project/model_weightsFinal')

    visualize_results(losses, accuracies)


    #(train_Real, train_Fake) = readData('trainData/fake/', 'trainData/real/')

    #dogLabelsHot = tf.zeros(tf.shape(train_Real)[0], dtype=tf.dtypes.int32)
    #catLabelsHot = tf.ones(tf.shape(train_Fake)[0], dtype=tf.dtypes.int32)
    #labelsHot = tf.concat([dogLabelsHot, catLabelsHot], axis = 0)

    #trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)
    #trainTot = tf.concat([train_Real, train_Fake], axis = 0)

    #modelcreator = ResNet50Modded()#tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False,  classes=1, pooling = "max")




    (train_Rdog, train_Fdog) = readData('data/progan/dog/0_test', 'data/progan/dog/1_test')
    (train_Rcat, train_Fcat) = readData('data/progan/cat/0_test', 'data/progan/cat/1_test')
    dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
    catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
    labelsHot = tf.concat([dogLabels, catLabels], axis = 0)
    trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)


    model.test_on_batch(trainTot, labelsHot)

    '''(train_Rdog, train_Fdog) = readData('dataTemp/progan/dog/0_real', 'dataTemp/progan/dog/1_fake')
    (train_Rcat, train_Fcat) = readData('dataTemp/progan/cat/0_real', 'dataTemp/progan/cat/1_fake')
    print ('finished read')


    dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
    catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
    dogLabelsHot = tf.one_hot(dogLabels, 2)
    catLabelsHot = tf.one_hot(catLabels, 2)

    labelsHot = tf.concat([dogLabelsHot, catLabelsHot], axis = 0)
    trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)

    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False,  classes=2)
    batch_size = 100
    for epoch in range(1):
    	train(model, trainTot, labelsHot, batch_size)
    model.save('model_weights')'''

    pass

if __name__ == '__main__':
    main()
