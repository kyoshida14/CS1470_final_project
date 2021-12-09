import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from readData import readData
from resnet50model import createModel
from matplotlib import pyplot as plt
import csv


#Created to train the recoded ResNet. To train the modified KerasResNet resnet and 
def train(model, train_inputs, train_labels, batch_size):

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

def loss(prob, labels):

        print(logits)
        print(labels)
        model_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, prob, from_logits=False))

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
    x2 = [i for i in range(len(accuracy))]
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

    model =  createModel(); 
    print('model loaded')
    epochNum = 1
    losses = []
    accuracies = []

    for ep in range(1):
        for photoStart in range(19): #20,000 images in each category, load 1000 per category due to memory issues 

            index = photoStart * 1000
            print( "Index start in training: " + str(index) )
            (train_Rdog, train_Fdog) = readData('data/progan/person/0_real', 'data/progan/person/1_fake', index) #readData('data/progan/dog/0_storage', 'data/progan/dog/1_storage', index)
            (train_Rcat, train_Fcat) = readData('data/progan/car/0_real', 'data/progan/car/1_fake', index) #readData('data/progan/cat/0_storage', 'data/progan/cat/1_storage', index)
            dogLabels = tf.concat([tf.zeros(tf.shape(train_Rdog)[0]), tf.ones(tf.shape(train_Fdog)[0])], axis = 0)
            catLabels = tf.concat([tf.zeros(tf.shape(train_Rcat)[0]), tf.ones(tf.shape(train_Fcat)[0])], axis = 0)
            labelsHot = tf.concat([dogLabels, catLabels], axis = 0)
            trainTot = tf.concat([train_Rdog, train_Fdog, train_Rcat, train_Fcat], axis = 0)
            print('finishRead')
            print(tf.shape(labelsHot))
            print(tf.shape(trainTot))



            model.fit(trainTot, labelsHot, batch_size=100, epochs=epochNum)
            model.save('CS1470_final_project/model_part')
            #model.save_weights('CS1470_final_project/model_weightspart3')
            del train_Rdog, train_Fdog,train_Rcat, train_Fcat, dogLabels, catLabels, labelsHot,trainTot

            ndex = photoStart * 1000
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
            model.save('CS1470_final_project/model_part4')
            del train_Rdog, train_Fdog,train_Rcat, train_Fcat, dogLabels, catLabels, labelsHot,trainTot


            # test after training 1 batch epoch
            (test_Rdog, test_Fdog) = readData('data/testprogan/pottedplant/0_real', 'data/testprogan/pottedplant/1_fake', None)
            (test_Rcat, test_Fcat) = readData('data/testprogan/sheep/0_real', 'data/testprogan/sheep/1_fake', None)
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
    #model.save_weights('CS1470_final_project/model_weightsFinal')

    visualize_results(losses, accuracies)


    

    

if __name__ == '__main__':
    main()
