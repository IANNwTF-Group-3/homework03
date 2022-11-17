# disable compiler warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# imports 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense
from typing import List

# load data
(train_ds_global, test_ds_global), ds_info = tfds.load("mnist" , split =["train", "test"] , as_supervised=True , with_info=True)

# print(ds_info)

"""
• How many training/test images are there?
    testimages:  10000
    trainimages: 60000

• What’s the image shape?
    'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),

• What range are pixel values in?
    uint8, [0,255]  [2^0, 2^8-1]
"""
# default batch_size value
batch_size_global = 30

# transforms the data into a set which can be interpreted by NN 
def prepare_data(mnist):

    # change datatype of pixels into float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # changes (28, 28) img format into (28*28, ) vector
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    # pixel value range reset from [0, 255] -> [-1, 1]
    mnist = mnist.map(lambda img, target: ((img/128. -1), target))

    # target shouldnt be a floating point number
    # target should give 10 outputs which are interpreted as likeness of every
    # digit from 0-9
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))

    # use cache for small datasets
    mnist = mnist.cache()

    # shuffles the dataset
    mnist = mnist.shuffle(1000)

    # take 32 imgs at the same time for efficiency
    mnist = mnist.batch(batch_size_global)

    # prepares next 10 datapoints for pipelining
    mnist = mnist.prefetch(10)

    return mnist


# Model Class which models our NN
class HomeWorkModel(tf.keras.Model):

    # Constructor, in which we define our Layers
    # input Layer isnt given since we feed our Class with Input in call step
    def __init__(self, layer_data):
        super(HomeWorkModel, self).__init__()

        # gets in layer data like [256, 256]
        # gets interpreted as 2 hidden layers with 256 units
        self.hidden_layers = []
        for n_units in layer_data:
            self.hidden_layers.append(tf.keras.layers.Dense(n_units, activation=tf.nn.relu))

        # activate = softmax: in our one-hot vector we dont want an activated or not,
        # we get an output [-1, 1] which sums up to 1
        # this fits well with our desire to choose the best fitting number
        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    # forward step, pass inputs through every layer
    @tf.function
    def call(self, inputs):
        
        #define x as inputs
        x = inputs

        # iterate through the layers and pass input through network
        for layer in self.hidden_layers:
            x = layer(x)

        # pass through output layer
        x = self.output_layer(x)
        return x


# performs 1 training_step in which we feed an input to the Network
# get a prediction, compute a loss and with the tf.GradientTape()
# we can easily compute the loss, gradients and implement backpropagation
# with the optimizer.
# returns loss to keep track
def training_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# takes current state of our model and computes loss/accuracys
def test(model, test_data, loss_function):
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    # iterate over test_data
    for (input, target) in test_data:
        # compute prediction
        prediction = model(input)

        # compute loss of target and prediction
        sample_test_loss = loss_function(target, prediction)

        # compute vector, which represents how often prediction and target are the same
        # tale mean to get accuracy
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)    
        sample_test_accuracy = np.mean(sample_test_accuracy)

        # keep track of loss/accuracy
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    # compute overall lost/accuracy by meaning over the whole list
    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

# seems to be unnecessary due to tf version
#tf.keras_backend.clear_session()



#epochs, learning rate, batch size, number/size of layers, optimizer
def training(epochs, learning_rate, batch_size, layer_data : List, optimizer):

    # change batch size
    batch_size_global = batch_size

    # prepare data
    test_ds = test_ds_global.apply(prepare_data)
    train_ds = train_ds_global.apply(prepare_data)

    # instantiate model
    model = HomeWorkModel(layer_data)

    # define loss function
    # Documentation says we are supposed to use this function if we get in data
    # and try to output a one-hot-vector, which represents categorys
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

    # instantiate lists to keep track of whats happening
    train_losses = []
    test_losses = []
    test_accuracies = []

    # values of test with test_data before training
    test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # values of test with training_data before training
    train_loss, _ = test(model, train_ds, cross_entropy_loss)
    train_losses.append(train_loss)

    # training_loop
    for epoch in range(epochs):
        #print(f'Epoch: {epoch} starts with accuracy: {test_accuracies[-1]}')
        epoch_loss_agg = []

        # iterate over training dataset
        for input, target in train_ds:

            # perform a training step, keep track of loss
            # the real training with backpropagation happens in the training_step()
            train_loss = training_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # calculate statistics
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # test values after training the network
    train_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)

    return train_losses, test_losses, test_accuracies


# train network with given attributes
# train_losses, test_losses, test_accuracies = training(epochs=3, learning_rate=0.1, batch_size=30,layer_data=[255, 255], optimizer=tf.keras.optimizers.SGD(learning_rate))


"""
ANALYZE PARAMETERS
"""
# define learning_rate since its 
learning_rate = 0.01
# instantiate optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate)

# keep track of statistics
data_dict_list = []

# iterate through these 
possible_dict = {"epoch": [2, 6],
                "learning_rate": [0.1, 0.4],
                "batch_size": [15, 45],
                "layer_data": [[], [32],[256, 256], [512, 512, 512]],
                "optimizer": ["SGD", "RMSprop"],
                }

# iterate throguh every combination to analyze
for epoch in possible_dict["epoch"]:
    for learning_rate in possible_dict["learning_rate"]:
        for optimizer_str in possible_dict["optimizer"]:

            # define optimizier here because it depends on the learning rate
            if optimizer_str == "SGD":
                optimizer = optimizer = tf.keras.optimizers.SGD(learning_rate)
            elif optimizer_str == "RMSprop":
                optimizer = optimizer = tf.keras.optimizers.RMSprop(learning_rate)

            for batch_size in possible_dict["batch_size"]:
                for layer_data in possible_dict["layer_data"]:

                    # train with given parameters
                    train_losses, test_losses, test_accuracies = training(epochs=epoch,
                                                                        learning_rate=learning_rate,
                                                                        batch_size=batch_size,
                                                                        layer_data=layer_data,
                                                                        optimizer=optimizer
                                                                        )

                    # keep track of losses/accuracy
                    data_dict = {"epoch": epoch,
                                 "learning_rate": learning_rate,
                                 "batch_size": batch_size,
                                 "layer_data": layer_data,
                                 "optimizer": optimizer_str,
                                 "test_accuracy": test_accuracies[-1].numpy(),
                                 "train_loss": train_losses[-1].numpy(),
                                 "test_loss": test_losses[-1].numpy()
                                }

                    data_dict_list.append(data_dict)

                    print(f"train_losses: epoch: {epoch} learningrate: {learning_rate} optimizier: {optimizer_str} batch_size:  {batch_size} layer_data: {layer_data} test_losses: {train_losses[-1].numpy()} \n")

                    print(f"test_losses: epoch: {epoch} learningrate: {learning_rate} optimizier: {optimizer_str} batch_size:  {batch_size} layer_data: {layer_data} test_losses: {test_losses[-1].numpy()}\n")

                    print(f"test_accuracies: epoch: {epoch} learningrate: {learning_rate} optimizier: {optimizer_str} batch_size:  {batch_size} layer_data: {layer_data} test_losses: {test_accuracies[-1].numpy()}\n\n")

# find best attribute value for criteria
parameters = ["epoch", "learning_rate", "optimizer", "batch_size", "layer_data"]

# create initial nested dict
# 1st key is parameter, 2nd key is value
statistic_dict = {}
for para in parameters:
    statistic_dict[str(para)] = {}
    for possible in possible_dict[str(para)]:
        statistic_dict[str(para)][str(possible)] = 0

num_results = len(data_dict_list)

# iterate through results
for result in data_dict_list:
    # iterate through the parameters
    for param in parameters:

        # param: what parameter we are inspecting
        # result[param] attribute value of the inspected parameter
        # num_results / len(possible_dict[param]) -> how often is this attribute value in the possible_dict_list?
        #                                        how often do we add on statistic_dict[param][result[param]]?
        # result["test_accuracy"] / (num_results / len(possible_dict[param])) adds up to mean of attribute value
        statistic_dict[str(param)][str(result[param])] += result["test_accuracy"] / (num_results / len(possible_dict[param]))

# print out our results
# bad avg test_accuracys come from bad learning examples whose score takes account into the mean value
for param in statistic_dict:
    print(f"{param}:")
    for value in statistic_dict[str(param)]:
        print(f"{statistic_dict[str(param)]}: {statistic_dict[str(param)][str(value)]}")
    print("\n")

items = [x["test_accuracy"] for x in data_dict_list]
max_val = max(items)
min_val = min(items)
print(f"max value: {data_dict_list[items.index(max_val)]}")
print(f"min value: {data_dict_list[items.index(min_val)]}")



# visualize the test_losses, test_accuracies and train_losses over time
# in the .pdf we are told to visualize train_accuracy aswell but that didnt make sense to us
def visualization (test_losses, test_accuracies, train_losses):
    plt.figure()
    line1 = plt.plot(train_losses, "b", label="train_losses")
    line2 = plt.plot(test_losses, "r", label="test_losses" )
    line3 = plt.plot(test_accuracies, "g", label="test_accuracies" )
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.show()


#visualization(test_losses, test_accuracies, train_losses)

