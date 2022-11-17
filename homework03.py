import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense

# load data
(train_ds, test_ds), ds_info = tfds.load("mnist" , split =["train", "test"] , as_supervised=True , with_info=True)

# print(ds_info)
print("\n\n\n\n")

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
    mnist = mnist.batch(32)

    # prepares next 10 datapoints for pipelining
    mnist = mnist.prefetch(10)

    return mnist


# Model Class which models our NN
class HomeWorkModel(tf.keras.Model):

    # Constructor, in which we define our Layers
    # input Layer isnt given since we feed our Class with Input in call step
    def __init__(self):
        super(HomeWorkModel, self).__init__()

        # activation = relu: makes sense here, because we want our perceptrons to 
        # be either activated or not. Relu implements this behaviour
        self.first_hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.second_hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu)

        # activate = softmax: in our one-hot vector we dont want an activated or not,
        # we get an output [-1, 1] which sums up to 1
        # this fits well with our desire to choose the best fitting number
        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    # forward step, pass inputs through every layer
    @tf.function
    def call(self, inputs):
        x = self.first_hidden(inputs)
        x = self.second_hidden(x)
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

# idk
#tf.keras_backend.clear_session()

# prepare data
test_ds = test_ds.apply(prepare_data)
train_ds = train_ds.apply(prepare_data)

# define number of epochs and learning_rate
epochs = 10
learning_rate = 0.1

# instantiate model
model = HomeWorkModel()

# define loss function
# Documentation says we are supposed to use this function if we get in data
# and try to output a one-hot-vector, which represents categorys
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

# instantiate optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate)

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
test_losses.append(train_loss)


# training_loop
for epoch in range(epochs):
    print(f'Epoch: {epoch} starts with accuracy: {test_accuracies[-1]}')
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

print(f"after Training, loss: {train_loss}, accuracy: {test_accuracy}")

# visualize the test_losses, test_accuracies and train_losses over time
# in the .pdf we are told to visualize train_accuracy aswell but that didnt make sense to us
def visualization (test_losses, test_accuracies, train_losses):
    plt.figure()
    line1 = plt.plot(train_losses, "b")
    line2 = plt.plot(test_losses, "r" )
    line3 = plt.plot(test_accuracies, "g" )
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3),("training loss", " testloss" , "test accuracy"))
    plt.show()


visualization(test_losses, test_accuracies, train_losses)

