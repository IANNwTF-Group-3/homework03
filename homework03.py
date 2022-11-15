import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.layers import Dense


(train_ds, test_ds), ds_info = tfds.load("mnist" , split =["train", "test"] , as_supervised=True , with_info=True)

#print("\n\n\n", ds_info)
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

def prepare_data(mnist):

    """
    In your first lambda mapping you want to change the
    datatype from uint8 to tf.float values
    """
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    """
    To feed your network the
    28x28 images also need to be flattened. Check out the reshape function 5 , and if
    you want to minimize your work, try and understand how it interacts with size
    elements set to the value -1 (infering the remainder shape)
    """
    mnist = mnist.map(lambda img, target: (tf.reshape(img, [-1]), target))


    """
    Generally this means bringing the input close to the standart normal (gaussian) distribution
    with µ = 0 and σ = 1, however we can make a quick approximation as that:
    Knowing the inputs are in the 0-255 interval, we can simply divide all numbers
    by 128 (bringing them between 0-2), and finally subtracting one (bringing them
    into -1 to 1 range)
    """
    mnist = mnist.map(lambda img, target: ((img/128. -1), target))


    """
    Additionally you need to encode your labels as one-hot-vectors
    """
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))

    return mnist


test_ds = test_ds.apply(prepare_data)
train_ds = train_ds.apply(prepare_data)


"""
Now that you have your data pipeline built, it is time to create your network.
Check out the courseware for how to go about building a network with Tensor-
Flow’s Keras. Following that method, we want you to build a fully connected
feed-forward neural network to classify MNIST images with. To do this, have a
look at ’Dense’ layers 7 ; they basically provide you with the same functionality
as the ’Layer’ class which you have implemented last week. TensorFlow also
provides you with every activation function you might need for this course 8 . A
good (albeit arbitrary) starting point would be to have two hidden layers with
256 units each. For your output layer, think about how many units you need,
and consider which activation function is most appropriate for this task.
"""

class HomeWorkModel(tf.keras.Model):


    def __init__(self):
        super(HomeWorkModel, self).__init__()
        self.first_hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.second_hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    # forward step, pass inputs through every layer
    @tf.function
    def call(self, inputs):
        x = self.first_hidden(inputs)
        x = self.second_hidden(x)
        x = self.output(x)
        return x


# build Network with 2 Hidden 256 Perceptron layers:
# input layer
input_layer = Dense(n_units=28*28, activation_function=tf.nn.relu)
input_layer(tf.random.normal(shape=(5,4)))

# 1. hidden layer
first_hidden = Dense(n_units=28*28, activation_function=tf.nn.relu)
first_hidden(tf.random.normal(shape=(5,4)))

# 2. hidden layer
second_hidden = Dense(n_units=28*28, activation_function=tf.nn.relu)
second_hidden(tf.random.normal(shape=(5,4)))

# output layer
layer = Dense(n_units=28*28, activation_function=tf.nn.relu)
layer(tf.random.normal(shape=(5,4)))


print(tf.random.normal(shape=(5,4)))

print(layer.trainable_variables)

"""input_data = tf.random.uniform((0,28*28)) #train_ds
print(train_ds)
print("input: ", input_data)
print(model(input_data))"""
