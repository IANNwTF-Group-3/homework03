import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Layer(tf.Module):
    def __init__(self, n_units, activation=tf.nn.relu, name=None):
        super(Layer, self).__init__(name=name)
        self.activation = activation
        self.n_units = n_units
        self.name = name

        if self.name is None:
            self.name = "noname"

    @tf.function
    def __call__(self, inputs):
        return self.activation((inputs @ self.w) + self.b)

    def build(self, input_shape):
        with tf.name_scope(self.name):
            self.w = tf.Variable(tf.random.normal([input_shape[-1], self.n_units]), name="weights", trainable=True)
            self.b = tf.Variable(tf.zeros([self.n_units]), name="bias", trainable=True)
        

class Network(tf.Module):
    def __init__(self, layers, input_shape, name=None):
        super(Network, self).__init__(name=name)
        self.layers = layers

        input_shape = tf.reshape(tf.zeros(input_shape), [-1]).shape

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.build(input_shape)
            else:
                layer.build((self.layers[i-1].n_units,))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train(self, inputs, labels, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            predictions = self.forward(inputs)
            predictions = tf.math.softmax(tf.math.sigmoid(predictions))
            loss = loss_fn(labels, predictions)
            #loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


def prepare(ds: tf.data.Dataset):
    def reshape(image, label):
        return tf.reshape(image, [-1]), label
    ds = ds.map(reshape)
    
    def normalize(image, label):
        return (tf.cast(image, tf.float32) / 128) - 1., label
    ds = ds.map(normalize)

    def one_hot(image, label):
        return image, tf.one_hot(label, 10)
    ds = ds.map(one_hot)

    ds = ds.cache()
    ds = ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.batch(32)
    ds = ds.prefetch(20)

    return ds

def accuracy(labels, predictions):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(predictions, axis=1)), tf.float32))

# TODO diff metrics between test and train
def train(epochs, model: Network, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, loss_fn, optimizer, metrics):
    output = dict()
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model.forward(x)
                loss = loss_fn(y, logits)

            if "loss" in metrics:
                if not "train_loss" in output:
                    output["train_loss"] = []
                output["train_loss"].append(tf.reduce_mean(loss))

            if "accuracy" in metrics:
                if not "train_accuracy" in output:
                    output["train_accuracy"] = []
                output["train_accuracy"].append(accuracy(y, logits))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        for batch, (x, y) in enumerate(test_ds):
            predictions = model.forward(x)
            if "loss" in metrics:
                if not "test_loss" in output:
                    output["test_loss"] = []
                output["test_loss"].append(tf.reduce_mean(loss))

            if "accuracy" in metrics:
                if not "test_accuracy" in output:
                    output["test_accuracy"] = []
                output["test_accuracy"].append(accuracy(y, predictions))

        return output

if __name__ == '__main__':
    (train_ds , test_ds), ds_info = tfds.load('mnist', split =['train', 'test'], as_supervised=True, with_info=True)

    print(ds_info)

    # How many training/test images are there?
    print(ds_info.splits['train'].num_examples)
    print(ds_info.splits['test'].num_examples)

    # What’s the image shape?
    print(ds_info.features['image'].shape)

    # What range are pixel values in?
    print(ds_info.features['image'].dtype)
    print(tf.uint8.min)
    print(tf.uint8.max)

    train_ds = train_ds.apply(prepare)
    test_ds = test_ds.apply(prepare)

    layers = [
        Layer(256, name="hidden1"),
        #Layer(256, name="hidden2"),
        Layer(10, name="output", activation=tf.nn.softmax)
    ]

    network = Network(layers, ds_info.features['image'].shape, name="network")

    optimizer = tf.optimizers.SGD(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metrics = [ 'accuracy', 'loss' ]

    output = train(10, network, train_ds, test_ds, loss_fn, optimizer, metrics)

    plt.plot(output['train_loss'])
    plt.plot(output['train_accuracy'])
    plt.plot(output['test_loss'])
    plt.plot(output['test_accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'], loc='upper left')
    plt.show()


