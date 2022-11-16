import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'mse']
    )

    model.fit(train_ds, epochs=2)

    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper left')
    plt.show()

    model.evaluate(test_ds)

    for image, label in test_ds.take(1):
        image = image.numpy()
        label = label.numpy()

        predictions = model.predict(image)

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image[i].reshape(28, 28), cmap=plt.cm.binary)
            plt.xlabel(np.argmax(predictions[i]))
        plt.show()
