# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# Number of training examples
n_train = X_train.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
import random
import csv


# function to plot figures
def draw_images(figures, nrows=1, ncols=1, labels=None):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 14))
    axs = axs.ravel()
    for index, title in zip(range(len(figures)), figures):
        axs[index].imshow(figures[title], plt.gray())
        if (labels != None):
            axs[index].set_title(labels[index])
        else:
            axs[index].set_title(title)

        axs[index].set_axis_off()

    plt.tight_layout()


# reading values from csv file to make key value pairs of class id with the description
class_ids_key_value = np.genfromtxt('signnames.csv', skip_header=1, dtype=[('myint', 'i8'), ('mysring', 'S55')],
                                    delimiter=',')

# choosing random 8 images & labels to plot for a quick look
take_till = 8  # to show 8 figures
figures = {}
labels = {}
for i in range(take_till):
    index = random.randint(0, n_train - 1)
    labels[i] = class_ids_key_value[y_train[index]][1].decode('ascii')
    figures[i] = X_train[index]

# plotting the figures with the labels
draw_images(figures, 4, 2, labels)

train_class_ids, train_class_count = np.unique(y_train, return_counts=True)
plt.bar(train_class_ids, train_class_count)
plt.grid()
plt.title("Training Dataset class-wise distribution")
plt.show()

test_class_ids, counts_test = np.unique(y_test, return_counts=True)
plt.bar(test_class_ids, counts_test)
plt.grid()
plt.title("Test Dataset class-wise distribution")
plt.show()

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(unique_valid, counts_valid)
plt.grid()
plt.title("Valid Dataset class-wise distribution")
plt.show()

# yuv = np.array([[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]])
# X_train_yuv = X_train*yuv
# from skimage import color

# X_train_yuv = color.convert_colorspace(X_train, 'RGB', 'YUV')
# X_train_yuv = color.rgb2yuv(X_train)
# X_train_yuv = color.rgb2yuv(X_train)
# X_train_y =  X_train_yuv[0:,:,]

# number_to_stop = 8
# figures = {}
# for i in range(number_to_stop):
#     index = random.randint(0, n_train-1)
#     print(name_values[y_train[index]])
#     figures[y_train[index]] = X_train_yuv[index]

# plot_figures(figures, 2, 4)
# print(X_train_y)

# X_train = X_train_yuv


### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# some imports which we need
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from math import ceil
from sklearn.utils import shuffle

# Here we converting the dataset to grayscale for further use
X_train_rgb = X_train
X_train_grayscale = np.sum(X_train / 3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_graycale = np.sum(X_test / 3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_grayscale = np.sum(X_valid / 3, axis=3, keepdims=True)

print(X_train_rgb.shape)
print(X_train_grayscale.shape)

print(X_valid_rgb.shape)
print(X_valid_grayscale.shape)

print(X_test_rgb.shape)
print(X_test_graycale.shape)

# From here our dataset points to grayscale
X_train = X_train_grayscale
X_test = X_test_graycale
X_valid = X_valid_grayscale

# as its grayscale its only single channel
image_channel = X_train.shape[3]
print(image_channel)

# here we display some images randomly for a quick look.
take_till = 8
figures = {}
random_signs = []
for i in range(take_till):
    index = random.randint(0, n_train - 1)
    labels[i] = class_ids_key_value[y_train[index]][1].decode('ascii')
    figures[i] = X_train[index].squeeze()
    random_signs.append(index)

# lets plot the images using helper function
draw_images(figures, 4, 2, labels)

print('----------------x train-', X_train.__len__())
import cv2

X_train_augmented = []
y_train_augmented = []

X_train_augmented_2 = []
y_train_augmented_2 = []

# here we do the image augmentation to generate extra data.
# we checks in 'new_counts_train' for frequency of each class if its less than 3000 than we do the data augmentation using
# opencv wrapaffine, Perespective Transform & rotation.
augmented_train_class_count = train_class_count
for i in range(n_train):
    if (augmented_train_class_count[y_train[i]] < 3000):
        for j in range(3):
            dx, dy = np.random.randint(-1.7, 1.8, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            dst = cv2.warpAffine(X_train[i], M, (X_train[i].shape[0], X_train[i].shape[1]))
            dst = dst[:, :, None]
            X_train_augmented.append(dst)
            y_train_augmented.append(y_train[i])

            # here we doing the perspective transformation
            upper_threshold = random.randint(27, 32)
            lower_threshold = random.randint(0, 5)
            points_one = np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])
            points_two = np.float32([[0, 0], [upper_threshold, lower_threshold], [lower_threshold, 32],
                                     [32, upper_threshold]])
            M = cv2.getPerspectiveTransform(points_one, points_two)
            dst = cv2.warpPerspective(X_train[i], M, (32, 32))
            X_train_augmented_2.append(dst)
            y_train_augmented_2.append(y_train[i])

            # Here we doing the Rotation(tilting) of image
            rotation_threshold = random.randint(-12, 12)
            M = cv2.getRotationMatrix2D((X_train[i].shape[0] / 2, X_train[i].shape[1] / 2), rotation_threshold, 1)
            dst = cv2.warpAffine(X_train[i], M, (X_train[i].shape[0], X_train[i].shape[1]))
            X_train_augmented_2.append(dst)
            y_train_augmented_2.append(y_train[i])

            augmented_train_class_count[y_train[i]] += 2

# here we concatenate(combine) the augmented images
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
X_train = np.concatenate((X_train, X_train_augmented), axis=0)
y_train = np.concatenate((y_train, y_train_augmented), axis=0)

X_train_augmented_2 = np.array(X_train_augmented)
y_train_augmented_2 = np.array(y_train_augmented)
X_train_augmented_2 = np.reshape(X_train_augmented_2, (np.shape(X_train_augmented_2)[0], 32, 32, 1))
X_train = np.concatenate((X_train, X_train_augmented_2), axis=0)
y_train = np.concatenate((y_train, y_train_augmented_2), axis=0)

X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)

figures1 = {}
labels = {}
figures1[0] = X_train[n_train + 1].squeeze()
labels[0] = y_train[n_train + 1]
figures1[1] = X_train[0].squeeze()
labels[1] = y_train[0]

draw_images(figures1, 1, 2, labels)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print("Augmented Dataset Train Size : {}".format(X_train.shape[0]))
print("Augmented Dataset Test Size : {}".format(X_test.shape[0]))
print("Augmented Dataset Valid Size : {}".format(X_valid.shape[0]))

train_class_ids, train_class_count = np.unique(y_train, return_counts=True)
plt.bar(train_class_ids, train_class_count)
plt.grid()
plt.title("Train Dataset class-wise distribution after augmentation")
plt.show()

test_class_ids, counts_test = np.unique(y_test, return_counts=True)
plt.bar(test_class_ids, counts_test)
plt.grid()
plt.title("Test Dataset class-wise distribution after augmentation")
plt.show()

valid_class_ids, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(valid_class_ids, counts_valid)
plt.grid()
plt.title("Valid Dataset class-wise distribution after augmentation")
plt.show()


def normalize(im):
    return -np.log(1 / ((1 + im) / 257) - 1)


# X_train_normalized = normalize(X_train)
# X_test_normalized = normalize(X_test)

# normalize the value beween -1 to 1
X_train_normalized = (X_train / 128) - 1
X_test_normalized = (X_test / 128) - 1

take_till = 8
figures = {}
count = 0
for i in random_signs:
    labels[count] = class_ids_key_value[y_train[i]][1].decode('ascii')
    figures[count] = X_train_normalized[i].squeeze()
    count += 1

draw_images(figures, 4, 2, labels)

X_train = X_train_normalized
X_test = X_test_normalized


# This Fn Computes a 2-D convolution add the bias and apply the relu on the output
def create_convolution_2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    print(x.shape)
    # applying Activation
    return tf.nn.relu(x)


# LeNet architecture
def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    weight_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, image_channel, 6), mean=mu, stddev=sigma))
    bias_1 = tf.Variable(tf.zeros(6))
    layer_1 = create_convolution_2d(x, weight_1, bias_1, 1)

    #  Pooling. Input = 28x28x6. Output = 14x14x6
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    weight_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    bias_2 = tf.Variable(tf.zeros(16))
    layer_2 = create_convolution_2d(layer_1, weight_2, bias_2, 1)

    # Pooling. Input = 10x10x16. Output = 5x5x16
    layer_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    weight_2_a = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 412), mean=mu, stddev=sigma))
    bias_2_a = tf.Variable(tf.zeros(412))
    layer_2_a = create_convolution_2d(layer_2, weight_2_a, bias_2_a, 1)

    # Flattens the input
    flat = flatten(layer_2_a)

    weight_3 = tf.Variable(tf.truncated_normal(shape=(412, 122), mean=mu, stddev=sigma))
    bias_3 = tf.Variable(tf.zeros(122))
    layer_3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, weight_3), bias_3))
    layer_3 = tf.nn.dropout(layer_3, keep_probability)

    weight_4 = tf.Variable(tf.truncated_normal(shape=(122, 84), mean=mu, stddev=sigma))
    bias_4 = tf.Variable(tf.zeros(84))
    layer_4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_3, weight_4), bias_4))
    layer_4 = tf.nn.dropout(layer_4, keep_probability)

    weight_5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    bias_5 = tf.Variable(tf.zeros(43))
    layer_5 = tf.nn.bias_add(tf.matmul(layer_4, weight_5), bias_5)

    return layer_5


x = tf.placeholder(tf.float32, (None, 32, 32, image_channel))
y = tf.placeholder(tf.int32, (None))

# applying one hot encoding for y
one_hot_encoded_y = tf.one_hot(y, 43)

keep_probability = tf.placeholder(tf.float32)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# Setting the training parameters
EPOCHS = 10
BATCH_SIZE = 150

rate = 0.00099

logits = LeNet(x)

# calculating cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_encoded_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_encoded_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate_accuracy(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_probability: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Lets Training begin...")
    print()
    validation_accuracy_figure = []
    test_accuracy_figure = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_probability: 0.5})

        validation_accuracy = evaluate_accuracy(X_valid, y_valid)
        validation_accuracy_figure.append(validation_accuracy)

        test_accuracy = evaluate_accuracy(X_train, y_train)
        test_accuracy_figure.append(test_accuracy)
        print("EPOCH {} ...".format(i + 1))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './LeNet')
    print("Model saved to file")

plt.plot(validation_accuracy_figure)
plt.title("Test Accuracy plot")
plt.show()

plt.plot(validation_accuracy_figure)
plt.title("Validation Accuracy plot")
plt.show()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate_accuracy(X_train, y_train)
    print("Train Accuracy = {:.3f}".format(train_accuracy))

    valid_accuracy = evaluate_accuracy(X_valid, y_valid)
    print("Valid Accuracy = {:.3f}".format(valid_accuracy))

    test_accuracy = evaluate_accuracy(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

import glob
import cv2

my_images = sorted(glob.glob('./mysigns/*.png'))
my_labels = np.array([1, 22, 35, 15, 37, 18])

figures = {}
labels = {}
my_signs = []
index = 0
for my_image in my_images:
    img = cv2.cvtColor(cv2.imread(my_image), cv2.COLOR_BGR2RGB)
    my_signs.append(img)
    figures[index] = img
    labels[index] = class_ids_key_value[my_labels[index]][1].decode('ascii')
    index += 1

draw_images(figures, 3, 2, labels)

my_signs = np.array(my_signs)
my_signs_gray = np.sum(my_signs / 3, axis=3, keepdims=True)
my_signs_normalized = my_signs_gray / 127.5 - 1

take_till = 6
figures = {}
labels = {}
for i in range(take_till):
    labels[i] = class_ids_key_value[my_labels[i]][1].decode('ascii')
    figures[i] = my_signs_gray[i].squeeze()

draw_images(figures, 3, 2, labels)

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_accuracy = evaluate_accuracy(my_signs_normalized, my_labels)
    print("My Data Set Accuracy = {:.3f}".format(my_accuracy))

### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
my_single_item_array = []
my_single_item_label_array = []

for i in range(6):
    my_single_item_array.append(my_signs_normalized[i])
    my_single_item_label_array.append(my_labels[i])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #         saver = tf.train.import_meta_graph('./lenet.meta')
        saver.restore(sess, "./lenet")
        my_accuracy = evaluate_accuracy(my_single_item_array, my_single_item_label_array)
        print('Image {}'.format(i + 1))
        print("Image Accuracy = {:.3f}".format(my_accuracy))
        print()

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.
k_size = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #     my_saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_signs_normalized, keep_probability: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_signs_normalized, keep_probability: 1.0})
    #     print(my_top_k)

    for i in range(6):
        figures = {}
        labels = {}

        figures[0] = my_signs[i]
        labels[0] = "Original"

        for j in range(k_size):
            #             print('Guess {} : ({:.0f}%)'.format(j+1, 100*my_top_k[0][i][j]))
            labels[j + 1] = 'Guess {} : ({:.0f}%)'.format(j + 1, 100 * my_top_k[0][i][j])
            figures[j + 1] = X_valid[np.argwhere(y_valid == my_top_k[1][i][j])[0]].squeeze()

        #         print()
        draw_images(figures, 1, 6, labels)
