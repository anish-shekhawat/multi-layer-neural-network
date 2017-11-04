#!/usr/bin/env python
"""Trains and Tests a Neural Net on Gene Snippets for Sticky Snippets"""

import sys
import random
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NEURALNET(object):
    """A Neural Net (NN) that classifies a gene snippet's stickiness

    A Neural Net that classifies gene snippets as NONSTICK, 12-STICKY,
    34-STICKY, 56-STICKY, 78-STICKY or STICK_PALINDROME

    Attributes:
        :__sticks: Dictionary that gives sticking rules
        :len: Length of the gene snippets
        :mini_batch_size: Batch size of data fed into NN
        :learning_rate: Learning rate of neural net
        :data: Input data
        ::
        ::
    """

    __sticks = {0: 2, 1: 3, 2: 0, 3: 1}
    mini_batch_size = 100
    learning_rate = 0.5
    epoch = 15

    def __init__(self, length):
        """Initialize a new NEURALNET Object

        :param data_folder: Input data folder name
        :returns: Returns nothing
        """

        self.len = length

    def __initialize_variables(self, mode):
        """Initializes Tensorflow variable
        """
        # Input
        X = tf.placeholder(tf.float32, [None, 40], name='X')
        # Variable initializer
        xavier = tf.contrib.layers.xavier_initializer()

        # Layer 1
        w1 = tf.get_variable("w1", shape=(40, 6), dtype=tf.float32,
                             initializer=xavier)
        b1 = tf.get_variable("b1", shape=(6), dtype=tf.float32,
                             initializer=xavier)

        w2 = tf.get_variable("w2", shape=(6, 100), dtype=tf.float32,
                             initializer=xavier)
        b2 = tf.get_variable("b2", shape=(100), dtype=tf.float32,
                             initializer=xavier)

        w3 = tf.get_variable("w3", shape=(100, 200), dtype=tf.float32,
                             initializer=xavier)
        b3 = tf.get_variable("b3", shape=(200), dtype=tf.float32,
                             initializer=xavier)

        w4 = tf.get_variable("w4", shape=(200, 6), dtype=tf.float32,
                             initializer=xavier)
        b4 = tf.get_variable("b4", shape=(6), dtype=tf.float32,
                             initializer=xavier)

        # Class labels
        y_ = tf.placeholder(tf.float32, [None, 6], name='y_')

        layer1 = self.perceptron(w1, X, b1)
        layer2 = self.perceptron(w2, layer1, b2)
        layer3 = self.perceptron(w3, layer2, b3)
        logits = self.perceptron(w4, layer3, b4)

        prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(
            tf.cast(prediction, tf.float32), name='accuracy')

        return accuracy, logits, X, y_

    def train(self, model_file='trained_model', data=None):
        """Trains the Neural Net on data folder
        """

        if data is None:
            # Get training data
            data = self.import_data(sys.argv[3])

        # Randomize the input data
        self.__randomize_inputs(data)

        accuracy, logits, X, y_ = self.__initialize_variables("train")

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

        train_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(cross_entropy)
        # Global Initializer
        init = tf.global_variables_initializer()

        # Initialize the TensorFlow session
        sess = tf.InteractiveSession()
        sess.run(init)

        for i in range(self.epoch):
            print "Epoch" + str(i) + ":",
            counter = 0
            for k in range(0, len(data), self.mini_batch_size):
                batch_xs = data[k: k + self.mini_batch_size]
                batch_ys = self.determine_labels(batch_xs)
                _, loss, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={
                                   X: batch_xs, y_: batch_ys})
                print acc
                counter += len(batch_xs)
                if (counter) % 1000 == 0:
                    print str(counter) + " inputs trained."

        print "Training complete!!"

        # Save model to file
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_file)

        print("Model saved in file: %s" % save_path)

    def __randomize_inputs(self, data):
        """Randomizes the input data
        """

        for i in range(0, len(data)-1):
            j = random.randint(i + 1, len(data)-1)
            temp = data[i]
            data[i] = data[j]
            data[j] = temp

    @staticmethod
    def perceptron(weights, inputs, biases):
        """Perceptron Unit

        :param weights: Weight vector
        :param input: Input vector
        :param biases: Bias vector
        :returns: Result of ReLU activation
        """
        node = tf.add(tf.matmul(inputs, weights), biases)
        return tf.nn.relu(node)

    def test(self, model_file='trained_model', data=None):
        """Tests the Neural Net on data folder
        """

        # Check if model file exists
        if os.path.isfile(sys.argv[2] + '.meta') is not True:
            print >> sys.stderr, "Enter a valid model_file name"
            exit(1)

        if data is None:
            # Get testing data
            data = self.import_data(sys.argv[3])

        labels = self.determine_labels(data)

        accuracy, logits, X, y_ = self.__initialize_variables("test")

        sess = tf.Session()
        # Get Tensorflow model
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        print "Model restored!"

        test_accuracy = accuracy.eval(session=sess, feed_dict={
                                     X: data, y_: labels})

        print test_accuracy

    def cross_validation(self, k, model_file):
        """Performs a k-fold cross validation training and testing
        """

        data = self.import_data(sys.argv[3])

        subset_size = len(data) / k

        for i in range(k):
            train_data = data[0:]
            train(model_file, )

    def import_data(self, folder):
        """Import data from folder

        :param folder: Path of data folder
        :returns: List of data inputs
        """
        conv_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        data = []
        for filename in os.listdir(folder):
            abs_path = os.path.abspath(folder) + "/" + filename
            # Get all txt files in folder
            if os.path.isfile(abs_path) and filename.endswith('.txt'):
                with open(abs_path) as in_file:
                    for line in in_file:
                        line = line.strip()
                        # Check if length is self.len
                        if len(line) == self.len:
                            line = [conv_dict[ch] for ch in line]
                            data.append(line)

        return data

    def determine_labels(self, snippets):
        """Determines the label of the input gene snippet

        :param snippet: Gene snippet
        :returns: Class of gene snippet
        """

        length = self.len
        one_hot_vector_dict = {(0, 0): [1, 0, 0, 0, 0, 0],
                               (1, 2): [0, 1, 0, 0, 0, 0],
                               (3, 4): [0, 0, 1, 0, 0, 0],
                               (5, 6): [0, 0, 0, 1, 0, 0],
                               (7, 8): [0, 0, 0, 0, 1, 0],
                               (length / 2, length / 2): [0, 0, 0, 0, 0, 1]}

        labels = []
        for snippet in snippets:
            i = 0
            while i < (len(snippet)/2):
                if snippet[i] != self.__sticks[snippet[length-1-i]]:
                    break
                i += 1

            for key in one_hot_vector_dict:
                if i in key:
                    label = one_hot_vector_dict[key]
                elif i > 8 and i < length/2:
                    label = one_hot_vector_dict[(7, 8)]

            labels.append(label)

        return labels


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 4:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sticky_snippet_net.py mode",
        print >> sys.stderr, "model_file data_folder\""
        exit(1)

    # Check if mode provided is valid
    if sys.argv[1] not in ['train', '5fold', 'test']:
        print >> sys.stderr, "Mode must be one of the following: ",
        print >> sys.stderr, "train, 5fold or test"
        exit(1)

    # Check if data folder exists
    if os.path.isdir(sys.argv[3]) is not True:
        print >> sys.stderr, "No folder name " + sys.argv[3] + " found."
        print >> sys.stderr, "Please enter a valid data folder name"
        exit(1)

    NET = NEURALNET(40)

    if sys.argv[1] == 'train':
        NET.train(sys.argv[2])
    elif sys.argv[1] == 'test':
        NET.test(sys.argv[2])
