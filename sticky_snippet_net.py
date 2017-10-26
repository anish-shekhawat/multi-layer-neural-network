#!/usr/bin/env python
"""Trains and Tests a Neural Net on Gene Snippets for Sticky Snippets"""

import sys
import os


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
    mini_batch_size = 20
    learning_rate = 0.01

    def __init__(self, length, data_folder):
        """Initialize a new NEURALNET Object

        :param data_folder: Input data folder name
        :returns: Returns nothing
        """

        self.len = length
        self.data = []

        # TODO:
        conv_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # Get all txt files in data_folder
        for filename in os.listdir(data_folder):
            abs_path = os.path.abspath(data_folder) + "/" + filename
            if os.path.isfile(abs_path) and filename.endswith('.txt'):
                with open(abs_path) as in_file:
                    for line in in_file:
                        line = line.strip()
                        # Check if length of
                        if len(line) == 40:
                            line = [conv_dict[ch] for ch in line]
                            print self.determine_label(line)
                            self.data.append(line)

    def train(self):
        """Trains the Neural Net on data folder
        """
        pass

    def test(self):
        """Tests the Neural Net on data folder
        """

        # Check if model file exists
        if os.path.isfile(sys.argv[2]) is not True:
            print >> sys.stderr, "Enter a valid model_file name"
            exit(1)

    def cross_validation(self, k):
        """Performs a k-fold cross validation training and testing
        """
        pass

    def __validate_input(self):
        """Checks if input data gene snippets are of valid length
        """
        pass

    def determine_label(self, snippet):
        """Determines the label of the input gene snippet

        :param snippet: Gene snippet
        :returns: Class of gene snippet
        """
        length = len(snippet)

        one_hot_vector_dict = {(0, 0): [1, 0, 0, 0, 0, 0],
                               (1, 2): [0, 1, 0, 0, 0, 0],
                               (3, 4): [0, 0, 1, 0, 0, 0],
                               (5, 6): [0, 0, 0, 1, 0, 0],
                               (7, 8): [0, 0, 0, 0, 1, 0],
                               (length / 2, length / 2): [0, 0, 0, 0, 0, 1]}

        i = 0
        while i < (len(snippet)/2):
            if snippet[i] != self.__sticks[snippet[length-1-i]]:
                break
            i += 1

        for key in one_hot_vector_dict:
            if i in key:
                label = one_hot_vector_dict[key]

        return label


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

    NET = NEURALNET(40, sys.argv[3])
