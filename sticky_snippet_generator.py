#!/usr/bin/env python
"""Generates Gene Snippets of given length and mutation rate at given index"""

import sys
import random


class STICKYGENERATOR(object):
    """A gene snippet generator

    Attributes:
        :__bases: Letters which make up a gene
        :__sticks: Dictionary that gives sticking rules
        :mutation_rate: Odds that a character gets mutated to random character
        :from_ends: Dist. from either start or end to apply mutation rate to
    """
    __bases = ['A', 'B', 'C', 'D']
    __sticks = {'A': 'C', 'B': 'D', 'C': 'A', 'D': 'B'}

    def __init__(self, length, rate, ends):
        """Initialize a new STICKYGENERATOR Object

        :param len: Length of gene snippets to be generated
        :param rate: Odds that a character gets mutated to a random character
        :param ends: Dist. from either start or end to apply mutation rate to
        :returns: Returns nothing
        """
        self.len = length
        self.mutation_rate = rate
        self.from_ends = ends

    def generate(self, output_file, num=1):
        """Writes generated gene snippets to output_file

        :param output_file: Name of output file to which snippets are written
        :param num: No. of gene snippets to generate
        :returns: Returns nothing
        """
        output = ""
        while num > 0:
            snippet = self.get_gene_snippet()
            output += snippet + "\n"
            num -= 1

        print output
        out_file = open(output_file, 'w')
        out_file.write(output)
        out_file.close()

    def get_gene_snippet(self):
        """Returns a string of gene snippet

        :returns: String of gene snippet
        """
        # Get the stick palindrome segment of gene
        first_palin = []
        second_palin = []

        if self.from_ends > 1:
            first_palin, second_palin = self.__generate_stick_palindrome()
        # Get the bases with the given mutation rate odds
        mutated_char1, mutated_char2 = self.__apply_mutation_rate()
        # Randomly generate the remaining characters
        random_len = (self.len - 2) - 2 * (self.from_ends - 1)
        snippet = [random.choice(self.__bases) for _ in range(random_len)]
        snippet = first_palin + mutated_char1 + snippet + \
            mutated_char2 + second_palin

        return "".join(snippet)

    def __generate_stick_palindrome(self):
        """
        Returns two strings (one in reverse) which are
        stick palindrome of each other of length from_ends-1

        :returns: Two stick palindrome strings
        """

        first = [random.choice(self.__bases) for _ in range(self.from_ends-1)]
        second = [self.__sticks[item] for item in first]
        second.reverse()

        return first, second

    def __apply_mutation_rate(self):
        """Returns a set of two bases with mutation probability

        :returns: Two bases which are mutated with mutation_rate odds
        """
        # TODO: Mutation rate
        return [random.choice(self.__bases)], [random.choice(self.__bases)]


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 5:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sticky_snippet_generator.py",
        print >> sys.stderr, "num_snippets mutation_rate from_ends",
        print >> sys.stderr, "output_file\""
        exit(1)

    GENERATOR = STICKYGENERATOR(40, float(sys.argv[2]), int(sys.argv[3]))
    GENERATOR.generate(sys.argv[4], int(sys.argv[1]))
