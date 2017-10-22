#!/usr/bin/env python
""" Implements S-K Algorithm for training SVMs on Zener Cards. """

import sys


class STICKYGENERATOR(object):
    """A gene snippet generator

    Attributes:
        bases: Letters which make up a gene
        num_snippets: No. of gene snippets to generate
        mutation_rate: Odds that a character gets mutated to a random character
        from_ends: Dist. from either start or end to apply the mutation rate to
    """

    def __init__(self, num, rate, ends):
        """Initialize a new STICKYGENERATOR Object

        :param num: No. of gene snippets to generate
        :param rate: Odds that a character gets mutated to a random character
        :param ends: Dist. from either start or end to apply mutation rate to
        :returns: Returns nothing
        """
        self.num_snippets = num
        self.mutation_rate = rate
        self.from_ends = ends
        self.bases = ['A', 'B', 'C', 'D']


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 5:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sticky_snippet_generator.py",
        print >> sys.stderr, "num_snippets mutation_rate from_ends",
        print >> sys.stderr, "output_file\""
        exit(1)
