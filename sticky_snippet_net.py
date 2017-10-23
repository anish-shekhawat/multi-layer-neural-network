#!/usr/bin/env python
"""Trains and Tests a Neural Net on Gene Snippets"""

import sys


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 4:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sticky_snippet_net.py mode",
        print >> sys.stderr, "model_file data_folder\""
        exit(1)
