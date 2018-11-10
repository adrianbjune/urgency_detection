# Toolkit for custom and possibly commonly used methods

import numpy as np
import pandas as pd


def label_docs(docs):
    '''
    Takes in a list of strings and prompts user on whether each
    string matches a binary label.

    Input:
    Numpy array - List of documents to be labelled

    Output: 
    Numpy array - list of binary results of whether each document
      matches the desired label.
    '''
    getch = _Getch()
    count = len(docs)
    output_array = np.zeros(lcount)

    for i, doc in enumerate(docs):
        print(doc)
        while True:
            response = getch()
            if response == '[':
                output_array[i] = 1
                break
            elif response == ']':
                break
            elif response == ['q']:
                break
        if response == ['q']:
            break
        print('{}/{} completed.'.format(i+1, count))
        print()
    return output_array, i+1


class _Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
