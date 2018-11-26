'''
Script to label data as one of 5 categories using corresponding key inputs:
0 : Not a call to action
1 : Rhetorical question
2 : Quick Question
3 : Task (an explicit call to action)
'''
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
    output_array = np.zeros(count)

    for i, doc in enumerate(docs):
        print(doc)
        while True:
            response = getch()
            if response == '0':
                output_array[i] = 0
                break
            elif response == '1':
                output_array[i] = 1
                break
            elif response == '2':
                output_array[i] = 2
                break
            elif response == '3':
                output_array[i] = 3
                break
            elif response == 'q':
                break
        if response == 'q':
            break
        print('{}/{} completed.'.format(i+1, count))
        print()
    return output_array, i


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
        

unlabelled = pd.read_csv('data/first_unlabelled.csv')
labelled = pd.read_csv('data/5_labelled.csv')

labels, idx = label_docs(unlabelled['text'].values)

new_labelled = unlabelled.iloc[:idx]
new_labelled['label'] = labels[:idx]

labelled = labelled.append(new_labelled)
labelled.to_csv('data/5_labelled.csv', index=False)

unlabelled.iloc[idx:].to_csv('data/first_unlabelled.csv', index=False)

print('Thanks for playing!')
