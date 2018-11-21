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
        if doc[1]==1.0:
            print(doc[0])
            while True:
                response = getch()
                if response == '[':
                    output_array[i] = 1
                    break
                elif response == ']':
                    output_array[i] = 2
                    break
                elif response == 'q':
                    break
            if response == 'q':
                break
            print('{}/{} completed.'.format(i+1, count))
            print()
        else:
            output_array[i] = doc[1]
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
        

unlabelled = pd.read_csv('data/not_tri_labelled.csv')
labelled = pd.read_csv('data/tri_labelled.csv')

labels, idx = label_docs(unlabelled[['text', 'label']].values)

new_labelled = unlabelled.iloc[:idx]
new_labelled['label'] = labels[:idx]

labelled = labelled.append(new_labelled)
labelled.to_csv('data/tri_labelled.csv', index=False)

unlabelled.iloc[idx:].to_csv('data/not_tri_labelled.csv', index=False)

print('Thanks for playing!')
