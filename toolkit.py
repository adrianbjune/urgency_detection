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
    output_array = np.zeros(count)

    for i, doc in enumerate(docs):
        print(doc)
        while True:
            response = getch()
            if response == '[':
                output_array[i] = 1
                break
            elif response == ']':
                break
            elif response == 'q':
                break
        if response == 'q':
            break
        print('{}/{} completed.'.format(i+1, count))
        print()
    return output_array, i

# Function to parse one Slack event log line
def parse_one_line(line):
    try:
        time = pd.to_datetime(re.findall(r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}', line)[0])
    except IndexError:
        time = None
    try:
        token = re.findall(r'token=.*, team_id', line)[0][7:-10]
    except IndexError:
        token = None
    try:
        team_id = re.findall(r'team_id=.*, api', line)[0][9:-6]
    except IndexError:
        team_id = None
    try:
        api_app_id = re.findall(r'api_app_id=.*, event=Slack', line)[0][12:-14]
    except IndexError:
        api_app_id = None
    try:
        event_detail_type = re.findall(r'type=.*, user=', line)[0][6:-8]
    except IndexError:
        event_detail_type = None
    try:
        user = re.findall(r'user=.*, text=', line)[0][6:-8]
    except IndexError:
        user = None
    try:
        text = re.findall(r'text=.*, client_msg', line)[0][6:-13]
    except IndexError:
        text = None
    try:
        client_msg_id = re.findall(r'client_msg_id=.*, ts=', line)[0][15:-6]
    except IndexError:
        client_msg_id = None
    try:
        ts = re.findall(r'ts=.*, channel=', line)[0][4:-11]
    except IndexError:
        ts = None
    try:
        channel = re.findall(r'channel=.*, event_ts=', line)[0][9:-12]
    except IndexError:
        channel = None
    try:
        event_ts = re.findall(r'event_ts=.*, channel_type=', line)[0][10:-16]
    except IndexError:
        event_ts = None
    try:
        authed_users = re.findall(r'authed_users=.*}\n', line)[0][13:-2]
    except IndexError:
        authed_users = None
    try:
        event_time = re.findall(r"event_time='.*', authed_users", line)[0][12:-15]
    except IndexError:
        event_time = None
    try:
        event_id = re.findall(r"event_id='.*', event_time", line)[0][10:-13]
    except IndexError:
        event_id = None
    try:
        type_event = re.findall(r"}, type='.*', event_id", line)[0][9:-11]
    except IndexError:
        type_event = None
    try:
        subtype = re.findall(r"subtype='.*'}, type", line)[0][9:-8]
    except IndexError:
        subtype = None
    try:
        username = re.findall(r"username='.*', subtype", line)[0][10:-10]
    except IndexError:
        username = None
    try:
        deleted_ts = re.findall(r"deleted_ts='.*', username", line)[0][12:-11]
    except IndexError:
        deleted_ts = None
    try:
        channel_type = re.findall(r"channel_type='.*', deleted_ts", line)[0][14:-13]
    except IndexError:
        channel_type = None
    
    return [time, token, team_id, api_app_id, event_detail_type, user, text, client_msg_id, 
            ts, channel, event_ts, authed_users, event_time, event_id, type_event, subtype, username, deleted_ts, channel_type]


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
