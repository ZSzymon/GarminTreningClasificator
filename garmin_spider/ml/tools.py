import re
from joblib import dump, load

def to_sec(pace):
    colon_index = str(pace).find(':')
    if colon_index == -1:
        return 0.0
    try:
        minutes = float(pace[:colon_index])
    except ValueError as error:
        print(error)

    seconds = float(pace[colon_index + 1:])
    return minutes * 60 + seconds


def to_seconds_time(row):
    try:
        if len(row) <= 5:
            to_return = to_sec(row)
        elif len(row) > 5:
            colon_index = str(row).find(':')
            hours = float(row[:colon_index])
            rest = to_sec(row[colon_index + 1:])
            to_return = hours * 60 * 60 + rest
        else:
            to_return = 0

    except TypeError as error:
        print(error)

    return to_return



def save_model(clf,name):
    dump(clf, name)
    pass

def load_model(abs_path):
    clf = load(abs_path)
    return clf