import re
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import stats

def delete_columns(df, columns_to_delete):
    not_deleted = []
    try:
        for col in columns_to_delete:
            df = df.drop(columns=[col])
    except KeyError:
        not_deleted.append(col)
    return df


def remove_outliners(df_col):
    return df_col[(np.abs(stats.zscore(df_col)) < 4)]

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
        if isinstance(row, pd.Series):
            return row.apply(to_seconds_time)
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