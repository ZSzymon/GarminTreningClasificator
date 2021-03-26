import TCXParser2 as tcxparser
import csv
import itertools
import re
import tailer as tl
import scipy
import pandas as pd
import numpy as np
import os
from os import path
import io
from scipy import stats
from scipy import ndimage
from pathlib import Path


def remove_outliners(df_col):
    return df_col[(np.abs(stats.zscore(df_col)) < 4)]


def process_hr(tcx):
    df = pd.DataFrame()
    df_hr = pd.DataFrame({'HR': tcx.hr_values()})
    df_hr = remove_outliners(df_hr)
    # MAD = mediana odchylenia bezwzględnego.
    df['HR median'] = [np.median(df_hr)]
    df['HR mad'] = scipy.stats.median_absolute_deviation(df_hr)
    return df


def process_altitude(tcx):
    df = pd.DataFrame()
    altitudes = pd.Series(tcx.altitude_points())
    altitudes = remove_outliners(altitudes)
    df['Alitude min'] = [np.min(altitudes)]
    df['Altitude max'] = [np.max(altitudes)]
    df['Altitude avg'] = [np.average(altitudes)]
    df['Altitude median'] = [np.average(altitudes)]
    df['Altitude mad'] = scipy.stats.median_absolute_deviation(altitudes)
    return df


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


def change_file_extension(file_name, ext):
    if '.' not in ext:
        ext = '.'.join(ext)
    if re.match(".*", file_name):
        dot_index = str(file_name).find('.')
        file_name = file_name[:dot_index] + ext
    return file_name


def to_seconds_time(row):
    if isinstance(row, float):
        return row
    try:
        if len(row) <= 6:
            if (len(row)) == 6:
                stop = 1
            to_return = to_sec(row)
        elif len(row) >= 7:
            colon_index = str(row).find(':')
            hours = float(row[:colon_index])
            rest = to_sec(row[colon_index + 1:])
            to_return = hours * 60 * 60 + rest
        else:
            to_return = 0

    except TypeError as error:
        print(error)

    return to_return


def get_clean_column_as_np_array(df, column_name, convert_to_sec=True, skip=1):
    col = df[column_name].copy()
    col = col.replace('0', np.nan)
    col = col.replace('--', np.nan)
    col = col.dropna()

    if convert_to_sec:
        col = col.apply(to_seconds_time)
    col = pd.to_numeric(col)
    col = np.array(col)
    if col.shape[0] == 2 and skip > 1:
        skip = 1
    col = col[:(-1 * skip)]
    return col


def remove_outliners(df_col):
    to_return = df_col[(np.abs(stats.zscore(df_col)) < 3)]
    return to_return

def to_number_my(series):
    tempo = series['Średnie tempo'].item()
    series['Średnie tempo'] = [to_seconds_time(tempo)]
    series['Średnia długość kroku'] = [float(series['Średnia długość kroku'].item())]
    series['Czas'] = [to_seconds_time(series['Czas'].item())]
    kalorie = series['Kalorie'].item()
    if isinstance(kalorie, str):
        kalorie = kalorie.replace(',','')

    wysokosc_wzrost = series['Wzrost wysokości'].item()
    if isinstance(wysokosc_wzrost, str):
        wysokosc_wzrost = wysokosc_wzrost.replace(',', '')

    wysokosc_spadek = series['Wzrost wysokości'].item()
    if isinstance(wysokosc_spadek, str):
        wysokosc_spadek = wysokosc_spadek.replace(',', '')

    series['Kalorie'] = [float(kalorie)]
    series['Wzrost wysokości'] = [float(wysokosc_wzrost)]
    series['Spadek wysokości'] = [float(wysokosc_spadek)]

    return series
def get_summary_from_csv(file_name):
    empty = pd.DataFrame()
    df = pd.read_csv(file_name)

    columns_to_delete = ['Okrążenia', 'Łączny czas',
                         'Czas ruchu', 'Średnio tempo ruchu', 'Najlepsze tempo'
                         ]
    df = delete_columns(df, columns_to_delete)
    try:
        df = df.drop(df.columns[10], axis=1)
        df = df.drop(df.columns[11], axis=1)
    except Exception:
        stop =1
    columns = list(df.columns)
    distances = get_clean_column_as_np_array(df, columns[1], convert_to_sec=False)

    skip_rows = 2 if float(distances[-1]) < 0.5 else 1
    if file_name =='/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/csvs_jakub/activity_4454636887.csv':
        stop =1
    if columns[2] == 'Średnie tempo':
        paces = get_clean_column_as_np_array(df, columns[2], skip=skip_rows,convert_to_sec=True)
    else:
        return empty
    if columns[3] == 'Średnie tętno':
        hrs = get_clean_column_as_np_array(df, columns[3], convert_to_sec=False)
    else:
        return empty
    if columns[7] == 'Średni rytm biegu':
        cadences = get_clean_column_as_np_array(df, columns[7], convert_to_sec=False)
    else:
        return empty

    new_df = df.tail(1)
    new_df = to_number_my(new_df)
    new_df['HR median'] = [np.median(hrs)]
    new_df['HR std'] = [np.std(hrs)]
    new_df['Pace std'] = [np.std(paces)]
    new_df['Lap pace median'] = [np.median(paces)]
    new_df['Avg Pace(s/km)'] = [np.average(paces)]
    new_df['Best Lap(s/km)'] = [np.min(paces)]
    new_df['Cadence median'] = [np.median(cadences)]
    new_df['Cadence std'] = [np.std(cadences)]
    new_df['Czas'] = new_df['Czas'].apply(to_seconds_time)
    basename = path.basename(file_name)
    activity_id = basename[basename.find("_") + 1: basename.find('.')]
    new_df['File id'] = [activity_id]

    return new_df


def create_summary(csv_file):
    df = get_summary_from_csv(csv_file)
    return df


def delete_columns(df, columns_to_delete):
    not_deleted = []
    try:
        for col in columns_to_delete:
            df = df.drop(columns=[col])
    except KeyError:
        not_deleted.append(col)
    return df


if __name__ == '__main__':

    csvs_dir = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/csvs_jakub'

    files = [path.join(csvs_dir, f) for f in os.listdir(csvs_dir) if path.isfile(path.join(csvs_dir, f))]

    for i, file in enumerate(files):
        new_row = create_summary(file)
        if i == 0:
            df = pd.DataFrame(new_row)
        else:
            df = df.append(new_row)

    df = delete_columns(df, ['Interval', 'Lap', 'Cumulative Time', 'Moving Time',
                             'Avg Moving Pace', 'Avg Ground Contact Time',
                             'Avg Vertical Oscillation', 'Step Type', 'Best Pace', 'Unnamed: 0',
                             'Avg Pace'])

    df['Type'] = 0
    df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_no_label_jakub.csv')
