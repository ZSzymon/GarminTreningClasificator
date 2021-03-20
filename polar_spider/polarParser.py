import csv
import itertools
import re
import tailer as tl
import scipy
import pandas as pd
import numpy as np
import os
import io
from os import path
from scipy import stats
from scipy import ndimage
from pathlib import Path
from datetime import datetime
from tools import *


def to_seconds(col, format):
    if format == '%H:%M:%S':
        base_time = pd.to_datetime('00:00:00', format=format)
    elif format == '%M:%S':
        base_time = pd.to_datetime('00:00', format=format)
    else:
        raise NotImplementedError

    col = pd.to_datetime(col, format=format) - base_time
    return col.dt.total_seconds()
    pass


class GarminParser:

    def __init__(self, sourcedir, destfile):
        self.sourcedir = sourcedir
        self.destfile = destfile
        self.required_cols = ['Average pace (min/km)', 'Average heart rate (bpm)', 'Average cadence (rpm)']

        self.columns_to_delete = ['Sample rate', 'Stride length (m)', 'Power (W)', 'Speed (km/h)', 'Running index',
                                  'HR max', 'HR sit', 'VO2max', 'Notes', ]
        pass

    def get_columns(self, dt, columns_name, f=None, drop_nan=True, *args):

        col = f(dt[columns_name].copy(), *args) if f != None else dt[columns_name]
        if drop_nan:
            col = col.replace('', np.nan)
            col = col.dropna()

        success = False if col.isnull().values.any() or col.shape[0] == 0 else True
        if not success:
            stop = 1

        return col, success

    def process_distances_and_cadence(self, dt_bigger):
        empty = pd.DataFrame()
        summary = pd.DataFrame()

        paces, success = self.get_columns(dt_bigger, 'Pace (min/km)', to_seconds_time, False, )
        if not success: return empty
        distances, success = self.get_columns(dt_bigger, 'Distances (m)')
        if not success: return empty

        lap = 1000
        delta = 1000
        last_index = distances.shape[0]
        i = 0
        lap_paces = []
        prev = i
        while i != last_index:
            prev = i
            i = distances.searchsorted(lap)
            lap += delta

            ##jeżeli ostatnie okrążenie trwało zbyt krótko istnieje duża szansa
            ##że ostatnie okrążenie będzie nienaturalnie szybkie
            if i - prev < 30:
                break
            lap_paces.append(np.average(paces[prev:i]))

        lap_paces = np.array(lap_paces)
        summary['Pace std'] = [np.std(lap_paces)]
        summary['Lap pace median'] = [np.median(lap_paces)]
        summary['Avg Pace(s/km)'] = [np.average(lap_paces)]
        summary['Best Lap'] = [np.max(lap_paces)]

        return summary

    def get_primary_sumarry_order(self, dt: pd.DataFrame, dt_bigger: pd.DataFrame = None):
        empty = pd.DataFrame()
        summary = pd.DataFrame()
        # def get_columns(self, dt, columns_name, f=None, accept_nan=False, drop_nan=True, *args):
        time, success = self.get_columns(dt, 'Duration', to_seconds, False, '%H:%M:%S')
        if not success:
            return empty
        summary['Time'] = time

        distance, success = self.get_columns(dt, 'Total distance (km)')
        if not success:
            return empty
        summary['Distance'] = distance

        avg_hr, success = self.get_columns(dt, 'Average heart rate (bpm)')
        if not success:
            return empty
        summary['Avg HR'] = avg_hr
        ##max HR
        hrs, success = self.get_columns(dt_bigger, 'HR (bpm)')
        if not success: return empty
        summary['Max HR'] = np.max(np.array(hrs))

        elevs, success = self.get_columns(dt, ['Ascent (m)', 'Descent (m)'])
        summary['Elev Gain'] = elevs['Ascent (m)'] if success else 0
        summary['Elev Loss'] = elevs['Descent (m)'] if success else 0

        cadence, success = self.get_columns(dt, ['Average cadence (rpm)', ])
        if not success: return empty
        summary['Avg Run Cadence'] = cadence

        calories, success = self.get_columns(dt, 'Calories')
        if not success: return empty
        summary['Calories'] = calories

        temp, success = self.get_columns(dt_bigger, 'Temperatures (C)')
        if not success: return empty
        summary['Avg Temperature'] = np.median(np.array(temp))

        cadence, success = self.get_columns(dt_bigger, 'Cadence')
        if not success: return empty
        summary['Max Run Cadence'] = np.max(np.array(cadence))

        hrs, success = self.get_columns(dt_bigger, 'HR (bpm)')
        if not success: return empty
        summary['HR median'] = np.median(np.array(hrs))
        summary['HR std'] = np.std(np.array(hrs))

        paces_summary = self.process_distances_and_cadence(dt_bigger)
        summary = summary.join(paces_summary)


        return summary

    def process_file(self, file):
        df = self.get_primary_sumarry_order(pd.read_csv(file, nrows=1), pd.read_csv(file, skiprows=2))

        if not df.empty:
            file_name = os.path.basename(file)
            splited = str.split(os.path.basename(file_name),"_")
            start_date = (splited[2])
            start_time = splited[3].split('.')[0]
            df['File name'] = [file_name]
            df['Date'] = [datetime.strptime(start_date, '%Y-%m-%d').date()]
            df['Start time'] = [start_time]
            df['Type'] = -1

        return df

    def process_dir(self):

        dir = self.sourcedir
        df_initialized = False
        files = [path.join(dir, f) for f in os.listdir(dir) if path.isfile(path.join(dir, f))]
        for file in files:
            if file.lower().endswith('rr.csv'):
                continue
            new_row = self.process_file(file)
            if not new_row.empty:
                if not df_initialized:
                    df = pd.DataFrame(new_row)
                    df_initialized = True
                else:
                    df = df.append(new_row)
            else:
                stop =1
        df.to_csv(self.destfile)


if __name__ == '__main__':
    file = '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/activity_example.CSV'
    sourcedir = '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/csvs'
    dest_file = '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summary_polar_to_label.csv'

    parser = GarminParser(sourcedir, dest_file)
    parser.process_dir()
    pass
