import csv
import math
import os
import pathlib
import pandas as pd
import numpy as np
from os import path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model


@staticmethod
def get_n_row(file, n):
    with open(file, 'r') as fp:
        csv_file = csv.reader(fp)
        for i, row in enumerate(csv_file, start=0):
            if i == n:
                return row


def toSeconds(row):
    base_time = pd.to_datetime('00:00', format='%M:%S')
    row = pd.to_datetime(row, format='%M:%S') - base_time
    return row.dt.total_seconds()
    pass


class SummaryMaker:

    def __init__(self, summary_path, dir_source_path):
        self.summary_path = summary_path
        self.source_path = dir_source_path

    def get_list_files(self):
        def is_okey(f):
            return not f.endswith('RR.CSV') and path.isfile(path.join(self.source_path, f))

        return [path.join(self.source_path, f) for f in os.listdir(self.source_path)
                if is_okey(f)]

    def create(self):
        files = self.get_list_files()
        df = pd.DataFrame()
        for file in files:
            new_df = FeaturesMaker(file, "")
            new_df.create_basic_summary()
            new_df = new_df.get_df()
            df = df.append(new_df)
            pass
        df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summaryv2.csv')

        stop = 1
    def create_bigger_summary(self, oldsummary=''):
        oldsummary='/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summaryv2.csv'
        df = pd.read_csv(oldsummary)
        new_summary = pd.DataFrame()
        for i,file in enumerate(df['file_path']):
            if file == '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/csvs/Szymon+_ywko+_2019-09-14_10-01-43.CSV':
                stop = 1
            new_df = FeaturesMaker(file, "").create_bigger_summary()
            new_summary = new_summary.append(new_df)
            print(f"done: {i}")
            pass
        to_save = df.join(new_summary)
        to_save.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/polar_data/summaryv3.csv')

class FeaturesMaker:
    def __init__(self, sourcefile, destination_file):
        self.source_file = sourcefile
        self.destination_file = destination_file
        self.required_cols = ['Average pace (min/km)', 'Average heart rate (bpm)','Average cadence (rpm)']

        self.columns_to_delete = ['Sample rate', 'Stride length (m)', 'Power (W)', 'Speed (km/h)','Running index',
                             'Ascent (m)', 'Descent (m)','HR max', 'HR sit', 'VO2max','Notes',]
        #all_cols = ['Name', 'Sport', 'Date', 'Start time', 'Duration', 'Total distance (km)',
        #            'Average heart rate (bpm)', 'Average speed (km/h)', 'Max speed (km/h)',
        #            'Average pace (min/km)', 'Max pace (min/km)', 'Calories', 'Fat percentage of calories(%)',
        #            'Average cadence (rpm)', 'Average stride length (cm)','Running index', 'Training load',
        #            'Ascent (m)', 'Descent (m)', 'Average power (W)', 'Max power (W)',
        #            ' Notes', 'Height (cm)', 'Weight (kg)', 'HR max', 'HR sit', 'VO2max']


    def create_basic_summary(self):
        self.df = self.obtain_existring_features()
        self.all_cols = self.get_cols_list()
        if all(col in self.all_cols for col in self.required_cols):
            for col in self.columns_to_delete:
                if col in self.all_cols:
                    self.df.drop(columns=[col], inplace=True)
        else:
            self.df = pd.DataFrame()
        return self
    def create_bigger_summary(self):
        empty_frame = pd.DataFrame()
        self.df = pd.read_csv(self.source_file, skiprows=2)
        self.df['Pace (min/km)'].replace('', np.nan, inplace=True)
        self.df.dropna(subset=['Pace (min/km)'], inplace=True)
        self.df['Pace (min/km)'] = self.df['Pace (min/km)'].apply(self.to_sec2)
        self.df.rename(columns={'Pace (min/km)': 'Pace (s/km)'}, inplace=True)


        bpm_features, succes = self.create_feature(self.df, "HR (bpm)")
        if not succes:
            return empty_frame

        if bpm_features.shape[1] < 10:
            return empty_frame

        pace_features, succes = self.create_feature(self.df,'Pace (s/km)')
        if not succes:
            return empty_frame

        altitude_features, succes = self.create_feature(self.df, 'Altitude (m)')
        if not succes:
            return empty_frame

        temperature_features, succes = self.create_temperature_features(self.df)
        if not succes:
            return empty_frame

        toReturn = bpm_features.join(pace_features).join(altitude_features).join(temperature_features)
        return toReturn


    def get_df(self):
        return self.df

    @staticmethod
    def to_sec2(string:str):
        comma_index = str(string).find(':')
        if isinstance(string, float):
            raise TypeError('Expected string instance ')
        if comma_index == -1:
            raise ValueError('Wrong string format.')
        minutes = float(string[0:comma_index])
        seconds = float(string[comma_index + 1:])
        return minutes * 60 + seconds

    @staticmethod
    def repair_date(date_cell):
        datetime_object = datetime.strptime(date_cell, '%m/%d/%Y')
        datetime_object = datetime_object.strftime('%d/%m/%Y')
        return datetime_object

    def get_cols_list(self):
        columns = self.df.columns
        columns = list(columns)

        return columns

    def obtain_existring_features(self):
        df_summary = pd.read_csv(self.source_file, nrows=1)
        df_summary.dropna(axis=1, inplace=True)
        df_summary['Date'] = df_summary['Date'].apply(self.repair_date)
        df_summary['file_path'] = [self.source_file]
        try:
            df_summary['Average pace (min/km)'] = df_summary['Average pace (min/km)'].apply(self.to_sec2)
        except KeyError:
            df_summary = pd.DataFrame()
        finally:
            return df_summary


    def remove_outliners(self, df_col):
        to_return = df_col[(np.abs(stats.zscore(df_col)) < 3)]
        return to_return

    def correlation_plot(self, x, y,
                         save_path="",
                         title="Title",
                         xlabel="X", ylabel='X'):
        plt.scatter(x, y, marker=".")
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.arange(x.min(), x.max())
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y,
                 label='$%.5fx + %.2f$, $R^2=%.2f$' % (slope, intercept, r_value ** 2), color='red')
        plt.legend(loc='best')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.tight_layout()
        # plt.savefig(save_path)
        plt.show()
        plt.clf()  # clear figure
        plt.close()

    def calc_slope_info(self, dataframe, col_name):
        y = dataframe
        X = np.arange(0, dataframe.shape[0], 1)
        # self.correlation_plot(X, y)
        # y= ax + b
        # slope = a
        # intercept = b
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        df = pd.DataFrame({f'{col_name} slope': [slope],
                           f'{col_name} intercept': [intercept],
                           f"{col_name} r2": [r_value ** 2],
                           f'{col_name} p_value': [p_value],
                           f'{col_name} std_err': [std_err]})

        return df


    def create_pace_features(self, df):
        new_df = pd.DataFrame()
        df_pace = df["Pace (s/km)"]
        df_pace = self.remove_outliners(df_pace)
        if df_pace.empty:
            return pd.DataFrame()
        new_df["Pace median"] = [df_pace.median()]
        new_df["Pace variance"] = [df_pace.var()]
        new_df["Pace sd"] = [math.sqrt(new_df["Pace variance"])]
        new_df['Pace min'] = [min(df_pace)]
        new_df['Pace max'] = [max(df_pace)]
        new_df = new_df.join(self.calc_slope_info(df_pace, 'Pace'))
        return new_df
        pass
    def create_feature(self,df, col_name):
        new_df = pd.DataFrame()
        df[col_name].replace('', np.nan, inplace=True)
        df.dropna(subset=[col_name], inplace=True)
        if df.shape[0] == 0:
            return pd.DataFrame(), False

        df_bpm = df[col_name]
        if df_bpm.shape[0] == 1:
            return pd.DataFrame(), False

        df_bpm = df_bpm[(np.abs(stats.zscore(df_bpm)) < 5)]
        if df_bpm.shape[0] == 0:
            return pd.DataFrame(), False
        new_df[f'{col_name} median'] = [df_bpm.median()]
        new_df[f"{col_name} variance"] = [df_bpm.var()]
        new_df[f"{col_name} sd"] = [math.sqrt(df_bpm.var())]
        try:
            new_df[f"{col_name} min"] = [min(df_bpm)]
            new_df[f"{col_name} max"] = [max(df_bpm)]
        except:
            stop = 1
            return pd.DataFrame(), False
            pass

        slope_info = self.calc_slope_info(df_bpm, col_name)
        new_df = new_df.join(slope_info)
        return new_df, True

    def create_temperature_features(self, df):

        new_df = pd.DataFrame()
        df_temp = df["Temperatures (C)"]
        df_temp = self.remove_outliners(df_temp)
        if df_temp.shape[0] == 0:
            return pd.DataFrame(), False
        new_df["Temperatures (C) median"] = [df_temp.median()]

        return new_df, True





if __name__ == '__main__':
    source = '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/csvs/summaryv2.CSV'
    dir_source = '/home/zywko/PycharmProjects/BA_Code/resources/polar_data/csvs'
    summary_maker = SummaryMaker(source, dir_source)
    summary_maker.create_bigger_summary()
