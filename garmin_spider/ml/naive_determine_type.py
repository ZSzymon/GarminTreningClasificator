import pandas as pd
import numpy as np
import scipy
from os import path

def determine_type(df : pd.DataFrame):
    for i, row in df.iterrows():
        if 120 < row['Średnie tętno'] < 160 and row['Avg Pace(s/km)'] > (3*60 + 50) and row['HR std'] < 20\
                and row['Pace std'] < 25:
            df.at[i,'Type'] = 0
        elif 150 < row['Średnie tętno'] < 180 and row['Pace std'] < 40 and ((3*60) + 20) < row['Avg Pace(s/km)'] < ((3*60) + 50) and 4 < row['Dystans']:
            df.at[i,'Type'] = 1
        elif row['Średnie tętno'] > 160 and row['Cadence std'] < 15 and row['Avg Pace(s/km)'] < ((3*60) + 30):
            df.at[i, 'Type'] = 2
        else:
            df.at[i, 'Type'] = -1
            pass


    return df


if __name__ == '__main__':
    summary_path = '/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_no_label_jakub.csv'
    df = pd.read_csv(summary_path)
    determine_type(df)
    stop = 1
    training_types = {
        'BC 1' : 0,
        'BC 2' : 1,
        'BC 3' : 2,
        'fartelek': 3,
        'BC 1 + RT': 5,
        'Rozgrzewka': 6,#
        'RT' : 7,
        'Inny' : -1,
    }
    columns = list(df.columns)
    df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_jakub_prelabeled.csv', header = columns)

    stop =1