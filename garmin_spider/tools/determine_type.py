
import pandas as pd
import numpy as np
import scipy
from os import path

def determine_type(df : pd.DataFrame):
    for i, row in df.iterrows():
        if 120< row['Avg HR'] < 160 and row['Avg Pace(s/km)'] > (3*60 + 50) and row['HR std'] < 20\
                and row['Pace std'] < 25:
            df.at[i,'Type'] = 0
        elif 150 < row['Avg HR'] < 180 and row['Pace std'] < 40 and row['Avg Pace(s/km)'] < (3*60 + 50):
            df.at[i,'Type'] = 1
        else:
            df.at[i, 'Type'] = -1
            pass


    return df


if __name__ == '__main__':
    summary_path = '/resources/summary_garmin3.csv'
    df = pd.read_csv(summary_path)
    determine_type(df)
    stop = 1
    training_types = {
        'BC 1' : 0,
        'BC 2' : 1,
        'BC 3' : 2,
        'fartelek': 3,
        'Trening Specjalistyczny': 4,
        'BC 1 + RT': 5,
        'Inny' : -1,
    }
    columns = list(df.columns)
    df.to_csv('/home/zywko/PycharmProjects/BA_Code/resources/summary_garmin_labeled.csv',header = columns)
    stop =1