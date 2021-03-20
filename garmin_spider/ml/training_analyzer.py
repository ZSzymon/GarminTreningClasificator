import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy import stats
from tools import to_seconds_time


class Analyzer:

    def __init__(self,training_path):

        self.df = pd.read_csv(training_path)
        self.df['Pace (min/km)'] = self.df['Pace (min/km)'].apply(to_seconds_time)
        self.df['Time'] = self.df['Time'].apply(to_seconds_time)
        self.df.rename(columns={'Pace (min/km)': 'Pace (s/km)'}, inplace=True)
        self.df = self.df[(np.abs(stats.zscore(self.df)) < 2).all(axis=1)]

        not_allowed_pairs = [('Pace (s/km)', 'Speed (km/h)'), ('Time', 'Temperatures (C)'),
                             ('Temperatures (C)', 'Distances (m)')]
        not_allowed_cols = ['Temperatures (C)']
        cols_to_delete = ['Speed (km/h)']

    def create(self):
        df = self.df
        not_allowed_pairs = self.not_allowed_pairs
        ##One to one
        columns = list(df.columns)
        for target_col in columns:

            y = np.array(df[target_col]).reshape(-1, 1)
            y = df[target_col]
            for col in columns:
                if (col, target_col) in not_allowed_pairs or (target_col, col) in not_allowed_pairs:
                    continue

                X = np.array(df[col]).reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

                regr = linear_model.LinearRegression()
                regr.fit(X_train, y_train)

                explaindex_variance = regr.score(X_test, y_test)
                if 0.990 > explaindex_variance > 0.100:
                    print(f'Badany: {target_col} : {col}')
                    # The coefficients
                    print('Coefficients: \n', regr.coef_)
                    # The mean square error
                    print("Residual sum of squares: %.2f"
                          % np.mean((regr.predict(X_test) - y_test) ** 2))
                    # Explained variance score: 1 is perfect prediction
                    print('Variance score: %.2f' % regr.score(X_test, y_test))
                    plt.figtext(.75, .9, 'Variance score: {}'.format(regr.score(X_test, y_test)))
                    plt.scatter(X_test, y_test, color='red')
                    plt.plot(X_test, regr.predict(X_test), color='blue')
                    plt.ylabel(target_col)

                    plt.xlabel(str(col))
                    plt.show()
                pass


# Ciekawe HR (bpm) : Altitude (m)
# Cadence : Altitude (m)

if __name__ == '__main__':
    file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/csvs/activity_5138499431.csv'
    analyzer = Analyzer(file)
