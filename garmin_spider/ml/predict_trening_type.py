import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tools import to_seconds_time
from tools import save_model

if __name__ == '__main__':
    file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled.csv'
    df = pd.read_csv(file)
    columns_to_delete = []

    X = np.array(df.iloc[ :, 2:-2].apply(pd.to_numeric))
    y = np.array(df.iloc[ :, -1].apply(pd.to_numeric))


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = make_pipeline(StandardScaler(), SGDClassifier())
    clf.fit(X_train, y_train)
    save_model(clf, '/home/zywko/PycharmProjects/BA_Code/resources/training_predicator.model')

    predictions = clf.predict(X_test)
    print(predictions[0:20])
    print(y_test[0:20])
    accurancy = accuracy_score(y_test,predictions)
    print(accurancy)
    training_types = {
        'BC 1' : 0,
        'BC 2' : 1,
        'BC 3' : 2,
        'fartelek': 3,
        'Trening Specjalistyczny': 4,
        'BC 1 + RT': 5,
        'Rozgrzewka': 6,#
        'RT' : 7,
        'Inny' : -1,
    }
