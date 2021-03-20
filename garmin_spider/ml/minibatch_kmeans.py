import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_garmin_labeled_ready.csv'
    df = pd.read_csv(file)

    X = df.iloc[:, 2:-2].apply(pd.to_numeric)

    X_train, X_test = train_test_split(X, test_size=0.2)
    shape = X_train.shape
    model = MiniBatchKMeans(n_clusters=4,
                            random_state=0,
                            batch_size=6)
    half_index = int(float(shape[0])/2.0)
    first_half = X_train.iloc[:half_index]
    second_half = X_train.iloc[half_index:]
    model.partial_fit(first_half)
    model.partial_fit(second_half)

    predictions = model.predict(X_test)
    accurancy = accuracy_score(predictions,X_test)
    clusters = model.cluster_centers_
    print(predictions[0:20])
