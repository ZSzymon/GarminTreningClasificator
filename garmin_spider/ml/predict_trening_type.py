import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from numpy import where
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
classifiers = {
    "SGD Classifier": SGDClassifier(),
    'Logistic Regression:': LogisticRegression(random_state=0),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
    "Random Forest Classifier": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM" :  SVC(gamma=2, C=1),
    "Naive Bayes": GaussianNB(),
}


def read_data(file):
    df = pd.read_csv(file)
    X = np.array(df.iloc[:, 2:-2].apply(pd.to_numeric))
    y = np.array(df.iloc[:, -1].apply(pd.to_numeric))
    return X, y


def get_accurancy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accurancy = accuracy_score(y_test, predictions)
    return predictions, accurancy


def createModel(X, y, classifier):
    clf = make_pipeline(StandardScaler(), classifier)
    clf.fit(X, y)
    return clf

def plot_dataset(X, y):
    counter = Counter(y)
    print(counter)
    # scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    oversample_data = True
    undersample_data = False
    steps = []

    if undersample_data:
        steps.append(('u', RandomUnderSampler(sampling_strategy={0: 100})))
        #better not use :)
    if oversample_data:
        steps.append(('o', SMOTE(random_state=101, k_neighbors=2)))


    file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled.csv'
    X, y = read_data(file)
    plot_dataset(X,y)

    pipeline = Pipeline(steps = steps)
    X_over, y_over = pipeline.fit_resample(X, y)
    print(Counter(y_over))
    plot_dataset(X_over, y_over)


    X_train_over, X_test_over, y_train_over, y_test_over = \
        train_test_split(X_over, y_over, test_size=.25, random_state=42)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.25, random_state=42)
    df = pd.DataFrame(columns=['Model', 'Accurancy','Accurancy over','Diffrence','Improvment'])
    i = 0

    for name, classifier in classifiers.items():
        clf = createModel(X_train, y_train, classifier)
        predictions, accurancy = get_accurancy(clf, X_test, y_test)
        clf_over = createModel(X_train_over, y_train_over, classifier)

        predictions_over, accurancy_over = get_accurancy(clf_over, X_test_over, y_test_over)

        delta = accurancy_over - accurancy
        df.loc[i] = [name, accurancy, accurancy_over, delta, (delta/accurancy)*100]
        i += 1

    print(df)
    #training_types = {
    #    'BC 1': 0,
    #    'BC 2': 1,
    #    'BC 3': 2,
    #    'fartelek': 3,
    #    'Trening Specjalistyczny': 4,
    #    'BC 1 + RT': 5,
    #    'Rozgrzewka': 6,  #
    #    'RT': 7,
    #    'Inny': -1,
    #}
