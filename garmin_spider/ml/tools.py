import sys
import json
import seaborn as sns
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
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from collections.abc import Iterable
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from yellowbrick.classifier import ClassificationReport
from enum import Enum
from os import path
from sklearn import metrics
from deprecated import deprecated


def prepare_data(file):
    X, y = read_data(file, {'Type': ['RT', '7'], })
    y = change_labels(y)
    return X, y

def get_training_types():
    training_types = {
        'BC 1': 0,
        'BC 2': 1,
        'BC 3': 2,
        'fartelek': 3,
        'BC 1 + RT': 5,
        'Rozgrzewka': 6,  #
        'RT': 7,
        'Inny': -1,
    }
    return training_types
def get_classifiers():
    classifiers = {
        "SGD Classifier": SGDClassifier(),
        'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=7),
        "Random Forest Classifier": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4),
        "Linear SVM": SVC(kernel="linear", C=4),
        "RBF SVM": SVC(gamma=.25, C=1),
        "Naive Bayes": GaussianNB(),
    }
    return classifiers


def read_data(file, not_allowed_col_val: dict = None):
    df = pd.read_csv(file)
    if not_allowed_col_val:
        for col, vals in not_allowed_col_val.items():
            if isinstance(vals, Iterable):
                for val in vals:
                    df = df[df[col] != val]
            else:
                df = df[df[col] != vals]

    X = np.array(df.iloc[:, 1:-2].apply(pd.to_numeric))
    y = np.array(df.iloc[:, -1])
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
    plt.clf()

def plot_classification_report(clf, X_test, y_test, name):
    plt.close('all')
    plt.clf()
    classes = list(np.unique(y_test))
    visualizer = ClassificationReport(clf, classes=classes, support=True, is_fitted='true')
    visualizer.score(X_test, y_test)
    visualizer.set_title(name)
    visualizer.show()

    pass
@deprecated('Old function. Use plot_matrix instead')
def plot_confucion_matrix(y_test, predictions, score, path_to_save, name):
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r', normalize='true')
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = '{}\nAccuracy Score: {:.3f}'.format(name, score, )
    plt.title(all_sample_title, size=15)
    plt.savefig(path_to_save + '.png')



class MatrixChoices(Enum):
    SAVE = 1
    SHOW = 2

def plot_matrix(clf, X_test, y_test, save_path, name, choice: MatrixChoices = MatrixChoices.SHOW):
    predictions, accurany = get_accurancy(clf, X_test, y_test)
    labels = list(np.unique(y_test))
    plt.grid(False)
    plot_confusion_matrix(clf, X_test, y_test, normalize='true')
    plt.grid(False)
    plt.title(name + "\nDokładność:{:.3f}".format(accurany))
    plt.ylabel('Prawdziwy typ',fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('Przewidziany typ',fontsize=12)
    plt.tight_layout()

    if choice == MatrixChoices.SHOW:
        plt.show()
    if choice == MatrixChoices.SAVE:
        plt.savefig(save_path)

    pass

def change_labels(y):
    training_types_labels = {
        '0': 'BC 1',
        '1': 'BC 2',
        '2': 'BC 3',
        '3': 'Interwały',
        '5': 'BC 1 + RT',
        '6': 'Rozgrzewka',  #
        '7': 'RT',
        '-1': 'Inny',
        'BC 1': 'BC 1',
        'BC 2': 'BC 2',
        'BC 3': 'BC 3',
        'fartelek': 'Interwały',
        'BC 1 + RT': 'BC 1 + RT',
        'Rozgrzewka': 'Rozgrzewka',  #
        'RT': 'RT',
        'Inny': 'Inny'

    }

    result = []
    for i, el in enumerate(y):
        result.append(training_types_labels[el])

    return np.array(result)


def tree_ploter(X, y, file, save_path, choice: MatrixChoices = MatrixChoices.SHOW):
    labels_names = ['BC 1', 'BC 2', 'BC 3',
                    'Interwały', 'BC 1 + RT', 'Rozgrzewka', 'RT',
                    ]
    features_name = list(pd.read_csv(file, nrows=1).columns)[1:-2]

    fig = plt.figure(figsize=(150, 120))
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    _ = tree.plot_tree(clf, feature_names=features_name, class_names=labels_names, filled=True)

    if choice == MatrixChoices.SHOW:
        plt.show()
    if choice == MatrixChoices.SAVE:
        plt.savefig(save_path)

    plt.clf()
    plt.close(fig)



def precision_recall_fscore_support_extend(clf,y_true, y_pred, name, save_dir):
    file_to_save = path.join(save_dir, name + '_scores.txt')
    labels = list(np.unique(y_true))

    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    labels = sorted(labels)
    with open(file_to_save, 'w+') as fp:
        fp.write('name: {}\n'.format(name))
        fp.write('labels: {}\n'.format(labels))
        fp.write('precision: {}\n'.format(precision))
        fp.write('recall: {}\n'.format(recall))
        fp.write('fscore: {}\n'.format(fscore))
        fp.write('support: {}\n'.format(support))

    pass

def load_settings():
    if sys.platform.startswith("win"):
        with open(
                "D:\\Szymon\\STUDIA\\SEMINARIA\\Python\\pythonProject\\BA_code_conda\\ba_v2\\garmin_spider\\\ml\\\settings_win.json") as fp:
            settings = json.load(fp)

    if sys.platform.startswith("linux"):
        with open("settings_linux.json") as fp:
            settings = json.load(fp)

    return settings