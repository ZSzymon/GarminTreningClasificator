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
from sklearn.metrics import precision_recall_fscore_support

from os import path
from sklearn import metrics
import seaborn as sns

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

classifiers = {
    "SGD Classifier": SGDClassifier(),
    'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(max_depth=5),
    "Random Forest Classifier": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Naive Bayes": GaussianNB(),
}


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


def plot_confucion_matrix(y_test, predictions, score, path_to_save, name):
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');

    all_sample_title = '{}\nAccuracy Score: {:.3f}'.format(name, score, )
    plt.title(all_sample_title, size=15)
    plt.savefig(path_to_save + '.png')


def plot_matrix(clf, X_test, y_test, save_path, name):
    predictions, accurancy = get_accurancy(clf_over, X_test, y_test)
    labels = ['BC 1', 'BC 2', 'BC 3', 'Interwały', 'BC 1 + RT', 'Rozgrzewka']

    plot_confusion_matrix(clf, X_test, y_test, normalize='true')
    plt.title(name + "\nDokładność:{:.3f}".format(accurancy))
    plt.ylabel('Prawdziwy typ');
    plt.xticks(rotation=45)
    plt.xlabel('Przewidziany typ');
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()


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


def tree_ploter(X, y):
    labels_names = ['BC 1', 'BC 2', 'BC 3',
                    'Interwały', 'BC 1 + RT', 'Rozgrzewka', 'RT',
                    ]
    features_name = list(pd.read_csv(file, nrows=1).columns)[1:-2]

    fig = plt.figure(figsize=(150, 120))
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    _ = tree.plot_tree(clf, feature_names=features_name, class_names=labels_names, filled=True)
    plt.savefig('/home/zywko/PycharmProjects/BA_Code/resources/garmin_plots/tree_jakub_szymon_medium.png')


def prepare_data(file):
    X, y = read_data(file, {'Type': ['RT', '7'], })
    y = change_labels(y)
    return X, y


def precision_recall_fscore_support_extend(y_true, y_pred, name):
    save_dir = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_plots'
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


if __name__ == '__main__':
    run_classificators = True
    oversample_data = True
    undersample_data = False
    plot_tree = True
    plot_matrix_decision = False
    create_scores = True
    steps = []
    if undersample_data:
        steps.append(('u', RandomUnderSampler(sampling_strategy={0: 100})))
        # better not use :)
    if oversample_data:
        steps.append(('o', SMOTE(random_state=101, k_neighbors=7)))

    file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_jakob_szymon_improved.csv'

    X, y = prepare_data(file)
    plot_dataset(X, y)

    pipeline = Pipeline(steps=steps)
    X_over, y_over = pipeline.fit_resample(X, y)
    print(Counter(y_over))
    plot_dataset(X_over, y_over)

    X_train_over, X_test_over, y_train_over, y_test_over = \
        train_test_split(X_over, y_over, test_size=.20, random_state=42)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.20, random_state=42)

    df = pd.DataFrame(columns=['Model', 'Accurancy', 'Accurancy over', 'Diffrence', 'Improvment'])

    i = 0
    garmin_dir = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_plots/v3'
    if plot_tree:
        tree_ploter(X=X_train_over, y=y_train_over)

    if run_classificators:
        for name, classifier in classifiers.items():
            clf = createModel(X_train, y_train, classifier)
            predictions, accurancy = get_accurancy(clf, X_test, y_test)

            clf_over = createModel(X_train_over, y_train_over, classifier)
            predictions_over, accurancy_over = get_accurancy(clf_over, X_test_over, y_test_over)

            if plot_matrix_decision:
                plot_matrix(clf_over, X_test_over, y_test_over,
                            path.join(garmin_dir, name), name)

            # plot_matrix(clf, X_test, y_test,
            #           path.join(garmin_dir, name), name)
            if create_scores:
                precision_recall_fscore_support_extend(y_test_over, predictions_over, name)

            delta = accurancy_over - accurancy
            df.loc[i] = [name, accurancy, accurancy_over, delta, (delta / accurancy_over) * 100]

            i += 1
        print(df)
