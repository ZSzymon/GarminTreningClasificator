from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import json
import sys
from tools import *

settings = load_settings()

if __name__ == '__main__':
    run_classificators = True
    oversample_data = True
    undersample_data = False
    plot_tree = False
    plot_matrix_decision = True
    create_scores = False
    plot_report_decission = True
    plot_dataset_decision = False
    steps = []
    if undersample_data:
        steps.append(('u', RandomUnderSampler(sampling_strategy={0: 100})))
        # better not use :)
    if oversample_data:
        steps.append(('o', SMOTE(random_state=101, k_neighbors=7)))

    # file = '/home/zywko/PycharmProjects/BA_Code/resources/garmin_data/summary_labeled_garmin.csv'
    file = path.join(settings['GARMIN_DATA'], 'summary_labeled_garmin.csv')

    X, y = prepare_data(file)
    if plot_dataset_decision:
        plot_dataset(X, y)

    pipeline = Pipeline(steps=steps)
    X_over, y_over = pipeline.fit_resample(X, y)
    print(Counter(y_over))
    if plot_dataset_decision:
        plot_dataset(X_over, y_over)

    X_train_over, X_test_over, y_train_over, y_test_over = \
        train_test_split(X_over, y_over, test_size=.20, random_state=42)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.20, random_state=42)

    df = pd.DataFrame(columns=['Model', 'Accurancy', 'Accurancy over', 'Diffrence', 'Improvment'])

    i = 0
    garmin_plots_dir = path.join(settings['GARMIN_PLOTS'], 'tmp')
    if plot_tree:
        tree_ploter(X=X_train_over, y=y_train_over, file=file,
                    save_path=path.join(settings['GARMIN_PLOTS'], 'tree_plots', 'tree.png'))

    if run_classificators:
        for name, classifier in get_classifiers().items():
            clf = createModel(X_train, y_train, classifier)
            predictions, accurancy = get_accurancy(clf, X_test, y_test)

            clf_over = createModel(X_train_over, y_train_over, classifier)
            predictions_over, accurancy_over = get_accurancy(clf_over, X_test_over, y_test_over)

            if plot_matrix_decision:
                plot_matrix(clf_over, X_test_over, y_test_over,
                            path.join(garmin_plots_dir, name + "after_oversampling"),
                            name + " z nadpróbkowaniem", MatrixChoices.SAVE)
                plot_matrix(clf, X_test, y_test,
                            path.join(garmin_plots_dir, name + "_before"),
                            name + " bez nadpróbkowania", MatrixChoices.SAVE)

            if create_scores:
                precision_recall_fscore_support_extend(clf_over, y_test_over, predictions_over, name)

            if plot_report_decission:
                plot_classification_report(clf_over, X_test_over, y_test_over, name)

            delta = accurancy_over - accurancy
            df.loc[i] = [name, accurancy, accurancy_over, delta, (delta / accurancy_over) * 100]

            i += 1
        print(df)
