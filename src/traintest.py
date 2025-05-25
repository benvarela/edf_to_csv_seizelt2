'''
This file implements the machine learning pipeline, and outputs the precision, accuracy and run time to classify the testing data

INPUTS: Requires the csvs in csvs/traintest that contain the train and testing features sets
OUTPUTS: Outputs a results csv, with columns Model, PCA, Precision, Accuracy, Runtime.
'''

## Imports
# sci-kit-learn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, accuracy_score
# other imports
import pandas as pd
import numpy as np
from time import process_time

def traintest(csv_path: str, cores: int) -> None:
    '''
    Implements the training and testing, as described in the file header

    Note: now includes the cores parameter, so the user can set how many cores they wish to use.
    '''

    ## Import the training and testing feature sets, unpack
    train = pd.read_csv(f'{csv_path}/train.csv.gz', compression='gzip').to_numpy()
    test = pd.read_csv(f'{csv_path}/test.csv.gz', compression='gzip').to_numpy()
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    ## Instantiate models and their hyperparameter tuning dictionaries
    models = {
    'KNN': [KNeighborsClassifier(weights='distance'),
            {'n_neighbors': [3, 5, 7, 9, 11]}],

    'SVM': [LinearSVC(class_weight='balanced',
                      max_iter=1000,
                      random_state=42),
            {'C': [0.01, 0.1, 1, 10, 100]}],

    'DT': [DecisionTreeClassifier(criterion='entropy',
                                  class_weight='balanced',
                                  random_state=42),
           {'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10]}],

    'RF': [RandomForestClassifier(criterion='entropy',
                                  n_estimators=200,
                                  class_weight='balanced',
                                  random_state=42,
                                  n_jobs=cores),
           {'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 5, 10]}],

    'MLP': [MLPClassifier(hidden_layer_sizes=(100,),
                          max_iter=1000,
                          random_state=42),
            {'alpha': [1e-5, 1e-4, 1e-3, 1e-2]}],

    'LR': [LogisticRegression(class_weight='balanced',
                              max_iter=1000,
                              random_state=42),
           {'C': [0.01, 0.1, 1, 10, 100]}]
}


    ## Produce a list of PCA reductions of the training and test feature sets
    # First entry without PCA, manual
    PCA_traintest = {12: [X_train, X_test]}
    # Iterative addition of the remaining entries
    for k in range(11, 1, -1):
        pca = PCA(n_components=k)
        X_train_k = pca.fit_transform(X_train)
        if np.sum(pca.explained_variance_ratio_) > 0.95:
            X_test_k = pca.transform(X_test)
            PCA_traintest[k] = [X_train_k, X_test_k]
        else:
            break


    ## Made a custom precision scorer that evaluates 0 as the positive label
    custom_precision = make_scorer(precision_score, pos_label=0)

    ## Instantiate the dictionary to store the results
    results = {'model': list(),
            'PCA': list(),
            'precision': list(),
            'accuracy': list(),
            'time': list(),
            'best_params': list()
}

    ## Instantiated the stratifiedKFold object
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ## For loop implementation
    for dim in PCA_traintest:
        for model in models:
            # Unpack
            X_train, X_test = PCA_traintest[dim]
            mod, params = models[model]

            # GridSearch
            grid = GridSearchCV(estimator=mod, param_grid=params, scoring=custom_precision, cv=cv, n_jobs=cores)
            grid.fit(X_train, y_train)

            # Use the best model, and output the metrics we're interested in
            t = process_time()
            y_pred = grid.predict(X_test)
            time_taken = process_time() - t
            precision = precision_score(y_test, y_pred, pos_label = 0, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            # Add the results
            results['model'].append(model)
            results['PCA'].append(dim)
            results['precision'].append(precision)
            results['accuracy'].append(accuracy)
            results['time'].append(time_taken)
            results['best_params'].append(str(grid.best_params_))

            # Log
            print(f'Completed PCA: {dim}, model: {model}')

    ## Export the results
    pd.DataFrame(results).to_csv('csvs/results.csv', index=False)

    return None