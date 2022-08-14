from lightgbm import LGBMClassifier
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
from imblearn.pipeline import Pipeline as imbpipeline
import pickle
import pandas as pd
from constants import MODEL_PATH, params, TRAIN_PATH, TEST_PATH
from preprocessing import  feature_engineering


def train_model(X, y, model_path=MODEL_PATH, train=False):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=11)
    clf = LGBMClassifier()

    pipeline = imbpipeline(steps=[['smote', SMOTE(random_state=11)], ['classifier', clf]])

    param_grid = {
        'classifier__learning_rate': [0.005, 0.01],
        'classifier__n_estimators': [100],
        'classifier__num_leaves': [6, 8, 12, 16],  # large num_leaves helps improve accuracy but might lead to over-fitting
        'classifier__boosting': ['gbdt', 'dart'],  # for better accuracy -> try dart
        'classifier__objective': ['binary'],
        'classifier__max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
        'classifier__random_state': [500],
        'classifier__subsample': [0.7, 0.75],
        'classifier__reg_alpha': [1, 1.2],
        'classifier__reg_lambda': [1, 1.2, 1.4]
    }
    print(54*'#')
    print('Cross validation')

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=4,
                               n_jobs=10,
                               verbose=1)

    grid_search.fit(X_train, y_train)
    cv_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    print(54 * '#')
    clf = grid_search.best_estimator_
    pickle.dump(clf, open(model_path, 'wb'))

    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

    return clf



train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X, y, categories_that_have_been_encoded = feature_engineering(train_df, test_df)
train_model(X, y)

