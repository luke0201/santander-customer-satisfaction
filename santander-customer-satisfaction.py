from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


def load_data(path):
    df = pd.read_csv(path)

    return df


def split_feature_label(Xy):
    y = Xy['TARGET']
    X = Xy.drop('TARGET', axis=1, inplace=False)

    return X, y


def transform(X):
    X.drop('ID', axis=1, inplace=True)

    X['var3'].replace(-999999, 2, inplace=True)

    return X


def train_model(X, y):
    param_grid = {
        'num_leaves': [32, 64],
        'max_depth': [128, 160],
        'min_child_samples': [60, 100],
        'subsample': [0.8, 1]
    }
    lgbm_classifier = LGBMClassifier(n_estimators=200)
    grid_search_cv = GridSearchCV(lgbm_classifier, param_grid=param_grid, cv=3)

    grid_search_cv.fit(X, y, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X, y)])

    return grid_search_cv.best_estimator_


def parse_args():
    parser = ArgumentParser(
        description='Generate the submission file for Kaggle Santander Customer Satisfaction competition.')
    parser.add_argument(
        '--train', type=Path, default='train.csv',
        help='path of train.csv downloaded from the competition')
    parser.add_argument(
        '--test', type=Path, default='test.csv',
        help='path of test.csv downloaded from the competition')

    return parser.parse_args()


def main(args):
    Xy_train = load_data(args.train)
    X_train, y_train = split_feature_label(Xy_train)
    X_train = transform(X_train)
    model = train_model(X_train, y_train)

    X_test = load_data(args.test)
    ids = X_test['ID']
    X_test = transform(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    submission = {
        'ID': ids,
        'TARGET': y_score
    }
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
