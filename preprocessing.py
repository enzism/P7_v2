import re
import numpy as np
import pandas as pd
import gc


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns, categorical_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(train, test,  num_rows=None, nan_as_category=False):
    print("Train samples: {}, test samples: {}".format(len(train), len(test)))
    train = train.append(test).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    train = train[train['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        train[bin_feature], uniques = pd.factorize(train[bin_feature])
        train[bin_feature] = train[bin_feature].astype('category')

    for cat_feature in ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                        ]:
        train[cat_feature] = train[cat_feature].astype('category')

    # Categorical features with One-Hot encode
    train, cat_cols, categories_that_have_been_encoded = one_hot_encoder(train, nan_as_category)
    train[cat_cols] = train[cat_cols].astype('category')
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    train['DAYS_EMPLOYED_PERC'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']
    train['INCOME_CREDIT_PERC'] = train['AMT_INCOME_TOTAL'] / train['AMT_CREDIT']
    train['INCOME_PER_PERSON'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS']
    train['ANNUITY_INCOME_PERC'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
    train['PAYMENT_RATE'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
    train.drop(columns=['index'], inplace=True)
    train['TARGET'] = train['TARGET'].astype('category')

    for doc in range(2, 22):
        train['FLAG_DOCUMENT_{0}'.format(doc)] = train['FLAG_DOCUMENT_{0}'.format(doc)].astype('category')

    for cat_feature in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                        'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']:
        train[cat_feature] = train[cat_feature].astype('category')

    del test
    gc.collect()
    print(categories_that_have_been_encoded)
    return train.dropna(), categories_that_have_been_encoded


def feature_engineering(train, test):
    df, categories_that_have_been_encoded = application_train_test(train, test)
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    X = df.drop(['TARGET'], axis=1)
    y = df['TARGET'].astype('int32')
    print(f"X shape : {X.shape}, y shape : {y.shape}")
    X.to_csv('export/train_X.csv')
    y.to_csv('export/train_y.csv')
    df.to_csv('export/train_data.csv')
    return X, y, categories_that_have_been_encoded

# environnement conda :
# $conda create -n projet7 python=3.7
# preferences >> python interpreter >> conda environnement >> existing >> interpreter : path to venv_name/bin/python
# pip install -r resquirements.txt