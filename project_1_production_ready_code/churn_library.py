"""
This file contains the library of functions used for the churn analysis

Created on feb 10th 2024

@author: Jürgen Bullinger
"""


# import libraries
import os
import pandas as pd

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df_data):
    '''
    perform eda on df_data and save figures to images folder

    input:
            df_data: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # gender encoded column
    gender_lst = []
    gender_groups = df_data.groupby('Gender').mean()['Churn']

    for val in df_data['Gender']:
        gender_lst.append(gender_groups.loc[val])

    df_data['Gender_Churn'] = gender_lst
    # education encoded column
    edu_lst = []
    edu_groups = df_data.groupby('Education_Level').mean()['Churn']

    for val in df_data['Education_Level']:
        edu_lst.append(edu_groups.loc[val])

    df_data['Education_Level_Churn'] = edu_lst

    # marital encoded column
    marital_lst = []
    marital_groups = df_data.groupby('Marital_Status').mean()['Churn']

    for val in df_data['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])

    df_data['Marital_Status_Churn'] = marital_lst

    # income encoded column
    income_lst = []
    income_groups = df_data.groupby('Income_Category').mean()['Churn']

    for val in df_data['Income_Category']:
        income_lst.append(income_groups.loc[val])

    df_data['Income_Category_Churn'] = income_lst

    # card encoded column
    card_lst = []
    card_groups = df_data.groupby('Card_Category').mean()['Churn']

    for val in df_data['Card_Category']:
        card_lst.append(card_groups.loc[val])

    df_data['Card_Category_Churn'] = card_lst


def perform_feature_engineering(df_data, response):
    '''
    input:
              df_data: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    feature_cols = [col for col in df_data.columns if col != response]
    return train_test_split(
        df_data[feature_cols],
        df_data[response],
        test_size=0.3,
        random_state=42
    )


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)


if __name__ == "__main__":
    pass
