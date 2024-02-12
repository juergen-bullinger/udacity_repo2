"""
This file contains the library of functions used for the churn analysis

Created on feb 10th 2024

@author: Jürgen Bullinger
"""


# import libraries
import os
import logging
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        logging.info("reading %s", pth)
        df_data = pd.read_csv(pth)
    except FileNotFoundError as ex:
        logging.error(
            "file %s was not found - import not possible: %s",
            pth,
            ex)
        return None
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df_data


def perform_eda(df_data):
    '''
    perform eda on df_data and save figures to images folder

    input:
            df_data: pandas dataframe

    output:
            None
    '''
    # create the histogram images for Churn and Customer_Age
    for col in ["Churn", "Customer_Age"]:
        logging.info("eda - creating histogram for %s", col)
        plt.figure(figsize=(20, 10))
        df_data[col].hist()
        plt.savefig(f"./images/eda/hist_{col}.png")

    # create a bar chart for the Marital Status
    col = "Marital_Status"
    logging.info("eda - creating diagram for %s", col)
    plt.figure(figsize=(20, 10))
    df_data[col].value_counts('normalize').plot(kind='bar')
    plt.savefig(f"./images/eda/bar_{col}.png")

    # hist with density for Total_Trans_Ct
    col = "Total_Trans_Ct"
    logging.info("eda - creating diagram for %s", col)
    plt.figure(figsize=(20, 10))
    sns.histplot(df_data[col], stat='density', kde=True)
    plt.savefig(f"./images/eda/hist_with_density_{col}.png")

    # heatmap for correlations
    logging.info("eda - creating heatmap for correlations")
    plt.figure(figsize=(20, 10))
    df_num_cols = df_data.select_dtypes(exclude="object")
    sns.heatmap(df_num_cols.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/heatmap_corr.png")


def categorize_helper(df_data, column_name, response):
    '''
    categorize the spezfied column name by creating a column named
    "{column_name}_{response}"

    input:
            df_data: pandas dataframe
            column_name: name of the column to be categorized
            response: response column name (target column name)
    output:
            result dataframe
    '''
    category_lst = []
    gender_groups = df_data.groupby(column_name).mean()[response]
    for val in df_data[column_name]:
        category_lst.append(gender_groups.loc[val])

    df_data[f'{column_name}_{response}'] = category_lst
    return df_data


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
    logging.info("starting column encoding...")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    for cat_col in category_lst:
        if cat_col != response:
            # skip the target column
            logging.info("processing categorical column %s.", cat_col)
            df_data = categorize_helper(df_data, cat_col, "Churn")
    logging.info("column encoding finished")
    return df_data


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
    logging.info("performing feature engineering")

    # get the list of object columns, which are all considered to be categories
    cat_cols = df_data.select_dtypes(include="object").columns.to_list()
    df_data = encoder_helper(df_data, cat_cols, "Churn")
    feature_cols = [col for col in df_data.columns if col != response]
    return train_test_split(
        df_data[feature_cols],
        df_data[response],
        test_size=0.3,
        random_state=42
    )


def save_classification_report_image(y_true, y_pred, title, pth):
    '''
    creates a classification report and plots it using seaborn.

    input:
            y_true: true values (from test set)
            y_pred:  predicted values
            title: title to be displayed on the diagram
            pth: file name of the output file

    output:
             None
    '''
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    plt.figure(figsize=(20, 10))
    plt.title(title)
    sns.heatmap(pd.DataFrame(cls_report).iloc[:-1, :].T)
    plt.savefig(pth)


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
    logging.info('generating images for random forest results')
    save_classification_report_image(
        y_test,
        y_test_preds_rf,
        "random forest classification results - test",
        "images/results/random_forest_test.png",
    )

    save_classification_report_image(
        y_train,
        y_train_preds_rf,
        "random forest classification results - train",
        "images/results/random_forest_train.png",
    )

    logging.info('generating images for logistic regression results')
    save_classification_report_image(
        y_test,
        y_test_preds_lr,
        "logistic regression classification results - test",
        "images/results/random_log_test.png",
    )

    save_classification_report_image(
        y_train,
        y_train_preds_lr,
        "logistic regression classification results - train",
        "images/results/random_log_train.png",
    )


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
    image_file = f"{output_pth}/feature_importances.png"
    logging.info("generating the image for the feature importances")
    logging.info("storing result in %s", image_file)
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(image_file)


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
    logging.info("training models")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)


if __name__ == "__main__":
    df_bank_data = import_data("./data/bank_data.csv")
    if df_bank_data is not None:
        # we got some data, so we can continue
        perform_eda(df_bank_data)
        perform_feature_engineering(df_bank_data, "Churn")
        feature_importance_plot(model, df_bank_data, output_pth)
