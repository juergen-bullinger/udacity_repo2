"""
Unit tests for logging.

Author: Jürgen Bullinger
Date: 10.02.2024
"""

import io
import os
import logging
import churn_library as cls

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import pandas as pd

logging.basicConfig(
    filename=f'./logs/{__name__}.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

EXAMPLE_CSV = """,CLIENTNUM,Attrition_Flag,Customer_Age,Gender,Dependent_count,Education_Level,Marital_Status,Income_Category,Card_Category,Months_on_book,Total_Relationship_Count,Months_Inactive_12_mon,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio
0,768805383,Existing Customer,45,M,3,High School,Married,$60K - $80K,Blue,39,5,1,3,12691.0,777,11914.0,1.335,1144,42,1.625,0.061
1,818770008,Existing Customer,49,F,5,Graduate,Single,Less than $40K,Blue,44,6,1,2,8256.0,864,7392.0,1.541,1291,33,3.714,0.105
2,713982108,Existing Customer,51,M,3,Graduate,Married,$80K - $120K,Blue,36,4,1,0,3418.0,0,3418.0,2.594,1887,20,2.333,0.0
3,769911858,Existing Customer,40,F,4,High School,Unknown,Less than $40K,Blue,34,3,4,1,3313.0,2517,796.0,1.405,1171,20,2.333,0.76
4,709106358,Existing Customer,40,M,3,Uneducated,Married,$60K - $80K,Blue,21,5,1,0,4716.0,0,4716.0,2.175,816,28,2.5,0.0
5,713061558,Existing Customer,44,M,2,Graduate,Married,$40K - $60K,Blue,36,3,1,2,4010.0,1247,2763.0,1.376,1088,24,0.846,0.311
6,810347208,Existing Customer,51,M,4,Unknown,Married,$120K +,Gold,46,6,1,3,34516.0,2264,32252.0,1.975,1330,31,0.722,0.066
7,818906208,Existing Customer,32,M,0,High School,Unknown,$60K - $80K,Silver,27,2,2,2,29081.0,1396,27685.0,2.204,1538,36,0.7140000000000001,0.048
8,710930508,Existing Customer,37,M,3,Uneducated,Single,$60K - $80K,Blue,36,5,2,0,22352.0,2517,19835.0,3.355,1350,24,1.182,0.113
9,719661558,Existing Customer,48,M,2,Graduate,Single,$80K - $120K,Blue,36,6,3,3,11656.0,1677,9979.0,1.524,1441,32,0.882,0.14400000000000002
21,708508758,Attrited Customer,62,F,0,Graduate,Married,Less than $40K,Blue,49,2,3,3,1438.3,0,1438.3,1.047,692,16,0.6,0.0
39,708300483,Attrited Customer,66,F,0,Doctorate,Married,Unknown,Blue,56,5,4,3,7882.0,605,7277.0,1.052,704,16,0.14300000000000002,0.077
51,779471883,Attrited Customer,54,F,1,Graduate,Married,Less than $40K,Blue,40,2,3,1,1438.3,808,630.3,0.997,705,19,0.9,0.562
54,714374133,Attrited Customer,56,M,2,Graduate,Married,$120K +,Blue,36,1,3,3,15769.0,0,15769.0,1.041,602,15,0.364,0.0
61,712030833,Attrited Customer,48,M,2,Graduate,Married,$60K - $80K,Silver,35,2,4,4,34516.0,0,34516.0,0.763,691,15,0.5,0.0
82,711013983,Attrited Customer,55,F,4,Unknown,Married,$40K - $60K,Blue,45,2,4,3,2158.0,0,2158.0,0.585,615,12,0.7140000000000001,0.0
99,711887583,Attrited Customer,47,M,2,Unknown,Married,$80K - $120K,Blue,37,2,3,3,5449.0,1628,3821.0,0.696,836,18,0.385,0.299
127,720201033,Attrited Customer,53,M,2,Graduate,Married,$80K - $120K,Blue,41,3,3,2,11669.0,2227,9442.0,0.622,720,23,0.353,0.191
140,789322833,Attrited Customer,48,F,5,High School,Married,Less than $40K,Blue,38,1,3,3,8025.0,0,8025.0,0.654,673,18,0.8,0.0
144,767712558,Attrited Customer,59,M,1,College,Single,$60K - $80K,Blue,53,2,3,3,14979.0,0,14979.0,0.71,530,10,1.0,0.0"""


def import_raw_test_data():
    """
    create a test data frame from the first 10 lines of the csv for each
    category as of feb 2024 (20 lines in total)
    """
    string_buffer = io.StringIO(EXAMPLE_CSV)
    df_data = pd.read_csv(string_buffer, index_col=0)
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    df_data.drop(columns=["Attrition_Flag"], inplace=True)
    return df_data


@pytest.fixture
def raw_test_data():
    """
    create a test data frame from the first 10 lines of the csv for each
    category as of feb 2024 (20 lines in total)
    """
    return import_raw_test_data()


def helper_create_subdirs(pth):
    """
    create the sub directory structure for testing
    """
    pth = Path(pth)
    for subdir in ("data", "images", "images/eda", "images/results", "logs", 
                   "models"):
        absolute_subdir = pth / subdir
        logging.info("creating directory %s", absolute_subdir)
        absolute_subdir.mkdir(parents=True)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_bank_data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_bank_data.shape[0] > 0
        assert df_bank_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and "
            "columns: %s", err
        )
        raise err


def test_eda(raw_test_data):
    '''
    test perform eda function
    '''
    logging.info("start of test_eda")
    curr_path = Path.cwd()
    with TemporaryDirectory() as td:
        try:
            # prepare test directory structure and switch to it's root
            test_path = Path(td)
            os.chdir(str(test_path))
            helper_create_subdirs(test_path)
            
            # execute code
            cls.perform_eda(raw_test_data)
            
            # asserts (check if all result files are created)
            for file in [
                        "hist_Churn.png", 
                        "hist_Customer_Age.png",
                        "bar_Marital_Status.png",
                        "hist_with_density_Total_Trans_Ct.png",
                        "heatmap_corr.png",
                    ]:
                file_path = test_path / "images" / "eda" / file
                logging.info("checking presence of %s", file_path)
                assert file_path.exists()
            logging.info("Testing eda: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_eda: %s",
                err
            )
            raise err
        finally:
            # switch back to the original path
            os.chdir(curr_path)


def test_encoder_helper(raw_test_data):
    '''
    test encoder helper - check if the Churn-columns are generated
    '''
    logging.info("start of test_encoder_helper")
    cat_cols = ['Card_Category', 'Education_Level']
    
    try:
        # execute code
        df_result = cls.encoder_helper(raw_test_data, cat_cols, "Churn")
        
        # asserts (check if all result files are created)
        for column in cat_cols:
            encoded_col = f"{column}_Churn"
            assert encoded_col in df_result.columns, (
                f"{encoded_col} was not generated"
            )
        logging.info("Testing encoder helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "test_encoder_helper: %s",
            err
        )
        raise err



def test_perform_feature_engineering(raw_test_data):
    '''
    test perform_feature_engineering
    '''
    logging.info("start of test_perform_feature_engineering")
    try:
        # data_tuple is expected to contain:
        # x_train_d, x_test_d, y_train_d, y_test_d
        data_tuple = cls.perform_feature_engineering(raw_test_data, "Churn")
        assert len(data_tuple) == 4
        assert len(data_tuple[0]) == 14 # x_train_d
        assert len(data_tuple[1]) == 6 # x_test_d
        assert len(data_tuple[2]) == 14 # y_train_d
        assert len(data_tuple[3]) == 6 # y_test_d
        logging.info("Testing perform feature engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "test_perform_feature_engineering: %s",
            err
        )
        raise err


def test_train_models(raw_test_data):
    '''
    test train_models
    '''
    logging.info("start of test_train_models")
    # prepare test data
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        raw_test_data, 
        "Churn"
    )
    
    curr_path = Path.cwd()
    with TemporaryDirectory() as td:
        try:
            # prepare test directory structure and switch to it's root
            test_path = Path(td)
            os.chdir(str(test_path))
            helper_create_subdirs(test_path)
            
            # execute code
            cls.train_models(x_train, x_test, y_train, y_test)
            
            # asserts (check if all result files are created)
            for file in [
                        "feature_importances_random_forest.png", 
                        "roc_random_forest.png",
                        "roc_linear_regression.png",
                    ]:
                file_path = test_path / "images" / "results" / file
                logging.info("checking presence of %s", file_path)
                assert file_path.exists()
            logging.info("Testing train models: SUCCESS")
        except AssertionError as err:
            logging.error(
                "test_eda: %s",
                err
            )
            raise err
        finally:
            # switch back to the original path
            os.chdir(curr_path)



if __name__ == "__main__":
    # don't know why we would need this, since this is usally run
    # from pytest, which also handles a great deal of logging,
    # but according to the sketched solution it looks like we
    # do not use features of pytest
    try:
        test_import()
    except AssertionError as err:
        logging.error("an exception was raised in test_import: ", err)
        
    try:
        test_eda(import_raw_test_data())
    except AssertionError as err:
        logging.error("an exception was raised in test_eda: ", err)

    try:
        test_encoder_helper(import_raw_test_data())
    except AssertionError as err:
        logging.error("an exception was raised in test_encoder_helper: ", err)

    try:
        test_perform_feature_engineering(import_raw_test_data())
    except AssertionError as err:
        logging.error(
            "an exception was raised in test_perform_feature_engineering: ", 
            err
        )

    try:
        test_perform_feature_engineering(import_raw_test_data())
    except AssertionError as err:
        logging.error(
            "an exception was raised in test_perform_feature_engineering: ", 
            err
        )



