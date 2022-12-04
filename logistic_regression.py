"""
Module: Logistic Regression
"""

import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection, pipeline, preprocessing

from utils import (
    card_precision_top_k,
    card_precision_top_k_custom,
    get_train_test_set,
    model_selection_wrapper,
    read_from_files,
    scaleData,
)


def fit_model_and_get_predictions(
    classifier, train_df, test_df, input_features, output_feature="TX_FRAUD", scale=True
):

    # By default, scales input data
    if scale:
        (train_df, test_df) = scaleData(train_df, test_df, input_features)

    # We first train the classifier using the `fit` method, and pass as arguments the input and output features
    start_time = time.time()
    classifier.fit(train_df[input_features], train_df[output_feature])
    training_execution_time = time.time() - start_time

    # We then get the predictions on the training and test data using the `predict_proba` method
    # The predictions are returned as a numpy array, that provides the probability of fraud for each transaction
    start_time = time.time()
    predictions_test = classifier.predict_proba(test_df[input_features])[:, 1]
    prediction_execution_time = time.time() - start_time

    predictions_train = classifier.predict_proba(train_df[input_features])[:, 1]

    # The result is returned as a dictionary containing the fitted models,
    # and the predictions on the training and test sets
    model_and_predictions_dictionary = {
        "classifier": classifier,
        "predictions_test": predictions_test,
        "predictions_train": predictions_train,
        "training_execution_time": training_execution_time,
        "prediction_execution_time": prediction_execution_time,
    }

    return model_and_predictions_dictionary


def performance_assessment(
    predictions_df,
    output_feature="TX_FRAUD",
    prediction_feature="predictions",
    top_k_list=[100],
    rounded=True,
):

    AUC_ROC = metrics.roc_auc_score(
        predictions_df[output_feature], predictions_df[prediction_feature]
    )
    AP = metrics.average_precision_score(
        predictions_df[output_feature], predictions_df[prediction_feature]
    )

    performances = pd.DataFrame(
        [[AUC_ROC, AP]], columns=["AUC ROC", "Average precision"]
    )

    for top_k in top_k_list:

        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances["Card Precision@" + str(top_k)] = mean_card_precision_top_k

    if rounded:
        performances = performances.round(3)

    return performances


def performance_assessment_model_collection(
    fitted_models_and_predictions_dictionary,
    transactions_df,
    type_set="test",
    top_k_list=[100],
):

    performances = pd.DataFrame()

    for (
        classifier_name,
        model_and_predictions,
    ) in fitted_models_and_predictions_dictionary.items():
        predictions_df = transactions_df
        predictions_df["predictions"] = model_and_predictions["predictions_" + type_set]
        performances_model = performance_assessment(
            predictions_df,
            output_feature="TX_FRAUD",
            prediction_feature="predictions",
            top_k_list=top_k_list,
        )
        performances_model.index = [classifier_name]
        performances = performances.append(performances_model)

    return performances


def execution_times_model_collection(fitted_models_and_predictions_dictionary):

    execution_times = pd.DataFrame()

    for (
        classifier_name,
        model_and_predictions,
    ) in fitted_models_and_predictions_dictionary.items():

        execution_times_model = pd.DataFrame()
        execution_times_model["Training execution time"] = [
            model_and_predictions["training_execution_time"]
        ]
        execution_times_model["Prediction execution time"] = [
            model_and_predictions["prediction_execution_time"]
        ]
        execution_times_model.index = [classifier_name]

        execution_times = execution_times.append(execution_times_model)

    return execution_times


def main():
    # Logistic Regression Model Selection

    DIR_INPUT = Path(__file__).parent / "simulated-data-transformed" / "data"

    BEGIN_DATE = "2018-06-11"
    END_DATE = "2018-09-14"

    print("Load  files")
    transactions_df = read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)

    print(
        "{0} transactions loaded, containing {1} fraudulent transactions".format(
            len(transactions_df), transactions_df.TX_FRAUD.sum()
        )
    )

    output_feature = "TX_FRAUD"

    input_features = [
        "TX_AMOUNT",
        "TX_DURING_WEEKEND",
        "TX_DURING_NIGHT",
        "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
        "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",
        "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
        "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",
        "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
        "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",
        "TERMINAL_ID_NB_TX_1DAY_WINDOW",
        "TERMINAL_ID_RISK_1DAY_WINDOW",
        "TERMINAL_ID_NB_TX_7DAY_WINDOW",
        "TERMINAL_ID_RISK_7DAY_WINDOW",
        "TERMINAL_ID_NB_TX_30DAY_WINDOW",
        "TERMINAL_ID_RISK_30DAY_WINDOW",
    ]

    n_folds = 4

    start_date_training = datetime.datetime.strptime("2018-07-25", "%Y-%m-%d")
    delta_train = delta_delay = delta_test = delta_valid = delta_assessment = 7

    start_date_training_for_valid = start_date_training + datetime.timedelta(
        days=-(delta_delay + delta_valid)
    )
    start_date_training_for_test = start_date_training + datetime.timedelta(
        days=(n_folds - 1) * delta_test
    )

    transactions_df_scorer = transactions_df[
        ["CUSTOMER_ID", "TX_FRAUD", "TX_TIME_DAYS"]
    ]

    card_precision_top_100 = metrics.make_scorer(
        card_precision_top_k_custom,
        needs_proba=True,
        top_k=100,
        transactions_df=transactions_df_scorer,
    )

    performance_metrics_list_grid = [
        "roc_auc",
        "average_precision",
        "card_precision@100",
    ]
    performance_metrics_list = ["AUC ROC", "Average precision", "Card Precision@100"]

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "card_precision@100": card_precision_top_100,
    }

    classifier = linear_model.LogisticRegression()

    parameters = {"clf__C": [0.1, 1, 10, 100], "clf__random_state": [0]}

    start_time = time.time()

    performances_df = model_selection_wrapper(
        transactions_df,
        classifier,
        input_features,
        output_feature,
        parameters,
        scoring,
        start_date_training_for_valid,
        start_date_training_for_test,
        n_folds=n_folds,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=-1,
    )

    execution_time_lr = time.time() - start_time

    parameters_dict = dict(performances_df["Parameters"])
    performances_df["Parameters summary"] = [
        parameters_dict[i]["clf__C"] for i in range(len(parameters_dict))
    ]

    print(performances_df)
    print(execution_time_lr)
    print("Logistic Regression Training Done")


if __name__ == "__main__":
    main()
