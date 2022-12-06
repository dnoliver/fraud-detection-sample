"""
Module: Logistic Regression
"""

import datetime
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
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


def get_summary_performances(
    performances_df, parameter_column_name="Parameters summary"
):

    metrics = ["AUC ROC", "Average precision", "Card Precision@100"]
    performances_results = pd.DataFrame(columns=metrics)

    performances_df.reset_index(drop=True, inplace=True)

    best_estimated_parameters = []
    validation_performance = []
    test_performance = []

    for metric in metrics:

        index_best_validation_performance = performances_df.index[
            np.argmax(performances_df[metric + " Validation"].values)
        ]

        best_estimated_parameters.append(
            performances_df[parameter_column_name].iloc[
                index_best_validation_performance
            ]
        )

        validation_performance.append(
            str(
                round(
                    performances_df[metric + " Validation"].iloc[
                        index_best_validation_performance
                    ],
                    3,
                )
            )
            + "+/-"
            + str(
                round(
                    performances_df[metric + " Validation" + " Std"].iloc[
                        index_best_validation_performance
                    ],
                    2,
                )
            )
        )

        test_performance.append(
            str(
                round(
                    performances_df[metric + " Test"].iloc[
                        index_best_validation_performance
                    ],
                    3,
                )
            )
            + "+/-"
            + str(
                round(
                    performances_df[metric + " Test" + " Std"].iloc[
                        index_best_validation_performance
                    ],
                    2,
                )
            )
        )

    performances_results.loc["Best estimated parameters"] = best_estimated_parameters
    performances_results.loc["Validation performance"] = validation_performance
    performances_results.loc["Test performance"] = test_performance

    optimal_test_performance = []
    optimal_parameters = []

    for metric in ["AUC ROC Test", "Average precision Test", "Card Precision@100 Test"]:

        index_optimal_test_performance = performances_df.index[
            np.argmax(performances_df[metric].values)
        ]

        optimal_parameters.append(
            performances_df[parameter_column_name].iloc[index_optimal_test_performance]
        )

        optimal_test_performance.append(
            str(round(performances_df[metric].iloc[index_optimal_test_performance], 3))
            + "+/-"
            + str(
                round(
                    performances_df[metric + " Std"].iloc[
                        index_optimal_test_performance
                    ],
                    2,
                )
            )
        )

    performances_results.loc["Optimal parameter(s)"] = optimal_parameters
    performances_results.loc["Optimal test performance"] = optimal_test_performance

    return performances_results


# Get the performance plot for a single performance metric
def get_performance_plot(
    performances_df,
    ax,
    performance_metric,
    expe_type_list=["Test", "Train"],
    expe_type_color_list=["#008000", "#2F4D7E"],
    parameter_name="Tree maximum depth",
    summary_performances=None,
):

    # expe_type_list is the list of type of experiments, typically containing 'Test', 'Train', or 'Valid'
    # For all types of experiments
    for i in range(len(expe_type_list)):

        # Column in performances_df for which to retrieve the data
        performance_metric_expe_type = performance_metric + " " + expe_type_list[i]

        # Plot data on graph
        ax.plot(
            performances_df["Parameters summary"],
            performances_df[performance_metric_expe_type],
            color=expe_type_color_list[i],
            label=expe_type_list[i],
        )

        # If performances_df contains confidence intervals, add them to the graph
        if performance_metric_expe_type + " Std" in performances_df.columns:

            conf_min = (
                performances_df[performance_metric_expe_type]
                - 2 * performances_df[performance_metric_expe_type + " Std"]
            )
            conf_max = (
                performances_df[performance_metric_expe_type]
                + 2 * performances_df[performance_metric_expe_type + " Std"]
            )

            ax.fill_between(
                performances_df["Parameters summary"],
                conf_min,
                conf_max,
                color=expe_type_color_list[i],
                alpha=0.1,
            )

    # If summary_performances table is present, adds vertical dashed bar for best estimated parameter
    if summary_performances is not None:
        best_estimated_parameter = summary_performances[performance_metric][
            ["Best estimated parameters"]
        ].values[0]
        best_estimated_performance = float(
            summary_performances[performance_metric][["Validation performance"]]
            .values[0]
            .split("+/-")[0]
        )
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            best_estimated_parameter,
            ymin,
            best_estimated_performance,
            linestyles="dashed",
        )

    # Set title, and x and y axes labels
    ax.set_title(performance_metric + "\n", fontsize=14)
    ax.set(xlabel=parameter_name, ylabel=performance_metric)


# Get the performance plots for a set of performance metric
def get_performances_plots(
    performances_df,
    performance_metrics_list=["AUC ROC", "Average precision", "Card Precision@100"],
    expe_type_list=["Test", "Train"],
    expe_type_color_list=["#008000", "#2F4D7E"],
    parameter_name="Tree maximum depth",
    summary_performances=None,
):

    # Create as many graphs as there are performance metrics to display
    n_performance_metrics = len(performance_metrics_list)
    fig, ax = plt.subplots(
        1, n_performance_metrics, figsize=(5 * n_performance_metrics, 4)
    )

    # Plot performance metric for each metric in performance_metrics_list
    for i in range(n_performance_metrics):

        get_performance_plot(
            performances_df,
            ax[i],
            performance_metric=performance_metrics_list[i],
            expe_type_list=expe_type_list,
            expe_type_color_list=expe_type_color_list,
            parameter_name=parameter_name,
            summary_performances=summary_performances,
        )

    ax[n_performance_metrics - 1].legend(
        loc="upper left",
        labels=expe_type_list,
        bbox_to_anchor=(1.05, 1),
        title="Type set",
    )

    plt.subplots_adjust(wspace=0.5, hspace=0.8)


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

    performances_df_lr = performances_df
    summary_performances_lr = get_summary_performances(
        performances_df_lr, parameter_column_name="Parameters summary"
    )

    get_performances_plots(
        performances_df_lr,
        performance_metrics_list=["AUC ROC", "Average precision", "Card Precision@100"],
        expe_type_list=["Test", "Validation"],
        expe_type_color_list=["#008000", "#FF0000"],
        parameter_name="Regularization value",
        summary_performances=summary_performances_lr,
    )

    performances_df_dictionary = {"Logistic Regression": performances_df_lr}

    execution_times = [execution_time_lr]

    DIR_OUTPUT = (
        Path(__file__).parent / "performances_model_selection_logistic_regression.pkl"
    )
    filehandler = open(DIR_OUTPUT, "wb")
    pickle.dump((performances_df_dictionary, execution_times), filehandler)
    filehandler.close()

    print("Logistic Regression Training Done")

    return (performances_df_lr, execution_times, summary_performances_lr)


if __name__ == "__main__":
    main()
