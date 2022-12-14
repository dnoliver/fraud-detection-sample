"""
Module: Utils
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, pipeline, preprocessing


def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    files = [
        os.path.join(DIR_INPUT, f)
        for f in os.listdir(DIR_INPUT)
        if f >= BEGIN_DATE + ".pkl" and f <= END_DATE + ".pkl"
    ]
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    df_final = df_final.sort_values("TRANSACTION_ID")
    df_final.reset_index(drop=True, inplace=True)
    #  Note: -1 are missing values for real world data
    df_final = df_final.replace([-1], 0)

    return df_final


def scaleData(train, test, features):
    scaler = preprocessing.StandardScaler()
    scaler.fit(train[features])
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])

    return (train, test)


def get_train_test_set(
    transactions_df, start_date_training, delta_train=7, delta_delay=7, delta_test=7
):

    # Get the training set data
    train_df = transactions_df[
        (transactions_df.TX_DATETIME >= start_date_training)
        & (
            transactions_df.TX_DATETIME
            < start_date_training + datetime.timedelta(days=delta_train)
        )
    ]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Then, for each day of the test set
    for day in range(delta_test):

        # Get test data for that day
        test_df_day = transactions_df[
            transactions_df.TX_TIME_DAYS
            == start_tx_time_days_training + delta_train + delta_delay + day
        ]

        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = transactions_df[
            transactions_df.TX_TIME_DAYS
            == start_tx_time_days_training + delta_train + day - 1
        ]

        new_defrauded_customers = set(
            test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID
        )
        known_defrauded_customers = known_defrauded_customers.union(
            new_defrauded_customers
        )

        test_df_day = test_df_day[
            ~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)
        ]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values("TRANSACTION_ID")
    test_df = test_df.sort_values("TRANSACTION_ID")

    return (train_df, test_df)


def prequentialSplit(
    transactions_df,
    start_date_training,
    n_folds=4,
    delta_train=7,
    delta_delay=7,
    delta_assessment=7,
):

    prequential_split_indices = []

    # For each fold
    for fold in range(n_folds):

        # Shift back start date for training by the fold index times the assessment period (delta_assessment)
        # (See Fig. 5)
        start_date_training_fold = start_date_training - datetime.timedelta(
            days=fold * delta_assessment
        )

        # Get the training and test (assessment) sets
        (train_df, test_df) = get_train_test_set(
            transactions_df,
            start_date_training=start_date_training_fold,
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_test=delta_assessment,
        )

        # Get the indices from the two sets, and add them to the list of prequential splits
        indices_train = list(train_df.index)
        indices_test = list(test_df.index)

        prequential_split_indices.append((indices_train, indices_test))

    return prequential_split_indices


def prequential_grid_search(
    transactions_df,
    classifier,
    input_features,
    output_feature,
    parameters,
    scoring,
    start_date_training,
    n_folds=4,
    expe_type="Test",
    delta_train=7,
    delta_delay=7,
    delta_assessment=7,
    performance_metrics_list_grid=["roc_auc"],
    performance_metrics_list=["AUC ROC"],
    n_jobs=-1,
):

    estimators = [("scaler", preprocessing.StandardScaler()), ("clf", classifier)]
    pipe = pipeline.Pipeline(estimators)

    prequential_split_indices = prequentialSplit(
        transactions_df,
        start_date_training=start_date_training,
        n_folds=n_folds,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
    )

    grid_search = model_selection.GridSearchCV(
        pipe,
        parameters,
        scoring=scoring,
        cv=prequential_split_indices,
        refit=False,
        n_jobs=n_jobs,
        verbose=1,
    )

    X = transactions_df[input_features]
    y = transactions_df[output_feature]

    grid_search.fit(X, y)

    performances_df = pd.DataFrame()

    for i in range(len(performance_metrics_list_grid)):
        performances_df[
            performance_metrics_list[i] + " " + expe_type
        ] = grid_search.cv_results_["mean_test_" + performance_metrics_list_grid[i]]
        performances_df[
            performance_metrics_list[i] + " " + expe_type + " Std"
        ] = grid_search.cv_results_["std_test_" + performance_metrics_list_grid[i]]

    performances_df["Parameters"] = grid_search.cv_results_["params"]
    performances_df["Execution time"] = grid_search.cv_results_["mean_fit_time"]

    return performances_df


def model_selection_wrapper(
    transactions_df,
    classifier,
    input_features,
    output_feature,
    parameters,
    scoring,
    start_date_training_for_valid,
    start_date_training_for_test,
    n_folds=4,
    delta_train=7,
    delta_delay=7,
    delta_assessment=7,
    performance_metrics_list_grid=["roc_auc"],
    performance_metrics_list=["AUC ROC"],
    n_jobs=-1,
):

    # Get performances on the validation set using prequential validation
    performances_df_validation = prequential_grid_search(
        transactions_df,
        classifier,
        input_features,
        output_feature,
        parameters,
        scoring,
        start_date_training=start_date_training_for_valid,
        n_folds=n_folds,
        expe_type="Validation",
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    # Get performances on the test set using prequential validation
    performances_df_test = prequential_grid_search(
        transactions_df,
        classifier,
        input_features,
        output_feature,
        parameters,
        scoring,
        start_date_training=start_date_training_for_test,
        n_folds=n_folds,
        expe_type="Test",
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list,
        n_jobs=n_jobs,
    )

    # Bind the two resulting DataFrames
    performances_df_validation.drop(
        columns=["Parameters", "Execution time"], inplace=True
    )
    performances_df = pd.concat(
        [performances_df_test, performances_df_validation], axis=1
    )

    # And return as a single DataFrame
    return performances_df


def card_precision_top_k_day(df_day, top_k):

    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = (
        df_day.groupby("CUSTOMER_ID")
        .max()
        .sort_values(by="predictions", ascending=False)
        .reset_index(drop=False)
    )

    # Get the top k most suspicious cards
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(
        df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID
    )

    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k

    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):

    # Sort days by increasing order
    list_days = list(predictions_df["TX_TIME_DAYS"].unique())
    list_days.sort()

    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []

    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []

    # For each day, compute precision top k
    for day in list_days:

        df_day = predictions_df[predictions_df["TX_TIME_DAYS"] == day]
        df_day = df_day[["predictions", "CUSTOMER_ID", "TX_FRAUD"]]

        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[
            df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False
        ]

        nb_compromised_cards_per_day.append(
            len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique())
        )

        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(
            df_day, top_k
        )

        card_precision_top_k_per_day_list.append(card_precision_top_k)

        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)

    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return (
        nb_compromised_cards_per_day,
        card_precision_top_k_per_day_list,
        mean_card_precision_top_k,
    )


def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):

    # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
    # (indices of the y_true vector)
    predictions_df = transactions_df.iloc[y_true.index.values].copy()
    predictions_df["predictions"] = y_pred

    # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
    (
        nb_compromised_cards_per_day,
        card_precision_top_k_per_day_list,
        mean_card_precision_top_k,
    ) = card_precision_top_k(predictions_df, top_k)

    # Return the mean_card_precision_top_k
    return mean_card_precision_top_k


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
