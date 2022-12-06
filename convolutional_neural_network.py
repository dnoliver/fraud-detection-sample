"""
Module: Convolutional Neural Network
"""

import datetime
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from skorch import NeuralNetClassifier

from utils import (
    card_precision_top_k_custom,
    get_performances_plots,
    get_summary_performances,
    get_train_test_set,
    model_selection_wrapper,
    read_from_files,
    scaleData,
)


def rolling_window(array, window):
    # pylint: disable=E1136  # pylint/issues/3139
    a = np.concatenate([np.ones((window - 1,)) * -1, array])
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides).astype(int)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class FraudConvNet(torch.nn.Module):
    def __init__(
        self,
        num_features,
        seq_len,
        hidden_size=100,
        conv1_params=(100, 2),
        conv2_params=None,
        max_pooling=True,
    ):

        super(FraudConvNet, self).__init__()

        # parameters
        self.num_features = num_features
        self.hidden_size = hidden_size

        # representation learning part
        self.conv1_num_filters = conv1_params[0]
        self.conv1_filter_size = conv1_params[1]
        self.padding1 = torch.nn.ConstantPad1d((self.conv1_filter_size - 1, 0), 0)
        self.conv1 = torch.nn.Conv1d(
            num_features, self.conv1_num_filters, self.conv1_filter_size
        )
        self.representation_size = self.conv1_num_filters

        self.conv2_params = conv2_params
        if conv2_params:
            self.conv2_num_filters = conv2_params[0]
            self.conv2_filter_size = conv2_params[1]
            self.padding2 = torch.nn.ConstantPad1d((self.conv2_filter_size - 1, 0), 0)
            self.conv2 = torch.nn.Conv1d(
                self.conv1_num_filters, self.conv2_num_filters, self.conv2_filter_size
            )
            self.representation_size = self.conv2_num_filters

        self.max_pooling = max_pooling
        if max_pooling:
            self.pooling = torch.nn.MaxPool1d(seq_len)
        else:
            self.representation_size = self.representation_size * seq_len

        # feed forward part at the end
        self.flatten = torch.nn.Flatten()

        # representation to hidden
        self.fc1 = torch.nn.Linear(self.representation_size, self.hidden_size)
        self.relu = torch.nn.ReLU()

        # hidden to output
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        representation = self.conv1(self.padding1(x))

        if self.conv2_params:
            representation = self.conv2(self.padding2(representation))

        if self.max_pooling:
            representation = self.pooling(representation)

        representation = self.flatten(representation)

        hidden = self.fc1(representation)
        relu = self.relu(hidden)

        output = self.fc2(relu)
        output = self.sigmoid(output)

        return output


def prepare_generators(training_set, valid_set, batch_size=64):

    train_loader_params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    valid_loader_params = {"batch_size": batch_size, "num_workers": 0}

    training_generator = torch.utils.data.DataLoader(
        training_set, **train_loader_params
    )
    valid_generator = torch.utils.data.DataLoader(valid_set, **valid_loader_params)

    return training_generator, valid_generator


class EarlyStopping:
    def __init__(self, patience=2, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf

    def continue_training(self, current_score):
        if self.best_score > current_score:
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print("New best score:", current_score)
        else:
            self.counter += 1
            if self.verbose:
                print(self.counter, " iterations since best score.")

        return self.counter <= self.patience


def evaluate_model(model, generator, criterion):
    model.eval()
    batch_losses = []
    for x_batch, y_batch in generator:
        # Forward pass
        y_pred = model(x_batch)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_batch)
        batch_losses.append(loss.item())
    mean_loss = np.mean(batch_losses)
    return mean_loss


def training_loop(
    model,
    training_generator,
    valid_generator,
    optimizer,
    criterion,
    max_epochs=100,
    apply_early_stopping=True,
    patience=2,
    verbose=False,
):
    # Setting the model in training mode
    model.train()

    if apply_early_stopping:
        early_stopping = EarlyStopping(verbose=verbose, patience=patience)

    all_train_losses = []
    all_valid_losses = []

    # Training loop
    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        train_loss = []
        for x_batch, y_batch in training_generator:
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(x_batch)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_batch)
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # showing last training loss after each epoch
        all_train_losses.append(np.mean(train_loss))
        if verbose:
            print("")
            print("Epoch {}: train loss: {}".format(epoch, np.mean(train_loss)))
        # evaluating the model on the test set after each epoch
        valid_loss = evaluate_model(model, valid_generator, criterion)
        all_valid_losses.append(valid_loss)
        if verbose:
            print("valid loss: {}".format(valid_loss))
        if apply_early_stopping:
            if not early_stopping.continue_training(valid_loss):
                if verbose:
                    print("Early stopping")
                break

    training_execution_time = time.time() - start_time
    return model, training_execution_time, all_train_losses, all_valid_losses


def get_all_predictions(model, generator):
    model.eval()
    all_preds = []
    for x_batch, y_batch in generator:
        # Forward pass
        y_pred = model(x_batch)
        # append to all preds
        all_preds.append(y_pred.detach().cpu().numpy())
    return np.vstack(all_preds)


class FraudSequenceDatasetForPipe(torch.utils.data.Dataset):
    def __init__(self, x, y):
        "Initialization"
        seq_len = 5

        # lets us assume that x[:,-1] are the dates, and x[:,-2] are customer ids, padding_mode is "mean"
        customer_ids = x[:, -2]
        dates = x[:, -1]

        # storing the features x in self.feature and adding the "padding" transaction at the end
        self.features = torch.FloatTensor(x[:, :-2])

        self.features = torch.vstack([self.features, self.features.mean(axis=0)])

        self.y = None
        if y is not None:
            self.y = torch.LongTensor(y.values)

        self.customer_ids = customer_ids
        self.dates = dates
        self.seq_len = seq_len

        # ===== computing sequences ids =====

        df_ids_dates_cpy = pd.DataFrame(
            {"CUSTOMER_ID": customer_ids, "TX_DATETIME": dates}
        )

        df_ids_dates_cpy["tmp_index"] = np.arange(len(df_ids_dates_cpy))
        df_groupby_customer_id = df_ids_dates_cpy.groupby("CUSTOMER_ID")
        sequence_indices = pd.DataFrame(
            {
                "tx_{}".format(n): df_groupby_customer_id["tmp_index"].shift(
                    seq_len - n - 1
                )
                for n in range(seq_len)
            }
        )
        self.sequences_ids = sequence_indices.fillna(
            len(self.features) - 1
        ).values.astype(int)

        df_ids_dates_cpy = df_ids_dates_cpy.drop("tmp_index", axis=1)

    def __len__(self):
        "Denotes the total number of samples"
        # not len(self.features) because of the added padding transaction
        return len(self.customer_ids)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample index

        tx_ids = self.sequences_ids[index]

        if self.y is not None:
            # transposing because the CNN considers the channel dimension before the sequence dimension
            return self.features[tx_ids, :].transpose(0, 1), self.y[index]
        else:
            return self.features[tx_ids, :].transpose(0, 1), -1


class FraudCNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        seq_len=5,
        hidden_size=100,
        num_filters=100,
        filter_size=2,
        num_conv=1,
        max_pooling=True,
        p=0,
    ):
        super(FraudCNN, self).__init__()
        # parameters
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.p = p

        # representation learning part
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding1 = torch.nn.ConstantPad1d((filter_size - 1, 0), 0)
        self.conv1 = torch.nn.Conv1d(num_features, self.num_filters, self.filter_size)
        self.representation_size = self.num_filters

        self.num_conv = num_conv

        if self.num_conv == 2:
            self.padding2 = torch.nn.ConstantPad1d((filter_size - 1, 0), 0)
            self.conv2 = torch.nn.Conv1d(
                self.num_filters, self.num_filters, self.filter_size
            )
            self.representation_size = self.num_filters

        self.max_pooling = max_pooling
        if max_pooling:
            self.pooling = torch.nn.MaxPool1d(seq_len)
        else:
            self.representation_size = self.representation_size * seq_len

        # feed forward part at the end
        self.flatten = torch.nn.Flatten()

        # representation to hidden
        self.fc1 = torch.nn.Linear(self.representation_size, self.hidden_size)
        self.relu = torch.nn.ReLU()

        # hidden to output
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(self.p)

    def forward(self, x):

        representation = self.conv1(self.padding1(x))
        representation = self.dropout(representation)
        if self.num_conv == 2:
            representation = self.conv2(self.padding2(representation))
            representation = self.dropout(representation)

        if self.max_pooling:
            representation = self.pooling(representation)

        representation = self.flatten(representation)

        hidden = self.fc1(representation)
        relu = self.relu(hidden)
        relu = self.dropout(relu)

        output = self.fc2(relu)
        output = self.softmax(output)

        return output


def main():
    # Convolutional Neural Network Model Selection

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

    start_date_training = datetime.datetime.strptime("2018-07-25", "%Y-%m-%d")
    delta_train = 7
    delta_delay = 7
    delta_test = 7

    delta_valid = delta_test

    start_date_training_with_valid = start_date_training + datetime.timedelta(
        days=-(delta_delay + delta_valid)
    )

    (train_df, valid_df) = get_train_test_set(
        transactions_df,
        start_date_training_with_valid,
        delta_train=delta_train,
        delta_delay=delta_delay,
        delta_test=delta_test,
    )

    (train_df, valid_df) = scaleData(train_df, valid_df, input_features)

    dates = train_df["TX_DATETIME"].values

    customer_ids = train_df["CUSTOMER_ID"].values

    seq_len = 5

    x_train = torch.FloatTensor(train_df[input_features].values)
    x_valid = torch.FloatTensor(valid_df[input_features].values)
    y_train = torch.FloatTensor(train_df[output_feature].values)
    y_valid = torch.FloatTensor(valid_df[output_feature].values)

    df_ids_dates = pd.DataFrame({"CUSTOMER_ID": customer_ids, "TX_DATETIME": dates})

    df_ids_dates["tmp_index"] = np.arange(len(df_ids_dates))

    df_groupby_customer_id = df_ids_dates.groupby("CUSTOMER_ID")

    sequence_indices = pd.DataFrame(
        {
            "tx_{}".format(n): df_groupby_customer_id["tmp_index"].shift(
                seq_len - n - 1
            )
            for n in range(seq_len)
        }
    )

    sequence_indices = sequence_indices.fillna(-1).astype(int)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print("Selected device is", DEVICE)

    transactions_df["TX_DATETIME_TIMESTAMP"] = transactions_df["TX_DATETIME"].apply(
        lambda x: datetime.datetime.timestamp(x)
    )
    input_features_new = input_features + ["CUSTOMER_ID", "TX_DATETIME_TIMESTAMP"]

    # Only keep columns that are needed as argument to custome scoring function
    # to reduce serialisation time of transaction dataset
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

    n_folds = 4
    start_date_training_for_valid = start_date_training + datetime.timedelta(
        days=-(delta_delay + delta_valid)
    )
    start_date_training_for_test = start_date_training + datetime.timedelta(
        days=(n_folds - 1) * delta_test
    )
    delta_assessment = delta_valid

    seed_everything(42)
    classifier = NeuralNetClassifier(
        FraudCNN,
        max_epochs=2,
        lr=0.001,
        optimizer=torch.optim.Adam,
        batch_size=64,
        dataset=FraudSequenceDatasetForPipe,
        iterator_train__shuffle=True,
    )
    classifier.set_params(train_split=False, verbose=0)

    # parameters = {
    #     "clf__lr": [0.0001, 0.0002, 0.001],
    #     "clf__batch_size": [64, 128, 256],
    #     "clf__max_epochs": [10, 20, 40],
    #     "clf__module__hidden_size": [500],
    #     "clf__module__num_conv": [1, 2],
    #     "clf__module__p": [0, 0.2],
    #     "clf__module__num_features": [int(len(input_features))],
    #     "clf__module__num_filters": [100, 200],
    # }

    parameters = {
        "clf__lr": [0.001],
        "clf__batch_size": [64, 128],
        "clf__max_epochs": [20],
        "clf__module__hidden_size": [500],
        "clf__module__num_conv": [2],
        "clf__module__p": [0.2],
        "clf__module__num_features": [int(len(input_features))],
        "clf__module__num_filters": [200],
    }

    start_time = time.time()

    performances_df = model_selection_wrapper(
        transactions_df,
        classifier,
        input_features_new,
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

    execution_time_cnn = time.time() - start_time

    parameters_dict = dict(performances_df["Parameters"])
    performances_df["Parameters summary"] = [
        str(parameters_dict[i]["clf__max_epochs"])
        + "/"
        + str(parameters_dict[i]["clf__module__num_conv"])
        + "/"
        + str(parameters_dict[i]["clf__batch_size"])
        + "/"
        + str(parameters_dict[i]["clf__module__num_filters"])
        + "/"
        + str(parameters_dict[i]["clf__module__p"])
        for i in range(len(parameters_dict))
    ]

    performances_df_cnn = performances_df

    execution_times = [execution_time_cnn]

    summary_performances_cnn = get_summary_performances(
        performances_df_cnn, parameter_column_name="Parameters summary"
    )

    parameters_dict = dict(performances_df_cnn["Parameters"])
    performances_df_cnn["Parameters summary"] = [
        str(parameters_dict[i]["clf__max_epochs"])
        + "/"
        + str(parameters_dict[i]["clf__batch_size"])
        + "/"
        + str(parameters_dict[i]["clf__module__num_filters"])
        for i in range(len(parameters_dict))
    ]

    summary_performances_cnn_subset = get_summary_performances(
        performances_df_cnn, parameter_column_name="Parameters summary"
    )
    indexes_summary = summary_performances_cnn_subset.index.values
    indexes_summary[0] = "Best estimated parameters"
    summary_performances_cnn_subset.rename(
        index=dict(zip(np.arange(len(indexes_summary)), indexes_summary))
    )
    get_performances_plots(
        performances_df_cnn,
        performance_metrics_list=["AUC ROC", "Average precision", "Card Precision@100"],
        expe_type_list=["Test", "Validation"],
        expe_type_color_list=["#008000", "#FF0000"],
        parameter_name="batch size",
        summary_performances=summary_performances_cnn_subset,
    )

    parameters_dict = dict(performances_df_cnn["Parameters"])
    performances_df_cnn["Parameters summary"] = [
        str(parameters_dict[i]["clf__max_epochs"])
        + "/"
        + str(parameters_dict[i]["clf__module__num_conv"])
        + "/"
        + str(parameters_dict[i]["clf__batch_size"])
        + "/"
        + str(parameters_dict[i]["clf__module__num_filters"])
        + "/"
        + str(parameters_dict[i]["clf__module__p"])
        for i in range(len(parameters_dict))
    ]

    performances_df_dictionary = {"Convolutional Neural Network": performances_df_cnn}

    execution_times = [execution_time_cnn]

    DIR_OUTPUT = (
        Path(__file__).parent
        / "performances_model_selection_convolutional_neural_network.pkl"
    )
    filehandler = open(DIR_OUTPUT, "wb")
    pickle.dump((performances_df_dictionary, execution_times), filehandler)
    filehandler.close()

    print("Convolutional Neural Network Done")

    return (performances_df_cnn, execution_times, summary_performances_cnn)


if __name__ == "__main__":
    main()
