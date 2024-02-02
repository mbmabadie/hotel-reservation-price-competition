import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBRegressor

from data_manager.loader import load_training_data, load_test_data, load_new_feature_set
from data_manager.utils import get_timestamp


def load_data():
    train = load_training_data()

    y = train[['price']].values
    y = y.ravel()

    train = train.drop(columns=['price', 'avatar_name', 'avatar_id', 'queryId', 'hotel_id'])
    # train = train.drop(columns=['price', 'avatar_name', 'avatar_id', 'queryId', 'hotel_id', 'group', 'brand'])
    # train = train.drop(columns=['price', 'avatar_name', 'avatar_id', 'queryId', 'hotel_id', 'group', 'brand',
    #                             'parking', 'pool', 'children_policy'])

    X_train = pd.get_dummies(train)

    test = load_test_data()
    test = test.drop(columns=['index', 'order_requests', 'avatar_id', 'hotel_id'])
    # test = test.drop(columns=['index', 'order_requests', 'avatar_id', 'hotel_id', 'group', 'brand'])
    # test = test.drop(columns=['index', 'order_requests', 'avatar_id', 'hotel_id', 'group', 'brand',
    #                           'parking', 'pool', 'children_policy'])

    test = test[['language', 'city', 'date', 'mobile', 'stock', 'group', 'brand', 'parking', 'pool', 'children_policy']]
    # test = test[['language', 'city', 'date', 'mobile', 'stock']]
    X_test = pd.get_dummies(test)

    return X_train, y, X_test


def RMSE(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def grid_search(X_train, y_train, X_test):
    """"
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """

    learning_rate = (0.05, 0.10, 0.15)
    max_depth = [3, 4, 5, 6, 8]
    min_child_weight = [1, 3, 5, 7]
    gamma = [0.0, 0.1, 0.2]
    colsample_bytree = [0.3, 0.4]

    grid = {'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'colsample_bytree': colsample_bytree}

    reg = XGBRegressor()
    gscv = GridSearchCV(estimator=reg, param_grid=grid, cv=3, verbose=10, n_jobs=-1,
                        scoring='neg_root_mean_squared_error')

    gscv.fit(X_train, y_train)

    print(gscv.best_params_)

    predictions = gscv.predict(X_test)

    return predictions


def regression_on_new_feature_set():
    X_train, X_test, y_train = load_new_feature_set()
    test_idxs = X_test.pop('index')
    y_train = X_train.pop('price')

    X_train = pd.get_dummies(X_train)
    print(X_train.shape)
    X_test = pd.get_dummies(X_test)

    predictions = grid_search(X_train, y_train, X_test)

    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    print(submission)
    submission = submission.sort_values(by=['index'])
    print(submission)

    filename = '../results/xgb_new_features.csv'
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    regression_on_new_feature_set()
