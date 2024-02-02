import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from data_manager.loader import load_full_feature_set


def load_data():
    X_train, X_test = load_full_feature_set()

    y_train = X_train.pop('price')
    # y_train = X_train[['price']].values
    # y_train = y_train.ravel()

    # X_train = X_train.drop(columns=['hotel_id', 'price'])
    # X_test = X_test.drop(columns=['hotel_id'])

    X_train = pd.get_dummies(X_train, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                               'children_policy'])
    X_test = pd.get_dummies(X_test, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                             'children_policy'])

    return X_train, y_train, X_test


def grid_search(X_train, y_train, X_test):
    model = CatBoostRegressor(task_type="GPU", devices='0:1', verbose=100)

    params = {
        'loss_function': ['RMSE'],
        'iterations': [1000, 1250, 1500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l2_leaf_reg': [1, 2, 3, 4, 5, 7, 9],
        'bagging_temperature': [1, 5, 10],
        'sampling_frequency': ['PerTree', 'PerTreeLevel'],
        'depth': [4, 6, 8, 10, 15],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'min_data_in_leaf': [1, 5, 10],
        'max_leaves': [25, 30, 35],
        'score_function': ['Cosine', 'L2'],
    }

    grid = {'learning_rate': [0.03, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]}

    grid_search_result = model.grid_search(params,
                                           X=X_train,
                                           y=y_train,
                                           verbose=100)

    print(model.best_score_)

    predictions = model.predict(X_test)

    return predictions


def main():
    X_train, y_train, X_test = load_data()
    test_idxs = X_test.pop('index')

    print(np.all(X_train.columns == X_test.columns))

    predictions = grid_search(X_train, y_train, X_test)

    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    submission = submission.sort_values(by=['index'])

    filename = '../results/catboost_grid_search.csv'
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
