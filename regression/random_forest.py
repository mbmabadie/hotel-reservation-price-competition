import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

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


def preprocessing(train, test):
    train['date_stock'] = np.where(train['date'] > 0, train['stock'] / train['date'], 0)
    test['date_stock'] = np.where(test['date'] > 0, test['stock'] / test['date'], 0)

    train['language_swedish_finnish'] = train['language_swedish'] + train['language_finnish']
    train['language_lithuanian_latvian'] = train['language_lithuanian'] + train['language_latvian']
    train['language_luxembourgish_french'] = train['language_luxembourgish'] + train['language_french']
    train['language_slovene_german'] = train['language_slovene'] + train['language_german']
    train['language_slovakian_hungarian'] = train['language_slovakian'] + train['language_hungarian']
    train.drop(columns=['language_swedish', 'language_finnish', 'language_lithuanian', 'language_latvian',
                        'language_luxembourgish', 'language_french', 'language_slovene', 'language_german',
                        'language_slovakian', 'language_hungarian'])

    test['language_swedish_finnish'] = test['language_swedish'] + test['language_finnish']
    test['language_lithuanian_latvian'] = test['language_lithuanian'] + test['language_latvian']
    test['language_luxembourgish_french'] = test['language_luxembourgish'] + test['language_french']
    test['language_slovene_german'] = test['language_slovene'] + test['language_german']
    test['language_slovakian_hungarian'] = test['language_slovakian'] + test['language_hungarian']
    test.drop(columns=['language_swedish', 'language_finnish', 'language_lithuanian', 'language_latvian',
                       'language_luxembourgish', 'language_french', 'language_slovene', 'language_german',
                       'language_slovakian', 'language_hungarian'])

    return train, test


def RMSE(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def grid_search(X_train, y_train, X_test):
    """"
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=1, cv=3, verbose=5,
                                   n_jobs=-1, scoring='neg_root_mean_squared_error')

    rf_random.fit(X_train, y_train)

    print(rf_random.best_params_)

    predictions = rf_random.predict(X_test)

    return predictions


def regression_on_new_feature_set():
    X_train, X_test, y_train = load_new_feature_set()
    test_idxs = X_test.pop('index')
    y_train = X_train.pop('price')

    X_train = pd.get_dummies(X_train)
    print(X_train.shape)
    X_test = pd.get_dummies(X_test)

    rfr = RandomForestRegressor()
    print('FITTING')
    rfr.fit(X_train, y_train)
    print('FINISH FITTING')

    print('PREDICTING')
    predictions = rfr.predict(X_test)
    print('FINISH PREDICTING')

    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    print(submission)
    submission = submission.sort_values(by=['index'])
    print(submission)

    filename = '../results/random_forest_new_features_v2.csv'
    submission.to_csv(filename, index=False)


def main():
    X_train, y, X_test = load_data()
    # X_train, X_test = preprocessing(X_train, X_test)

    print(list(X_train.columns))
    print(list(X_test.columns))
    print(list(X_train.columns) == list(X_test.columns))

    # forest = RandomForestRegressor()
    # forest.fit(X_train, y)

    preds = grid_search(X_train, y, X_test)
    predictions = pd.DataFrame(data=preds, columns=['price'])

    ts = get_timestamp()
    filename = '../results/random_forest__' + ts + '.csv'
    predictions.to_csv(filename, index_label='index')


if __name__ == "__main__":
    main()
