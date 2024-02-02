import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GridSearchCV


def map_hotel_group(group):
    groups = {'Boss Western': 'Boss_Western_Group', 'Accar Hotels': 'Accar_Hotels', 'Independant': 'Independant_Group',
              'Yin Yang': 'Yin_Yang', 'Chillton Worldwide': 'Chillton_Worldwide',
              'Morriott International': 'Morriott_International'}

    return groups[group]


def map_hotel_brand(brand):
    brands = {'J.Halliday Inn': 'J_Halliday_Inn', 'Marcure': 'Marcure', 'Independant': 'Independant_Brand',
              'Ibas': 'Ibas', 'Safitel': 'Safitel', '8 Premium': '8_Premium', 'Tripletree': 'Tripletree',
              'CourtYord': 'CourtYord', 'Royal Lotus': 'Royal_Lotus', 'Boss Western': 'Boss_Western_Brand',
              'Corlton': 'Corlton', 'Navatel': 'Navatel', 'Ardisson': 'Ardisson', 'Morriot': 'Morriot',
              'Chill Garden Inn': 'Chill_Garden_Inn', 'Quadrupletree': 'Quadrupletree'}

    return brands[brand]


def get_user_history(df):
    gr = df.groupby(['order_requests', 'avatar_id'])

    new_df = pd.DataFrame()
    for _, v in gr:
        new_df = new_df.append(v.head(1)[['order_requests', 'avatar_id']])
    new_df['user_history'] = new_df.groupby('avatar_id').cumcount()

    df = pd.merge(df, new_df, how='inner', on=['order_requests', 'avatar_id'])
    df = df.sort_values(by='index')

    return df


def load_full_feature_set():
    # load data
    data_path = '../data/'
    queries = pd.read_csv(data_path + 'all_queries.csv')
    prices = pd.read_csv(data_path + 'all_prices.csv')
    hotels = pd.read_csv(data_path + 'features_hotels.csv')
    test = pd.read_csv(data_path + 'test_set.csv')

    # drop query duplicates
    queries = queries.drop_duplicates(subset=['language', 'city', 'date', 'mobile'])

    # user history encoding pt.1
    queries['user_history'] = queries.groupby('avatar_id').cumcount()

    ### X_TRAIN ###
    # merge queries, prices and hotel_features
    X_train = pd.merge(queries, prices, how='inner', on='queryId')
    X_train = pd.merge(X_train, hotels, how='inner', on='hotel_id')
    X_train = X_train.drop(columns='city_y')
    X_train = X_train.rename(columns={'city_x': 'city'})

    # brand and group correction
    X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # encode as categorical
    # categories = ['city', 'language', 'mobile', 'hotel_id', 'group', 'brand', 'parking', 'pool', 'children_policy']
    categories = ['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool', 'children_policy']

    # user history encoding
    X_train = X_train.drop(columns=['queryId', 'avatar_id', 'avatar_name'])

    # feature ordering to match test set
    X_train = X_train[['city', 'date', 'language', 'mobile', 'user_history',
                       'stock', 'group', 'brand', 'parking', 'pool',
                       'children_policy', 'price']]

    for cat in categories:
        X_train[cat] = X_train[cat].astype("category")
    ### X_TRAIN ###

    ### X_TEST ###
    # merge test_set with hotel_features
    X_test = pd.merge(test, hotels, how='inner', on='hotel_id')
    X_test = X_test.drop(columns='city_y')
    X_test = X_test.rename(columns={'city_x': 'city'})

    # brand and group correction
    X_test['brand'] = X_test.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_test['group'] = X_test.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # user history encoding
    X_test = get_user_history(X_test)
    X_test = X_test.drop(columns=['order_requests', 'avatar_id'])

    X_test = X_test[['index', 'city', 'date', 'language', 'mobile', 'user_history',
                     'stock', 'group', 'brand', 'parking', 'pool',
                     'children_policy']]

    for cat in categories:
        X_test[cat] = X_test[cat].astype("category")
    ### X_TEST ###

    # X_train = pd.get_dummies(X_train, columns=categories)
    # X_test = pd.get_dummies(X_test, columns=categories)

    return X_train, X_test


def main():
    X_train, X_test = load_full_feature_set()

    y_train = X_train.pop('price')
    test_idxs = X_test.pop('index')

    X_train = X_train[['date', 'stock']]
    X_test = X_test[['date', 'stock']]

    print(f'X_train columns == X_test columns: {np.all(X_train.columns == X_test.columns)}')
    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of features: {X_train.shape[1]}')

    alphas = [1, 2, 5, 10]
    quantiles = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]

    params = {'alpha': alphas, 'quantile': quantiles}

    reg = QuantileRegressor()
    search = GridSearchCV(estimator=reg, param_grid=params, cv=3, verbose=5, scoring='neg_root_mean_squared_error')

    search.fit(X_train, y_train)
    print(search.best_params_)
    print(search.best_score_)

    predictions = search.predict(X_test)

    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    submission = submission.sort_values(by=['index'])

    filename = '../results/quantile_reg.csv'
    submission.to_csv(filename, index=False)
