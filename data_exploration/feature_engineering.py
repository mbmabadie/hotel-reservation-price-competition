from sklearn.ensemble import RandomForestRegressor

from itertools import combinations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

import pandas as pd
import numpy as np
import scipy.stats as stats

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

PLOTS_PATH = '../plots/'


def encode(df):
    # Nominal categories
    features_nom = ['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool', 'children_policy', 'Cluster']
    for name in features_nom:
        df[name] = df[name].astype("category")
    return df


def score_dataset(X, y, model=XGBRegressor()):
    """
    Baseline score to judge our feature engineering against.
    The function will compute the cross-validated RMSLE score for a feature set with XGBoost as the model.
    It can be reused anytime to try out a new feature set
    :param X:
    :param y:
    :param model:
    :return: score
    """
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge.

    # for colname in X.select_dtypes(["category"]):
    #     X[colname] = X[colname].cat.codes

    X = X.copy()

    if 'index' in list(X.columns):
        print('POP INDEX')
        X.pop('index')
    if 'price' in list(X.columns):
        print('POP PRICE')
        X.pop('price')

    X = pd.get_dummies(X)

    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_root_mean_squared_error", verbose=2, n_jobs=-1
    )
    score = -1 * score.mean()
    return score


def make_mi_scores(X, y):
    X = X.copy()

    X = encode(X)

    if 'index' in list(X.columns):
        print('POP INDEX')
        X.pop('index')
    if 'price' in list(X.columns):
        print('POP PRICE')
        X.pop('price')

    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]

    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.01]


def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X


def interactions(df):
    X = pd.get_dummies(df.city, prefix='city')
    X = X.mul(df.stock, axis=0)
    return X


def mathematical_transforms(df):
    X = pd.DataFrame()
    X["DateMulStock"] = df.date * df.stock
    return X


def group_transforms(df):
    X = pd.DataFrame()
    X["MedianBrandStock"] = df.groupby("brand")["stock"].transform("median")
    X["MedianGroupStock"] = df.groupby("group")["stock"].transform("median")
    X["MedianCityStock"] = df.groupby("city")["stock"].transform("median")
    X["MedianLanguageStock"] = df.groupby("language")["stock"].transform("median")
    X["NormalizedBrandStock"] = df.groupby("brand")["stock"].transform(lambda x: (x - x.mean()) / x.std())
    X["NormalizedGroupStock"] = df.groupby("group")["stock"].transform(lambda x: (x - x.mean()) / x.std())
    X["NormalizedCityStock"] = df.groupby("city")["stock"].transform(lambda x: (x - x.mean()) / x.std())
    X["NormalizedLanguageStock"] = df.groupby("language")["stock"].transform(lambda x: (x - x.mean()) / x.std())
    return X


def group_encode_categories(df):
    X = pd.DataFrame()
    gr = df.groupby(['parking', 'pool', 'children_policy'])
    X['HotelTypeStock'] = gr['stock'].transform('median')

    gr_keys = list(gr.groups.keys())
    hotel_types = dict(zip(gr_keys, range(len(gr_keys))))

    X['HotelType'] = df.apply(lambda x: map_hotel_type(x['parking'], x['pool'], x['children_policy'], hotel_types),
                              axis=1)

    return X


def map_hotel_type(parking, pool, children_policy, hotel_types):
    return hotel_types[(parking, pool, children_policy)]


def cluster_labels(df, features, n_clusters=5):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = pd.get_dummies(X_scaled)
    # X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    X_scaled = MinMaxScaler().fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new


def cluster_distance(df, features, n_clusters=5):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = pd.get_dummies(X_scaled)
    # X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    X_scaled = MinMaxScaler().fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd


def apply_pca(X, standardize=True):
    X = pd.get_dummies(X)
    columns = X.columns
    X = MinMaxScaler().fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


def outlier_detection(data):

    # check_in = ['price', 'date', 'stock']

    data = pd.get_dummies(data, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                         'children_policy', 'Cluster'])
    scores = data.apply(stats.zscore)

    # conditions = []
    # for column in list(data.columns):
    #     if column in check_in:
    #         c = (scores[column] < 1) & (scores[column] > -1)
    #         conditions.append(c)
    #
    # for cond in conditions:
    #     # print(len(scores.loc[cond]))
    #     scores = scores.loc[cond]

    scores = scores.loc[
        (scores['price'] < 1) & (scores['price'] > -1) &
        (scores['date'] < 1) & (scores['date'] > -1) &
        (scores['stock'] < 1) & (scores['stock'] > -1)
    ]

    print(f'number of samples: {len(scores)}')

    return scores.index


def get_uninformative(mi_scores, only_PCA=False):
    uninformative = []
    for i, v in mi_scores.items():
        if v < 0.01 and not only_PCA:
            uninformative.append(i)
        if 'PC' in str(i) and v < 1.0:
            uninformative.append(i)
    return uninformative

#
# def feature_engineering():
#     # load data
#     train = load_training_data()
#     test = load_test_data()
#
#     # train = train.reset_index()
#
#     X_train = train.copy()
#     X_train = X_train.drop(columns=['queryId', 'avatar_name', 'avatar_id', 'hotel_id'])
#     # X = encode(X)
#     # y = X_train.pop("price")
#     y = X_train['price']
#     price = y.copy()
#     price = price.reset_index()
#
#     X_test = test.copy()
#     # X_test = X_test.drop(columns=['order_requests', 'avatar_id', 'hotel_id', 'index'])
#     X_test = X_test.drop(columns=['order_requests', 'avatar_id', 'hotel_id'])
#     X_test = X_test[['index', 'language', 'city', 'date', 'mobile', 'stock', 'group', 'brand', 'parking',
#                      'pool', 'children_policy']]
#
#     # correct group and brand names
#     X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
#     X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)
#     X_test['brand'] = X_test.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
#     X_test['group'] = X_test.apply(lambda x: map_hotel_group(x['group']), axis=1)
#
#     # calculate baseline score
#     baseline_score = score_dataset(X_train, X_train['price'])
#     print(f"Baseline score: {baseline_score:.5f} RMSE")
#
#     # calculate MI scores and drop uninformative features
#     # print(X_train.columns)
#     mi_scores = make_mi_scores(X_train, X_train['price'])
#     print('\nMI Scores')
#     print(mi_scores)
#
#     # X_train = drop_uninformative(X_train, mi_scores)
#     uninformative = get_uninformative(mi_scores)
#     X_train = X_train.drop(columns=uninformative)
#     X_test = X_test.drop(columns=uninformative)
#
#     # create new features with pandas
#     X_train = X_train.join(group_transforms(X_train))
#     X_train = X_train.join(mathematical_transforms(X_train))
#     X_train = X_train.join(group_encode_categories(X_train))
#
#     X_test = X_test.join(group_transforms(X_test))
#     X_test = X_test.join(mathematical_transforms(X_test))
#     X_test = X_test.join(group_encode_categories(X_test))
#
#     cluster_features = [
#         "language",
#         "city",
#         "stock",
#         "group",
#         "brand",
#     ]
#
#     # create new features with KMeans
#     X_train = X_train.join(cluster_labels(X_train, cluster_features, 5))
#     # X_train = X_train.join(cluster_distance(X_train, cluster_features, 5))
#
#     X_test = X_test.join(cluster_labels(X_test, cluster_features, 5))
#     # X_test = X_test.join(cluster_distance(X_test, cluster_features, 5))
#
#     pca_features = [
#         "date",
#         "stock",
#         "group",
#         "brand",
#     ]
#
#     # create new features with PCA
#     X_train = X_train.join(pca_components(X_train, pca_features))
#     X_test = X_test.join(pca_components(X_test, pca_features))
#
#     # remove outliers
#     without_outliers_indexes = outlier_detection(X_train)
#     X_train = X_train.iloc[without_outliers_indexes, :]
#     y = y.iloc[without_outliers_indexes]
#     # price = price.iloc[without_outliers_indexes, :]
#
#     # calculate MI scores and drop uninformative features
#     # print(X_train.columns)
#     mi_scores = make_mi_scores(X_train, X_train['price'].values.ravel())
#     print('\nMI Scores')
#     print(mi_scores)
#     # X_train = drop_uninformative(X_train, mi_scores)
#     # X_test = drop_uninformative(X_test, mi_scores)
#     uninformative = get_uninformative(mi_scores)
#     X_train = X_train.drop(columns=uninformative)
#     X_test = X_test.drop(columns=uninformative)
#
#     # dataset info
#     print(X_train)
#     # print(X_test)
#     # print(X_train.columns)
#     # print(X_test.columns)
#     # print(X_train.columns == X_test.columns)
#     # print(y)
#     # print(price)
#
#     # calculate new score
#     new_score = score_dataset(X_train, X_train['price'])
#     print(f"New score: {new_score:.5f} RMSE")
#
#     # store new dataset
#     X_train.to_csv('../data/new_feature_set/X_train_v3.csv', index=False)
#     X_test.to_csv('../data/new_feature_set/X_test_v3.csv', index=False)
#     y.to_csv('../data/new_feature_set/y_train_v3.csv', index=False)
#
