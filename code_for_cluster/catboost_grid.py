import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats


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


def encode(df):
    # Nominal categories
    features_nom = ['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool', 'children_policy', 'Cluster']
    for name in features_nom:
        df[name] = df[name].astype("category")
    return df


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


def mathematical_transforms(df):
    X = pd.DataFrame()
    X["DateMulStock"] = df.date * df.stock
    return X


def cluster_labels(df, features, n_clusters=5):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = pd.get_dummies(X_scaled)
    X_scaled = MinMaxScaler().fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new


def apply_pca(X):
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

    data = pd.get_dummies(data, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                         'children_policy', 'Cluster'])
    scores = data.apply(stats.zscore)

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


def load_full_feature_set(feature_eng=False, perform_PCA=False, remove_outliers=False):
    # load data
    queries = pd.read_csv('./all_queries.csv')
    prices = pd.read_csv('./all_prices.csv')
    hotels = pd.read_csv('./features_hotels.csv')
    test = pd.read_csv('./test_set.csv')

    # drop query duplicates
    queries = queries.drop_duplicates(subset=['language', 'city', 'date', 'mobile'])

    # user history encoding pt.1
    queries['user_history'] = queries.groupby('avatar_id').cumcount()

    # merge queries, prices and hotel_features
    X_train = pd.merge(queries, prices, how='inner', on='queryId')
    X_train = pd.merge(X_train, hotels, how='inner', on='hotel_id')
    X_train = X_train.drop(columns='city_y')
    X_train = X_train.rename(columns={'city_x': 'city'})

    # brand and group correction
    X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # user history encoding
    X_train = X_train.drop(columns=['queryId', 'avatar_id', 'avatar_name'])

    # feature ordering to match test set
    X_train = X_train[['city', 'date', 'language', 'mobile', 'user_history',
                       'hotel_id', 'stock', 'group', 'brand', 'parking', 'pool',
                       'children_policy', 'price']]

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
                     'hotel_id', 'stock', 'group', 'brand', 'parking', 'pool',
                     'children_policy']]

    if feature_eng:
        train = X_train.copy()
        test = X_test.copy()

        train.pop('hotel_id')
        test.pop('hotel_id')

        # create new features with pandas
        train = train.join(mathematical_transforms(train))
        test = test.join(mathematical_transforms(test))

        # create new features with KMeans
        cluster_features = [
            "language",
            "city",
            "stock",
            "group",
            "brand",
        ]

        train = train.join(cluster_labels(train, cluster_features, 5))
        test = test.join(cluster_labels(test, cluster_features, 5))

        if perform_PCA:
            # create new features with PCA
            pca_features = [
                "date",
                "stock",
                "group",
                "brand",
            ]

            train = train.join(pca_components(train, pca_features))
            test = test.join(pca_components(test, pca_features))

        # remove uninformative features
        mi_scores = make_mi_scores(train, train['price'])
        uninformative = get_uninformative(mi_scores, only_PCA=True)
        train = train.drop(columns=uninformative)
        test = test.drop(columns=uninformative)

        # remove outliers
        if remove_outliers:
            without_outliers_indexes = outlier_detection(train)
            train = train.iloc[without_outliers_indexes, :]

        return train, test

    else:
        return X_train, X_test


def load_data(feature_eng=False, perform_PCA=False, remove_outliers=False):
    X_train, X_test = load_full_feature_set(feature_eng, perform_PCA, remove_outliers)
    y_train = X_train.pop('price')

    X_train = pd.get_dummies(X_train, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                               'children_policy', 'Cluster'])
    X_test = pd.get_dummies(X_test, columns=['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool',
                                             'children_policy', 'Cluster'])

    return X_train, y_train, X_test


def grid_search(X_train, y_train, X_test):
    model = CatBoostRegressor(task_type="GPU", devices='0:1', verbose=100)

    grid = {
        'iterations': [1000, 1250, 1500, 2000],
        'learning_rate': [0.1, 0.2],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    params = {
        'iterations': [950, 1000, 1250, 1500, 2000],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'l2_leaf_reg': [1, 2, 3, 4, 5, 7, 9],
        'bagging_temperature': [1, 5, 10],
        'sampling_frequency': ['PerTree', 'PerTreeLevel'],
        'depth': [4, 5, 6, 8, 10],
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'min_data_in_leaf': [1, 5, 10],
        'max_leaves': [25, 30, 35],
        'score_function': ['Cosine', 'L2'],
    }

    grid_search_result = model.grid_search(grid, X=X_train, y=y_train, verbose=100)
    # grid_search_result = model.randomized_search(params, X=X_train, y=y_train, n_iter=10, verbose=100)

    print(model.best_score_)

    preds = model.predict(X_test)

    return preds


X_train, y_train, X_test = load_data(True, True, False)
test_idxs = X_test.pop('index')

print(f'X_train columns == X_test columns: {np.all(X_train.columns == X_test.columns)}')
print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of features: {X_train.shape[1]}')

predictions = grid_search(X_train, y_train, X_test)

submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
submission = submission.sort_values(by=['index'])

filename = './catboost_grid_search.csv'
submission.to_csv(filename, index=False)
