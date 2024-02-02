import pandas as pd
from sklearn import preprocessing
from data_exploration.feature_engineering import mathematical_transforms, cluster_labels, pca_components, \
    make_mi_scores, get_uninformative, outlier_detection


def load_api_request_parameters():
    """
    this function loads API request parameters options
    :return: cities, dates, languages, mobile
    """
    cities = ['amsterdam', 'copenhagen', 'madrid', 'paris', 'rome', 'sofia', 'valletta', 'vienna', 'vilnius']
    dates = list(range(45))
    languages = ['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian',
                 'finnish', 'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian',
                 'luxembourgish', 'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish',
                 'swedish']
    mobile = [0, 1]

    return cities, dates, languages, mobile


def load_test_set_probabilities():
    """
    this function loads the probabilities that should be applied to each feature when generating a query based on the
    test set distributions (counts)
    :return: p_language, p_mobile, p_date, p_city
    """
    p_language = {}
    p_mobile = {}
    p_date = {}
    p_city = {}

    probs = pd.read_csv('./queries_probs.csv', index_col=0)

    _c = probs.loc[probs['feature'] == 'city']
    _l = probs.loc[probs['feature'] == 'language']
    _d = probs.loc[probs['feature'] == 'date']
    _m = probs.loc[probs['feature'] == 'mobile']

    for el in _c.values:
        p_city[str(el[1])] = float(el[2])

    for el in _l.values:
        p_language[str(el[1])] = float(el[2])

    for el in _d.values:
        p_date[str(el[1])] = float(el[2])

    for el in _m.values:
        p_mobile[str(el[1])] = float(el[2])

    return p_language, p_mobile, p_date, p_city


def load_training_data(duplicates=False):
    """
    this function loads the joined training data from queries.csv and prices.csv
    :arg: duplicates (True, False): whether to load or not duplicate queries based on (language, city, date, mobile)
    :return: training set dataframe
    """
    queries = pd.read_csv('../data/all_queries.csv')
    prices = pd.read_csv('../data/all_prices.csv')
    hotels = pd.read_csv('../data/features_hotels.csv')

    if not duplicates:
        queries = queries.drop_duplicates(subset=['language', 'city', 'date', 'mobile'])
    result = pd.merge(queries, prices, how='inner', on='queryId')
    result = pd.merge(result, hotels, how='inner', on='hotel_id')
    result = result.drop(columns='city_y')
    result = result.rename(columns={'city_x': 'city'})

    return result


def load_test_data():
    """
    this function loads the testing data
    :return: testing set dataframe
    """
    test = pd.read_csv('../data/test_set.csv')
    hotels = pd.read_csv('../data/features_hotels.csv')

    result = pd.merge(test, hotels, how='inner', on='hotel_id')
    result = result.drop(columns='city_y')
    result = result.rename(columns={'city_x': 'city'})

    return result


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


def load_new_feature_set():
    test = pd.read_csv('../data/test_set.csv')

    X_train = pd.read_csv('../data/new_feature_set/X_train_v3.csv')
    X_test = pd.read_csv('../data/new_feature_set/X_test_v3.csv')
    y_train = pd.read_csv('../data/new_feature_set/y_train_v3.csv')

    # X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    # X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)
    # X_test['brand'] = X_test.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    # X_test['group'] = X_test.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # Nominal categories
    features_nom = ['language', 'city', 'group', 'brand', 'parking', 'pool', 'children_policy']
    for name in features_nom:
        X_train[name] = X_train[name].astype("category")
        X_test[name] = X_test[name].astype("category")

    return X_train, X_test, y_train


def get_user_history(df):
    gr = df.groupby(['order_requests', 'avatar_id'])

    new_df = pd.DataFrame()
    for _, v in gr:
        new_df = new_df.append(v.head(1)[['order_requests', 'avatar_id']])
    new_df['user_history'] = new_df.groupby('avatar_id').cumcount()

    df = pd.merge(df, new_df, how='inner', on=['order_requests', 'avatar_id'])
    df = df.sort_values(by='index')

    return df


def load_full_feature_set(feature_eng=False, remove_outliers=False):
    # load data
    queries = pd.read_csv('../data/all_queries.csv')
    prices = pd.read_csv('../data/all_prices.csv')
    hotels = pd.read_csv('../data/features_hotels.csv')
    test = pd.read_csv('../data/test_set.csv')

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
    # X_train['avatar_id'] = preprocessing.LabelEncoder().fit_transform(X_train['avatar_id'])
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
    # print(X_test.iloc[:50, [0,1,6,7,14]])
    # print(X_test)

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
        print(mi_scores)
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
