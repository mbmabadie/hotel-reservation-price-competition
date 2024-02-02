import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import tensorflow as tf


def get_user_history(df):
    gr = df.groupby(['order_requests', 'avatar_id'])

    new_df = pd.DataFrame()
    for _, v in gr:
        new_df = new_df.append(v.head(1)[['order_requests', 'avatar_id']])
    new_df['user_history'] = new_df.groupby('avatar_id').cumcount()

    df = pd.merge(df, new_df, how='inner', on=['order_requests', 'avatar_id'])

    return df


def get_user_history_by_city(df):
    gr = df.groupby(['avatar_id', 'city'])
    df['user_history_by_city'] = gr.cumcount()
    return df


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


def load_full_feature_set():
    # load data
    queries = pd.read_csv('../data/all_queries.csv')
    prices = pd.read_csv('../data/all_prices.csv')
    hotels = pd.read_csv('../data/features_hotels.csv')
    test = pd.read_csv('../data/test_set.csv')

    # drop query duplicates
    # queries = queries.drop_duplicates(subset=['language', 'city', 'date', 'mobile'])
    queries = queries.rename(columns={'queryId': 'order_requests'})
    prices = prices.rename(columns={'queryId': 'order_requests'})
    queries = get_user_history(queries)
    # queries = get_user_history_by_city(queries)
    queries = queries.loc[queries['user_history'] <= 3]

    ### X_TRAIN ###
    # merge queries, prices and hotel_features
    X_train = pd.merge(queries, prices, how='inner', on='order_requests')
    X_train = pd.merge(X_train, hotels, how='inner', on='hotel_id')
    X_train = X_train.drop(columns='city_y')
    X_train = X_train.rename(columns={'city_x': 'city'})

    # brand and group correction
    X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # encode as categorical
    categories = ['city', 'language', 'mobile', 'group', 'brand', 'parking', 'pool', 'children_policy']

    # X_train = X_train.drop(columns=['order_requests', 'avatar_id', 'avatar_name'])
    X_train = X_train.drop(columns=['avatar_name'])

    # feature ordering to match test set
    X_train = X_train[['order_requests', 'avatar_id', 'city', 'language', 'date', 'mobile',
                       'user_history',
                       'stock', 'group', 'brand', 'parking', 'pool', 'hotel_id',
                       'children_policy', 'price']]
    ### X_TRAIN ###

    ### X_TEST ###
    # merge test_set with hotel_features
    test = get_user_history(test)
    # test = get_user_history_by_city(test)
    X_test = pd.merge(test, hotels, how='inner', on='hotel_id')
    X_test = X_test.drop(columns='city_y')
    X_test = X_test.rename(columns={'city_x': 'city'})

    # brand and group correction
    X_test['brand'] = X_test.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_test['group'] = X_test.apply(lambda x: map_hotel_group(x['group']), axis=1)

    # X_test = X_test.drop(columns=['order_requests', 'avatar_id'])

    X_test = X_test[['index', 'order_requests', 'avatar_id', 'city', 'language', 'date', 'mobile',
                     'user_history',
                     'stock', 'group', 'brand', 'parking', 'pool', 'hotel_id',
                     'children_policy']]
    ### X_TEST ###

    return X_train, X_test


X_train, X_test = load_full_feature_set()

y_train = X_train.pop('price')
test_idxs = X_test.pop('index')

X_train = X_train.set_index(['order_requests', 'avatar_id', 'hotel_id'])
# X_train['date'] = pd.Categorical(X_train['date'])
X_train['city'] = pd.Categorical(X_train['city'])
X_train['language'] = pd.Categorical(X_train['language'])
X_train['mobile'] = pd.Categorical(X_train['mobile'])
X_train['group'] = pd.Categorical(X_train['group'])
X_train['brand'] = pd.Categorical(X_train['brand'])
X_train['parking'] = pd.Categorical(X_train['parking'])
X_train['pool'] = pd.Categorical(X_train['pool'])
X_train['children_policy'] = pd.Categorical(X_train['children_policy'])

X_train.pop('mobile')

X_test = X_test.set_index(['order_requests', 'avatar_id', 'hotel_id'])
# test['date'] = pd.Categorical(test['date'])
X_test['city'] = pd.Categorical(X_test['city'])
X_test['language'] = pd.Categorical(X_test['language'])
X_test['mobile'] = pd.Categorical(X_test['mobile'])
X_test['group'] = pd.Categorical(X_test['group'])
X_test['brand'] = pd.Categorical(X_test['brand'])
X_test['parking'] = pd.Categorical(X_test['parking'])
X_test['pool'] = pd.Categorical(X_test['pool'])
X_test['children_policy'] = pd.Categorical(X_test['children_policy'])

X_test.pop('mobile')

# categories = ['user_history', 'user_history_by_city', 'city', 'language', 'group', 'brand', 'parking', 'pool', 'children_policy']
categories = ['city', 'language', 'group', 'brand', 'parking', 'pool', 'children_policy']
X_train = pd.get_dummies(X_train, columns=categories)
X_test = pd.get_dummies(X_test, columns=categories)

X_train[['date', 'stock']] = MinMaxScaler().fit_transform(X_train[['date', 'stock']])
X_test[['date', 'stock']] = MinMaxScaler().fit_transform(X_test[['date', 'stock']])

print(f'X_train columns == X_test columns: {np.all(X_train.columns == X_test.columns)}')
print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of features: {X_train.shape[1]}')

model = Sequential()
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(units=16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1))

model.compile(loss=tf.keras.metrics.mean_squared_error,
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],
              optimizer='adadelta')

model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

predictions = model.predict(X_test)
predictions = predictions.squeeze()

submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
submission = submission.sort_values(by=['index'])

filename = './mlp_test_2.csv'
submission.to_csv(filename, index=False)
