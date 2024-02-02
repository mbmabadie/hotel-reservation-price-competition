import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import StackingRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


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
    queries = pd.read_csv('/kaggle/input/defi-ia/all_queries.csv')
    prices = pd.read_csv('/kaggle/input/defi-ia/all_prices.csv')
    hotels = pd.read_csv('/kaggle/input/defi-ia/features_hotels.csv')
    test = pd.read_csv('/kaggle/input/defi-ia/test_set.csv')

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
    ### X_TEST ###

    X_train = pd.get_dummies(X_train, columns=categories)
    X_test = pd.get_dummies(X_test, columns=categories)

    return X_train, X_test


def load_data():
    X_train, X_test = load_full_feature_set()
    y_train = X_train.pop('price')

    return X_train, y_train, X_test


def cv_rmse(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return rmse


kfolds = KFold(n_splits=10, shuffle=True)

X_train, y_train, X_test = load_data()
test_idxs = X_test.pop('index')

print(f'X_train columns == X_test columns: {np.all(X_train.columns == X_test.columns)}')
print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of features: {X_train.shape[1]}')

alphas_alt = [10, 15]
alphas2 = [0.0001, 0.1, 1]
e_alphas = [0.0001, 0.1, 1]
e_l1ratio = [0.8, 0.9, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=10000, alphas=alphas2, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=10000, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

xgboost = XGBRegressor(objective='reg:linear', learning_rate=0.7, max_depth=10, n_estimators=1500)
lightgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.5, n_estimators=3000)
catboost = CatBoostRegressor(task_type="GPU", devices='0:1', verbose=1000, depth=15, l2_leaf_reg=1, iterations=1500,
                             learning_rate=0.2)

stack_gen = StackingRegressor(
        estimators=[ridge, lasso, elasticnet, xgboost, lightgbm],
        final_estimator=catboost,
        passthrough=True
    )

# Using various prediction models that we just created
score = cv_rmse(ridge, X_train, y_train)
print("RIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = cv_rmse(lasso, X_train, y_train)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet, X_train, y_train)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm, X_train, y_train)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost, X_train, y_train)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

print('START Fit')
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X_train, y_train)

print('Lasso')
lasso_model_full_data = lasso.fit(X_train, y_train)

print('Ridge')
ridge_model_full_data = ridge.fit(X_train, y_train)

print('xgboost')
xgb_model_full_data = xgboost.fit(X_train, y_train)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X_train, y_train)

print('catboost')
catboost_full_data = catboost.fit(X_train, y_train)

predictions = ((0.1 * elastic_model_full_data.predict(X_test)) +
               (0.05 * lasso_model_full_data.predict(X_test)) +
               (0.05 * ridge_model_full_data.predict(X_test)) +
               (0.2 * xgb_model_full_data.predict(X_test)) +
               (0.1 * lgb_model_full_data.predict(X_test)) +
               (0.2 * catboost_full_data.predict(X_test)) +
               (0.3 * stack_gen_model.predict(np.array(X_test))))

submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
submission = submission.sort_values(by=['index'])

filename = './stacked_regression_6_models.csv'
submission.to_csv(filename, index=False)
