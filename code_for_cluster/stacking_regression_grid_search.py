import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder


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


def load_data():
    queries = pd.read_csv('./all_queries.csv')
    prices = pd.read_csv('./all_prices.csv')
    hotels = pd.read_csv('./features_hotels.csv')
    test = pd.read_csv('./test_set.csv')

    queries = queries.drop_duplicates(subset=['language', 'city', 'date', 'mobile'])

    X_train = pd.merge(queries, prices, how='inner', on='queryId')
    X_train = pd.merge(X_train, hotels, how='inner', on='hotel_id')
    X_train = X_train.drop(columns='city_y')
    X_train = X_train.rename(columns={'city_x': 'city'})
    X_train['avatar_id'] = LabelEncoder().fit_transform(X_train['avatar_id'])
    X_train = X_train.rename(columns={'queryId': 'order_requests'})
    X_train.pop('avatar_name')
    X_train['brand'] = X_train.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_train['group'] = X_train.apply(lambda x: map_hotel_group(x['group']), axis=1)
    X_train = X_train[['order_requests', 'city', 'date', 'language', 'mobile',
                       'avatar_id', 'hotel_id', 'stock', 'group', 'brand', 'parking', 'pool',
                       'children_policy', 'price']]

    X_test = pd.merge(test, hotels, how='inner', on='hotel_id')
    X_test = X_test.drop(columns='city_y')
    X_test = X_test.rename(columns={'city_x': 'city'})
    X_test['brand'] = X_test.apply(lambda x: map_hotel_brand(x['brand']), axis=1)
    X_test['group'] = X_test.apply(lambda x: map_hotel_group(x['group']), axis=1)

    return X_train, X_test


def main():

    # load data
    X_train, X_test, = load_data()
    test_idxs = X_test.pop('index')
    y = X_train.pop('price')

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # create grid search
    estimators = [
        ('linear', LinearRegression()),
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('ada', AdaBoostRegressor()),
        ('extra', ExtraTreesRegressor()),
        ('gradboost', GradientBoostingRegressor()),

    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor()
    )

    params = {
        'final_estimator__bootstrap': ['True', 'False'],
        'final_estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'final_estimator__max_features': ['auto', 'sqrt'],
        'final_estimator__min_samples_leaf': [1, 2, 4],
        'final_estimator__min_samples_split': [2, 5, 10],
        'final_estimator__n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        'ridge__alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        'ada__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'ada__loss': ['linear', 'square'],
        'ada__n_estimators': [50, 75, 100, 125, 150, 175, 200, 250, 300],
        'extra__max_depth': [10, 20, 30, 40, 50, None],
        'extra__max_features': ['auto', 'sqrt'],
        'extra__min_samples_leaf': [1, 2, 4],
        'extra__min_samples_split': [2, 5, 10],
        'extra__n_estimators': [50, 75, 100, 125, 150, 175, 200],
        'gradboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'gradboost__max_depth': [4, 6, 8, 10],
        'gradboost__max_features': ['auto', 'sqrt'],
        'gradboost__min_samples_leaf': [3, 5, 7, 9, 12, 15],
        'gradboost__min_samples_split': [3, 5, 7, 9],
        'gradboost__n_estimators': [100, 250, 500, 1000],
        'gradboost__subsample': [0.9, 0.5, 0.2, 0.1],
    }

    # fit estimators
    search = RandomizedSearchCV(estimator=reg, param_distributions=params, n_iter=10, cv=5, verbose=5, n_jobs=-1,
                                scoring='neg_root_mean_squared_error', refit=True)

    # search = GridSearchCV(estimator=reg, param_grid=params, cv=5, verbose=5, n_jobs=-1,
    #                       scoring='neg_root_mean_squared_error', refit=True)

    search.fit(X_train, y)

    print()
    print('best params:', search.best_params_)
    print('best score:', search.best_score_)

    predictions = search.predict(X_test)

    # make predictions on test set
    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    submission = submission.sort_values(by=['index'])
    print(submission)

    # store submission
    filename = './result_stacking_regressor_cluster.csv'
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
