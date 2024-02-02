import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from data_manager.loader import load_new_feature_set, load_full_feature_set


def grid_search_on_full_dataset():
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
    # https://www.analyticsvidhya.com/blog/2020/12/improve-predictive-model-score-stacking-regressor/

    # load data
    X_train, X_test, = load_full_feature_set()
    test_idxs = X_test.pop('index')
    y = X_train.pop('price')

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    print(np.all(X_train.columns == X_test.columns))

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=adaboost#sklearn.ensemble.AdaBoostRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html?highlight=extra+trees
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random+forest

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
        # 'final_estimator__ccp_alpha',
        # 'final_estimator__criterion',
        'final_estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'final_estimator__max_features': ['auto', 'sqrt'],
        # 'final_estimator__max_leaf_nodes',
        # 'final_estimator__max_samples',
        # 'final_estimator__min_impurity_decrease',
        'final_estimator__min_samples_leaf': [1, 2, 4],
        'final_estimator__min_samples_split': [2, 5, 10],
        # 'final_estimator__min_weight_fraction_leaf',
        'final_estimator__n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],
        # 'final_estimator__oob_score',
        # 'final_estimator__warm_start',
        'ridge__alpha': [0.01, 0.1, 1.0, 5.0, 10.0],
        # 'ridge__tol',
        'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        # 'lasso__tol',
        # 'lasso__warm_start',
        # 'ada__base_estimator',
        'ada__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'ada__loss': ['linear', 'square'],
        'ada__n_estimators': [50, 75, 100, 125, 150, 175, 200, 250, 300],
        # 'extra__bootstrap',
        # 'extra__ccp_alpha',
        # 'extra__criterion',
        'extra__max_depth': [10, 20, 30, 40, 50, None],
        'extra__max_features': ['auto', 'sqrt'],
        # 'extra__max_leaf_nodes',
        # 'extra__max_samples',
        # 'extra__min_impurity_decrease',
        'extra__min_samples_leaf': [1, 2, 4],
        'extra__min_samples_split': [2, 5, 10],
        # 'extra__min_weight_fraction_leaf',
        'extra__n_estimators': [50, 75, 100, 125, 150, 175, 200],
        # 'extra__oob_score',
        # 'extra__warm_start',
        # 'gradboost__alpha',
        # 'gradboost__ccp_alpha',
        # 'gradboost__criterion',
        # 'gradboost__init',
        'gradboost__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        # 'gradboost__loss',
        'gradboost__max_depth': [4, 6, 8, 10],
        'gradboost__max_features': ['auto', 'sqrt'],
        # 'gradboost__max_leaf_nodes',
        # 'gradboost__min_impurity_decrease',
        'gradboost__min_samples_leaf': [3, 5, 7, 9, 12, 15],
        'gradboost__min_samples_split': [3, 5, 7, 9],
        # 'gradboost__min_weight_fraction_leaf',
        'gradboost__n_estimators': [100, 250, 500, 1000],
        # 'gradboost__n_iter_no_change',
        'gradboost__subsample': [0.9, 0.5, 0.2, 0.1],
        # 'gradboost__tol',
        # 'gradboost__validation_fraction',
        # 'gradboost__warm_start'
    }

    # fit estimators
    search = RandomizedSearchCV(estimator=reg, param_distributions=params, n_iter=1, cv=3, verbose=5, n_jobs=-1,
                                scoring='neg_root_mean_squared_error', refit=True)

    # search = GridSearchCV(estimator=reg, param_grid=params, cv=3, verbose=5, n_jobs=-1,
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
    filename = '../results/stacking_regressor_full_dataset.csv'
    submission.to_csv(filename, index=False)


def main():
    X, _X_test, _ = load_new_feature_set()
    test_idxs = _X_test.pop('index')
    y = X.pop('price')

    X = pd.get_dummies(X)
    _X_test = pd.get_dummies(_X_test)

    estimators = [
        ('linear', LinearRegression()),
        # ('ridge', RidgeCV()),
        # ('svr', LinearSVR()),
        ('ada', AdaBoostRegressor()),
        ('extra', ExtraTreesRegressor()),
        ('gradboost', GradientBoostingRegressor()),

    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y
    )
    reg.fit(X_train, y_train).score(X_test, y_test)

    predictions = reg.predict(_X_test)

    submission = pd.DataFrame(data={'index': test_idxs, 'price': predictions})
    print(submission)
    submission = submission.sort_values(by=['index'])
    print(submission)

    filename = '../results/stacking_regressor_new_features.csv'
    submission.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
