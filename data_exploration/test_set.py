from collections import Counter

import numpy as np
import pandas as pd

import data_manager.loader as loader


def test_set_mobile_replication():

    test = loader.load_test_data()

    # print(test.columns)

    groups = test.groupby(['city', 'language', 'date', 'mobile'])

    new_test = pd.DataFrame(columns=['index', 'order_requests', 'city', 'date', 'language', 'mobile',
                                     'avatar_id', 'hotel_id', 'stock', 'group', 'brand', 'parking', 'pool',
                                     'children_policy'])
    counts = []
    for k, v in groups:
        if len(v) == 1:
            new_test = new_test.append(v)
        counts.append(len(v))

    # print(new_test.columns)
    new_test = new_test.drop(columns=['index', 'order_requests', 'avatar_id', 'hotel_id', 'stock', 'group', 'brand',
                                      'parking', 'pool', 'children_policy'])
    new_test = new_test.drop_duplicates()
    new_test['new_mobile'] = np.where(new_test['mobile'] == 0, 1, 0)
    new_test['submitted'] = np.zeros(len(new_test), dtype=int)

    queries = pd.read_csv('../data/all_queries.csv')
    sub_queries = queries[['language', 'city', 'date', 'mobile']]

    for i, row in new_test.iterrows():
        if len(sub_queries.loc[(sub_queries['language'] == row['language']) &
                               (sub_queries['city'] == row['city']) &
                               (sub_queries['date'] == row['date']) &
                               (sub_queries['mobile'] == row['new_mobile'])]) > 0:
            new_test.loc[i, 'submitted'] = int(1)

    print(new_test)
    new_test.to_csv('../data/test_set_mobile_replicate.csv', index=False)

    print(Counter(new_test['submitted']))


def test_set_date_replication():

    test = loader.load_test_data()

    # print(test.columns)

    groups = test.groupby(['city', 'language', 'date', 'mobile'])

    new_test = pd.DataFrame(columns=['city', 'language', 'date', 'mobile'])

    for k, _ in groups:
        city, language, date, mobile = k
        if date > 0:
            new_test = new_test.append({'city': city, 'language': language, 'date': date-1, 'mobile': mobile}, ignore_index=True)
        new_test = new_test.append({'city': city, 'language': language, 'date': date+1, 'mobile': mobile}, ignore_index=True)

    new_test = new_test.drop_duplicates()
    new_test['submitted'] = np.zeros(len(new_test), dtype=int)

    queries = pd.read_csv('../data/all_queries.csv')
    sub_queries = queries[['city', 'language', 'date', 'mobile']]

    for i, row in new_test.iterrows():
        if len(sub_queries.loc[(sub_queries['language'] == row['language']) &
                               (sub_queries['city'] == row['city']) &
                               (sub_queries['date'] == row['date']) &
                               (sub_queries['mobile'] == row['mobile'])]) > 0:
            new_test.loc[i, 'submitted'] = int(1)

    print(new_test)
    new_test.to_csv('../data/test_set_date_replicate.csv', index=False)

    print(Counter(new_test['submitted']))
