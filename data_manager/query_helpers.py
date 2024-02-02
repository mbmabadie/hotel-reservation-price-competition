import os
import random
import string
import urllib.parse

import numpy as np
import pandas as pd
import requests

from data_manager.loader import load_api_request_parameters

# API path
DOMAIN = "51.91.251.0"
PORT = 3000
HOST = f"http://{DOMAIN}:{PORT}"

# API Key
USER_ID = '1fa7d964-7113-4e2c-90d9-d52097370cef'

TRAINING_DATA_PATH = '../data/training_set'


def get_path(x):
    return urllib.parse.urljoin(HOST, x)


def create_new_avatar(avatar_name):
    path = get_path(f'avatars/{USER_ID}/{avatar_name}')
    r = requests.post(path)
    return r


def get_remaining_requests():
    path = get_path(f'remaining-requests/{USER_ID}')
    r = requests.get(path)
    return r.json()


def generate_random_avatar():
    end = False

    while not end:

        result_str = ''.join(random.sample(string.ascii_lowercase, 15))

        response = create_new_avatar(result_str)
        if response.status_code == 200:
            end = True

    return response.json()['name'], response.json()


def generate_random_query(avatar_name='', p_lan=None, p_mob=None, p_date=None, p_city=None):
    cities, dates, languages, mobile = load_api_request_parameters()

    if avatar_name == '':
        avatar_name, _ = generate_random_avatar()

    params = {
        "avatar_name": avatar_name,
        "language": str(np.random.choice(languages, p=list(p_lan.values()))),
        "city": str(np.random.choice(cities, p=list(p_city.values()))),
        "date": int(np.random.choice(dates, p=list(p_date.values()))),
        "mobile": int(np.random.choice(mobile, p=list(p_mob.values()))),
    }

    return params


def generate_random_dates(size, p_date=None):
    _, dates, _, _ = load_api_request_parameters()
    return sorted(list(np.random.choice(dates, size=size, replace=False, p=list(p_date.values()))), reverse=True)


def generate_random_dates_higher_than(size, higher_than):
    # print(f'dates must be higher than {higher_than}')
    _, dates, _, _ = load_api_request_parameters()
    dates_2 = [i for i in dates if i > higher_than]
    # print(dates_2)
    return sorted(list(np.random.choice(dates_2, size=size, replace=False)), reverse=True)


def generate_random_query_without_dates(avatar_name='', p_lan=None, p_mob=None, p_city=None):
    cities, _, languages, mobile = load_api_request_parameters()

    if avatar_name == '':
        avatar_name, _ = generate_random_avatar()

    params = {
        "avatar_name": avatar_name,
        "language": str(np.random.choice(languages, p=list(p_lan.values()))),
        "city": str(np.random.choice(cities, p=list(p_city.values()))),
        "mobile": int(np.random.choice(mobile, p=list(p_mob.values())))
    }

    return params


def generate_user_queries(cities, languages, dates, mobiles, num_queries):
    combs = []
    for c in cities:
        for l in languages:
            for d in dates:
                for m in mobiles:
                    combs.append((c, l, d, m))
    combs = np.array(combs)

    valid_queries = False
    while not valid_queries:
        idxs = np.random.choice(range(len(combs)), replace=False, size=num_queries)
        queries = combs[idxs]
        if len(np.unique(queries[:, 0])) == len(cities) and len(np.unique(queries[:, 1])) == len(languages) and len(
                np.unique(queries[:, 2])) == len(dates) and len(np.unique(queries[:, 3])) == len(mobiles):
            valid_queries = True

    return queries


def merge_all_queries():
    queries = pd.DataFrame(columns=['queryId', 'avatar_name', 'avatar_id', 'language', 'city', 'date', 'mobile'])
    prices = pd.DataFrame(columns=['queryId', 'hotel_id', 'price', 'stock'])

    folders = os.listdir(TRAINING_DATA_PATH)
    for folder in folders:
        files = os.listdir(os.path.join(TRAINING_DATA_PATH, folder))
        if len(files) > 0:
            for file in files:
                path = os.path.join(TRAINING_DATA_PATH, folder, file)
                aux = pd.read_csv(path)
                if 'queries' in file:
                    queries = queries.append(aux)
                else:
                    prices = prices.append(aux)

    queries.to_csv('../data/all_queries.csv', index=False)
    prices.to_csv('../data/all_prices.csv', index=False)


def merge_folder(folder):
    if folder == '' or folder is None:
        return

    queries = pd.DataFrame(columns=['queryId', 'avatar_name', 'avatar_id', 'language', 'city', 'date', 'mobile'])
    prices = pd.DataFrame(columns=['queryId', 'hotel_id', 'price', 'stock'])

    files = os.listdir(os.path.join(TRAINING_DATA_PATH, folder))
    if len(files) > 0:
        for file in files:
            path = os.path.join(TRAINING_DATA_PATH, folder, file)
            aux = pd.read_csv(path)
            if 'queries' in file:
                queries = queries.append(aux)
            else:
                prices = prices.append(aux)

    queries.to_csv(TRAINING_DATA_PATH + '/queries_' + folder + '.csv', index=False)
    prices.to_csv(TRAINING_DATA_PATH + '/prices_' + folder + '.csv', index=False)


def is_already_submitted_query(params):
    queries = pd.read_csv('../data/all_queries.csv')

    sub_queries = queries[['language', 'city', 'date', 'mobile']]

    if len(sub_queries.loc[
               (sub_queries['language'] == params['language']) & (sub_queries['city'] == params['city']) & (
                       sub_queries['date'] == params['date']) & (
                       sub_queries['mobile'] == params['mobile'])]) > 0:
        return True
    else:
        return False


def get_last_query_id(queries_df):
    merge_all_queries()
    queries = pd.read_csv('../data/all_queries.csv')

    if len(queries_df) > 0:
        v1 = max(queries_df.loc[:, 'queryId'])
        v2 = max(queries.loc[:, 'queryId'])
        return max(v1, v2)
    else:
        return max(queries.loc[:, 'queryId'])

# print(get_remaining_requests())
# merge_all_queries()
