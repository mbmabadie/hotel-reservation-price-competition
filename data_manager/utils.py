import json
import logging
import os
import time

import pandas as pd

from data_manager import query_helpers

# logging
LOG_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')


def get_timestamp():
    ts = time.gmtime()
    return str(time.strftime("%d_%m_%Y__%H_%M_%S", ts))


def get_date():
    ts = time.gmtime()
    return str(time.strftime("%d_%m_%Y", ts))


def create_directory(parent_path, dir_name):
    dirs = os.listdir(parent_path)
    if dir_name not in dirs:
        path = os.path.join(parent_path, dir_name)
        os.mkdir(path)
    return


def setup_logger(name, log_file, level=logging.INFO):
    """To set up as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(LOG_FORMATTER)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def parse_log_file(log):
    with open(os.path.join('../log', log)) as f:
        f = f.readlines()

    queries = []
    prices = []
    for line in f:
        if 'query' in line:
            try:
                q = line.split('query: ')[1]
            except Exception as e:
                continue
            q = q.strip().replace('\'', '"')
            q = json.loads(q)
            queries.append(q)
        if 'price' in line:
            p = line.split('Response: ')[1]
            p = p.strip().replace('\'', '"')
            p = json.loads(p)
            prices.append(p)

    queries_df = pd.read_csv('../data/all_queries.csv')
    prices_df = pd.read_csv('../data/all_prices.csv')

    log_queries = pd.DataFrame(columns=queries_df.columns)
    log_prices = pd.DataFrame(columns=prices_df.columns)

    query_id = query_helpers.get_last_query_id(queries_df)

    for _q, _p in zip(queries, prices):
        query_id += 1
        prices_resp = _p['prices']
        req = _p['request']

        q = {'queryId': int(query_id), 'avatar_name': _q['avatar_name'],
             'avatar_id': req['avatar_id'], 'language': _q['language'], 'city': _q['city'],
             'date': _q['date'], 'mobile': _q['mobile']}
        queries_df = queries_df.append(q, ignore_index=True)
        log_queries = log_queries.append(q, ignore_index=True)

        for entry in prices_resp:
            pr = {'queryId': int(q['queryId']), 'hotel_id': entry['hotel_id'], 'price': entry['price'],
                  'stock': entry['stock']}
            prices_df = prices_df.append(pr, ignore_index=True)
            log_prices = log_prices.append(pr, ignore_index=True)

    filepath = '../data/backup'
    filename = os.path.join(filepath, 'backup_queries.csv')
    queries_df.to_csv(filename, index=False)
    filename = os.path.join(filepath, 'backup_prices.csv')
    prices_df.to_csv(filename, index=False)

    log_name = log.split('pipeline')[1]
    log_queries_name = 'queries' + log_name.split('.')[0] + '.csv'
    log_prices_name = 'prices' + log_name.split('.')[0] + '.csv'

    filename = os.path.join(filepath, log_queries_name)
    log_queries.to_csv(filename, index=False)

    filename = os.path.join(filepath, log_prices_name)
    log_prices.to_csv(filename, index=False)
