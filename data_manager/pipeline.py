import os

import numpy as np
import pandas as pd
import requests

import data_manager.query_helpers as query_helpers
import data_manager.utils as utils
from data_manager.loader import load_api_request_parameters, load_test_set_probabilities

# CONSTANTS ###
#
# DATA path
TRAINING_DATA_PATH = '../data/training_set'
TRAINING_BACKUP_PATH = '../data/backup'
DATA_PATH = '../data'

# API path
DOMAIN = "51.91.251.0"
PORT = 3000
HOST = f"http://{DOMAIN}:{PORT}"

# API Key
USER_ID = '1fa7d964-7113-4e2c-90d9-d52097370cef'


#
# CONSTANTS ###


class Pipeline:
    def __init__(self, queries_df=None, prices_df=None):
        self.cities, self.dates, self.languages, self.mobile = load_api_request_parameters()
        self.p_language, self.p_mobile, self.p_date, self.p_city = load_test_set_probabilities()

        # first file logger
        ts = utils.get_timestamp()
        log_file = '../log/pipeline__' + ts + '.log'
        self.logger = utils.setup_logger('pipeline', log_file)

        if queries_df is None:
            self.queries = pd.DataFrame(
                columns=['queryId', 'avatar_name', 'avatar_id', 'language', 'city', 'date', 'mobile'])
        else:
            self.queries = queries_df
        if prices_df is None:
            self.prices = pd.DataFrame(columns=['queryId', 'hotel_id', 'price', 'stock'])
        else:
            self.prices = prices_df

    def save_data(self):
        date = utils.get_date()
        utils.create_directory(TRAINING_DATA_PATH, date)

        ts = utils.get_timestamp()

        filename = 'queries__' + ts + '.csv'
        file_path = os.path.join(TRAINING_DATA_PATH, date, filename)
        self.logger.info('Saving %s', file_path)
        self.queries.to_csv(file_path, index=False)

        filename = 'prices__' + ts + '.csv'
        file_path = os.path.join(TRAINING_DATA_PATH, date, filename)
        self.logger.info('Saving %s', file_path)
        self.prices.to_csv(file_path, index=False)

    def backup(self):

        ts = utils.get_timestamp()

        filename = 'backup_queries__' + ts + '.csv'
        file_path = os.path.join(TRAINING_BACKUP_PATH, filename)
        self.logger.info('Saving %s', file_path)
        self.queries.to_csv(file_path, index=False)

        filename = 'backup_prices__' + ts + '.csv'
        file_path = os.path.join(TRAINING_BACKUP_PATH, filename)
        self.logger.info('Saving %s', file_path)
        self.prices.to_csv(file_path, index=False)

    def parse_response(self, res, params):
        # get prices and request
        prices_resp = res.json()['prices']
        req = res.json()['request']

        query_id = query_helpers.get_last_query_id(self.queries) + 1

        q = {'queryId': int(query_id), 'avatar_name': params['avatar_name'],
             'avatar_id': req['avatar_id'], 'language': params['language'], 'city': params['city'],
             'date': params['date'], 'mobile': params['mobile']}
        self.queries = self.queries.append(q, ignore_index=True)

        for entry in prices_resp:
            pr = {'queryId': int(q['queryId']), 'hotel_id': entry['hotel_id'], 'price': entry['price'],
                  'stock': entry['stock']}
            self.prices = self.prices.append(pr, ignore_index=True)

    def generate_random_queries(self, num_queries):

        count = 0

        while count < num_queries:

            avatar_name, _ = query_helpers.generate_random_avatar()
            self.logger.info('Generated random avatar name: %s', avatar_name)
            params = query_helpers.generate_random_query(avatar_name, self.p_language, self.p_mobile, self.p_date,
                                                         self.p_city)
            self.logger.info('Generated random query: %s', params)
            if query_helpers.is_already_submitted_query(params):
                self.logger.warn('Generated query has already been submitted, need of generating a new one')
                continue

            # get query response
            path = query_helpers.get_path(f'pricing/{USER_ID}')

            self.logger.info('Sending request to API')
            res = requests.get(path, params=params)
            res_status = res.status_code
            # print(res.json())
            self.logger.info('Response status: %s', str(res_status))
            self.logger.info('Response: %s', res.json())

            # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
            # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

            if res_status == 200:
                self.parse_response(res, params)
                count += 1

            else:
                if res_status == 422:
                    self.backup()
                    return
                else:
                    self.backup()
                    return

        self.save_data()
        return

    def generate_bruteforce_queries(self, n, parameter):

        count = 0
        cities, dates, languages, mobile = load_api_request_parameters()

        for _ in range(n):
            loop_list = []
            loop_variable = -1

            params = {
                "language": str(np.random.choice(languages, p=list(self.p_language.values()))),
                "city": str(np.random.choice(cities, p=list(self.p_city.values()))),
                "date": int(np.random.choice(dates, p=list(self.p_date.values()))),
                "mobile": int(np.random.choice(mobile, p=list(self.p_mobile.values()))),
            }

            if parameter == 'language':
                params["language"] = ''
                loop_list = self.languages
                loop_variable = 0

            elif parameter == 'mobile':
                params["mobile"] = -1
                loop_list = self.mobile
                loop_variable = 1

            elif parameter == 'city':
                params["city"] = ''
                loop_list = self.cities
                loop_variable = 2

            for element in loop_list:
                if loop_variable == 0:
                    params['language'] = element
                elif loop_variable == 1:
                    params['mobile'] = element
                elif loop_variable == 2:
                    params['city'] = element

                avatar_name, _ = query_helpers.generate_random_avatar()
                params['avatar_name'] = avatar_name

                # get query response
                path = query_helpers.get_path(f'pricing/{USER_ID}')

                res = requests.get(path, params=params)
                res_status = res.status_code
                # print(res.json())

                # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
                # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

                if res_status == 200:
                    self.parse_response(res, params)
                    count += 1

                else:
                    if res_status == 422:
                        self.backup()
                        return
                    else:
                        self.backup()
                        return

        self.save_data()

    def generate_queries_from_df(self, max_num_queries, df, store_result_df=''):

        filepath = '../results'

        count = 0

        # city, date, language, mobile, submitted

        queries = df.loc[df['submitted'] == 0]
        max_query_index = max(queries.index)

        for index, row in queries.iterrows():
            if count < max_num_queries:
                avatar_name, _ = query_helpers.generate_random_avatar()
                self.logger.info('Generated random avatar name: %s', avatar_name)

                language = row['language']
                city = row['city']
                date = row['date']
                mobile = row['mobile']
                params = {
                    "avatar_name": avatar_name,
                    "language": language,
                    "city": city,
                    "date": date,
                    "mobile": mobile,
                }
                self.logger.info('Generated query: %s', params)

                if query_helpers.is_already_submitted_query(params):
                    self.logger.warn('Generated query has already been submitted, need of generating a new one')
                    continue

                # get query response
                path = query_helpers.get_path(f'pricing/{USER_ID}')

                self.logger.info('Sending request to API')
                res = requests.get(path, params=params)
                res_status = res.status_code
                # print(res.json())
                self.logger.info('Response status: %s', str(res_status))
                self.logger.info('Response: %s', res.json())

                # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
                # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

                if res_status == 200:
                    self.parse_response(res, params)
                    count += 1
                    df.loc[index, 'submitted'] = int(1)

                else:
                    if res_status == 422:
                        self.backup()
                        filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                        df.to_csv(os.path.join(filepath, filename), index=False)
                        return
                    else:
                        self.backup()
                        filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                        df.to_csv(os.path.join(filepath, filename), index=False)
                        return

                if index == max_query_index:
                    self.save_data()

            else:
                self.save_data()
                if store_result_df != '':
                    filename = store_result_df
                else:
                    filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                    filename = os.path.join(filepath, filename)
                df.to_csv(filename, index=False)
                return

    def generate_random_queries_with_user_history(self, num_queries):

        p_user_history = [749/785, 17/785, 15/785, 4/785]

        count = 0
        num_queries_by_avatar = 0
        previous_date = 44

        while count < num_queries:

            if num_queries_by_avatar == 0:
                previous_date = 44
                # in the test set the max number of queries from one avatar is 4, so we generate at most 4 queries per
                # avatar
                avatar_name, _ = query_helpers.generate_random_avatar()
                num_queries_by_avatar = np.random.choice([1, 2, 3, 4], p=p_user_history)
                self.logger.info('Generated random avatar name: %s, that will make %s queries',
                                 avatar_name, str(num_queries_by_avatar))

            params = query_helpers.generate_random_query(avatar_name, self.p_language, self.p_mobile, self.p_date,
                                                         self.p_city)
            if previous_date == 0:
                params['date'] = 0

            if params['date'] > previous_date:
                self.logger.info('Generated query has date %s, which is later than previous date %s',
                                 params['date'], previous_date)
                continue
            else:
                previous_date = params['date']

            self.logger.info('Generated random query: %s', params)

            if query_helpers.is_already_submitted_query(params):
                self.logger.warn('Generated query has already been submitted, need of generating a new one')
                continue

            # get query response
            path = query_helpers.get_path(f'pricing/{USER_ID}')

            self.logger.info('Sending request to API')
            res = requests.get(path, params=params)
            res_status = res.status_code
            # print(res.json())
            self.logger.info('Response status: %s', str(res_status))
            self.logger.info('Response: %s', res.json())

            # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
            # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

            if res_status == 200:
                self.parse_response(res, params)
                count += 1
                num_queries_by_avatar -= 1

            else:
                if res_status == 422:
                    self.backup()
                    return
                else:
                    self.backup()
                    return

        self.save_data()
        return

    def generate_random_queries_with_descending_dates(self, num_queries):

        p_user_history = [400/1000, 300/1000, 200/1000, 50/1000, 25/1000, 25/1000]

        count = 0
        num_queries_by_avatar = 0

        while count < num_queries:

            if num_queries_by_avatar == 0:
                # we generate at most 6 queries per avatar
                avatar_name, _ = query_helpers.generate_random_avatar()
                num_queries_by_avatar = np.random.choice([1, 2, 3, 4, 5, 6], p=p_user_history)
                params = query_helpers.generate_random_query_without_dates(avatar_name, self.p_language, self.p_mobile,
                                                                           self.p_city)
                avatar_dates = query_helpers.generate_random_dates(num_queries_by_avatar, self.p_date)

                self.logger.info('Generated random avatar name: %s, that will make %s queries within dates %s',
                                 avatar_name, str(num_queries_by_avatar), " ".join(str(x) for x in avatar_dates))

            if len(avatar_dates) > 0:
                date = avatar_dates.pop(0)
                params['date'] = date
            else:
                self.logger.info('Generated query has no date')
                continue

            self.logger.info('Generated random query: %s', params)

            if query_helpers.is_already_submitted_query(params):
                self.logger.warn('Generated query has already been submitted, submitting it anyway')
            #     num_queries_by_avatar = 0
            #     continue

            # get query response
            path = query_helpers.get_path(f'pricing/{USER_ID}')

            self.logger.info('Sending request to API')
            res = requests.get(path, params=params)
            res_status = res.status_code
            # print(res.json())
            self.logger.info('Response status: %s', str(res_status))
            self.logger.info('Response: %s', res.json())

            # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
            # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

            if res_status == 200:
                self.parse_response(res, params)
                count += 1
                num_queries_by_avatar -= 1

            else:
                if res_status == 422:
                    self.backup()
                    return
                else:
                    self.backup()
                    return

        self.save_data()
        return

    def generate_random_queries_with_descending_dates_copying_train_set(self, num_queries, df, store_result_df=''):

        filepath = '../results'

        p_user_history = [400 / 1000, 300 / 1000, 200 / 1000, 100 / 1000]
        count = 0
        num_queries_by_avatar = 0

        # city, date, language, mobile, submitted

        queries = df.loc[df['submitted'] == 0]
        max_query_index = max(queries.index)

        for index, row in queries.iterrows():
            if count < num_queries:

                language = row['language']
                city = row['city']
                date = row['date']
                mobile = row['mobile']

                if num_queries_by_avatar == 0:
                    # we generate at most 6 queries per avatar
                    avatar_name, _ = query_helpers.generate_random_avatar()
                    num_queries_by_avatar = np.random.choice([1, 2, 3, 4], p=p_user_history)
                    params = {
                        "avatar_name": avatar_name,
                        "language": language,
                        "city": city,
                        "mobile": mobile,
                    }
                    avatar_dates = query_helpers.generate_random_dates_higher_than(num_queries_by_avatar, date)
                    avatar_dates.append(date)
                    num_queries_by_avatar += 1

                    self.logger.info('Generated random avatar name: %s, that will make %s queries within dates %s',
                                     avatar_name, str(num_queries_by_avatar), " ".join(str(x) for x in avatar_dates))

                if len(avatar_dates) > 0:
                    date = avatar_dates.pop(0)
                    params['date'] = date
                else:
                    self.logger.info('Generated query has no date')
                    continue

                self.logger.info('Generated query: %s', params)

                # get query response
                path = query_helpers.get_path(f'pricing/{USER_ID}')

                self.logger.info('Sending request to API')
                res = requests.get(path, params=params)
                res_status = res.status_code
                # print(res.json())
                self.logger.info('Response status: %s', str(res_status))
                self.logger.info('Response: %s', res.json())

                # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
                # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

                if res_status == 200:
                    self.parse_response(res, params)
                    count += 1
                    num_queries_by_avatar -= 1
                    df.loc[index, 'submitted'] = int(1)

                else:
                    if res_status == 422:
                        self.backup()
                        filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                        df.to_csv(os.path.join(filepath, filename), index=False)
                        return
                    else:
                        self.backup()
                        filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                        df.to_csv(os.path.join(filepath, filename), index=False)
                        return

                if index == max_query_index:
                    self.save_data()

            else:
                self.save_data()
                if store_result_df != '':
                    filename = store_result_df
                else:
                    filename = 'generate_queries_from_df_result__' + utils.get_timestamp()
                    filename = os.path.join(filepath, filename)
                df.to_csv(filename, index=False)
                return

    def generate_random_queries_imitating_types_of_users(self, num_users):

        count = 0
        num_queries_by_avatar = 0

        # cities,languages,dates,mobile
        user_types = pd.read_csv('./types_of_users.csv')

        for _ in range(num_users):
            avatar_name, _ = query_helpers.generate_random_avatar()
            num_cities, num_languages, num_dates, num_mobiles = user_types.sample(1, weights="prob")[['cities', 'languages', 'dates', 'mobile']].values[0]
            num_queries_by_avatar = max(num_cities, num_languages, num_dates, num_mobiles)

            cities, dates, languages, mobiles = load_api_request_parameters()
            cities = np.random.choice(cities, p=(list(self.p_city.values())), size=num_cities, replace=False)
            dates = np.random.choice(dates, p=(list(self.p_date.values())), size=num_dates, replace=False)
            languages = np.random.choice(languages, p=(list(self.p_language.values())), size=num_languages, replace=False)
            mobiles = np.random.choice(mobiles, p=(list(self.p_mobile.values())), size=num_mobiles, replace=False)

            queries = query_helpers.generate_user_queries(cities, languages, dates, mobiles, num_queries_by_avatar)
            queries = queries[queries[:, 2].argsort()[::-1]]
            queries = sorted(queries, key=lambda x: int(x[2]), reverse=True)

            self.logger.info('Generated random avatar name: %s, that will make %s queries %s',
                             avatar_name, str(num_queries_by_avatar), queries)

            for query in queries:
                params = {
                    "avatar_name": avatar_name,
                    "city": query[0],
                    "language": query[1],
                    "date": int(query[2]),
                    "mobile": int(query[3]),
                }
                print(params)
                self.logger.info('Generated query: %s', params)

                # get query response
                path = query_helpers.get_path(f'pricing/{USER_ID}')

                self.logger.info('Sending request to API')
                res = requests.get(path, params=params)
                res_status = res.status_code
                # print(res.json())
                self.logger.info('Response status: %s', str(res_status))
                self.logger.info('Response: %s', res.json())

                # r.status_code = 422, 'A pricing request for this avatar already exists for a sooner date'
                # r.status_code = ???, 'Too many requests in the past week. You've done 1000 requests, while the limit is 1000.'

                if res_status == 200:
                    self.parse_response(res, params)
                    count += 1
                    num_queries_by_avatar -= 1

                else:
                    if res_status == 422:
                        self.backup()
                        return
                    else:
                        self.backup()
                        return

        self.save_data()
