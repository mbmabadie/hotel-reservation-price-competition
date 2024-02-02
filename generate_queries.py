from data_manager.utils import parse_log_file
from data_manager.pipeline import Pipeline
from data_manager.query_helpers import get_remaining_requests, merge_all_queries, merge_folder
import pandas as pd


def main():
    print('remaining requests:', get_remaining_requests())
    merge_all_queries()

    pipe = Pipeline()

    # df = pd.read_csv('../train_queries_to_imitate.csv')
    # print('remaining test set reqs:', len(df.loc[df['submitted'] == 0]))

    num_queries = 15
    pipe.logger.info('Call to generate_random_queries_imitating_types_of_users() with %s as parameter', str(num_queries))
    pipe.generate_random_queries_imitating_types_of_users(num_queries)
    # pipe.generate_bruteforce_queries(num_queries, 'language')

    merge_all_queries()
    print('remaining requests:', get_remaining_requests())

    # parse_log_file('pipeline__26_11_2022__21_28_08.log')

    # merge_folder('06_12_2022')


if __name__ == "__main__":
    main()
