import data_manager.loader as loader

PLOTS_PATH = '../plots/'


def find_mobile_computer_factor():
    train = loader.load_training_data()

    train_computer = train.loc[train['mobile'] == 0]
    train_computer = train_computer[['mobile', 'price']]

    train_phone = train.loc[train['mobile'] == 1]
    train_phone = train_phone[['mobile', 'price']]

    comp = train_computer['price'].mean()
    mob = train_phone['price'].mean()
    print('computer:', comp)
    print('mobile:', mob)

    print('mobile x factor = computer')
    print(f'mobile x {comp / mob} = computer')


def plot_dist():
    train = loader.load_training_data()
    date_groups = train.groupby('date')
    for k, v in date_groups:
        print(k, len(v))
