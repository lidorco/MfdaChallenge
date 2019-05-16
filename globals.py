import datetime
import os
import pandas as pd
import joblib

from consts import *

## Load Datasets

partial_labels = pd.read_csv('data/partial_labels.csv')
partial_labels.head()

#  This cell loads the data for each user to dictionaries
#  each entry of the dictionary will be a list of the chunks of each user.
def split_to_chunks(user_commands_list):
    if len(user_commands_list) != 15000:
        raise Exception("missing commands!")

    chunks = []
    for i in range(150):
        chunks.append(user_commands_list[i*100 : i*100 + 99])

    return chunks


commands = pd.Series()
for user in os.listdir('data/FraudedRawData'):
    if user.startswith('User'):
        with open(os.path.join('data/FraudedRawData',user),'rb') as f:
                commands[user] = split_to_chunks([r.strip() for r in f.readlines()])


train_users = ['User%d' % i for i in range(TRAIN_USER_COUNT)]


# load Masqueraders
masqueraders_commands_per_user = [[], [], [], [], [], [], [], [], [], []] # list(map(lambda _: list(), range(10)))
for i in range(150):
    index = str(i*100) + '-' + str(i*100+100)
    for user_id in range(TRAIN_USER_COUNT):
        if partial_labels[index][user_id] == 1:
            masqueraders_commands_per_user[user_id].extend(commands[user_id][i])


impersonators_commands_set = set()
benign_commands_set = set()


for i, user in enumerate(commands[train_users]):
    user = pd.Series(user)
    impersonators_commands_set = impersonators_commands_set.union(set(masqueraders_commands_per_user[i]))

    for segment in user[:BENIGN]:
        benign_commands_set = benign_commands_set.union(set(segment))

commands_only_impersonators_use = impersonators_commands_set.difference(benign_commands_set)
commands_only_benign_use = benign_commands_set.difference(impersonators_commands_set)



def get_data_features():
    df = pd.DataFrame(columns=['User', 'Chunk']).set_index(['User', 'Chunk'])
    df = df.from_csv(r"data/df_backup.csv")
    return df


def get_classifiers():
    clfs = joblib.load(r"data/clfs.bin")
    return clfs


def get_now_string():
    return "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", "-").rsplit(".")[0]