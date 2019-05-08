from itertools import groupby
from functools import partial, reduce
from time import time
import os

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.svm import SVC


from sklearn.ensemble import IsolationForest

from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


import preprocessing


pd.options.display.max_rows = 101
pd.options.display.max_columns = 101


###  Constants

BENIGN = 50
TOTAL_SEGMENTS = 150
USER = '0'
USER_COUNT = 40
TRAIN_USER_COUNT = 10



## Load Datasets

challengeToFill = pd.read_csv('data/partial_labels.csv')
challengeToFill.head()

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
masqueraders_commands_per_user = [[], [], [], [], [], [], [], [], [], []]
for i in range(150):
    index = str(i*100) + '-' + str(i*100+100)
    for user_id in range(TRAIN_USER_COUNT):
        if challengeToFill[index][user_id] == 1:
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



## Start Preprocessing

# Create empty DataFrame with "user" and "chunk number" as indices.
df = pd.DataFrame(columns=['User','Chunk']).set_index(['User','Chunk'])

### Common Commands:
#### The number of appearances of the 50 most common commands among the benign segments (of all users)

known_commands_list = []
for user_commands in commands:
    for user_command_segment in user_commands:
        known_commands_list.extend(user_command_segment)

known_commands = pd.Series(known_commands_list)
known_commands.value_counts().hist()
plt.show() #show the histogram


common_commands = known_commands.value_counts().nlargest(50).index.tolist()

for user in commands.keys(): #comment to make run faster with less featurers
    for i, chunk in enumerate(commands[user]):
        for com in common_commands:
            df.loc[(user, i), com] = chunk.count(com)

df.fillna(0, inplace=True)  # in case a command did not appear in the chunk, the cell will contain null, fill these nulls with zeros
df.head()



### Usage of unknown commands:
#### Number of appearances of unknown commands which where not appear in the benign chunks

distinct_known_commands = set()
for user in commands.keys():
    for chunk in commands[user][:BENIGN]:
        for command in chunk:
            distinct_known_commands.add(command)

# new feature - number of new commands in a chunk (new = did not appear in the first 50 chunks)
for user in commands.keys():
    for i, chunk in enumerate(commands[user]):
        df.loc[(user, i), 'new_commands'] = len(set(chunk) - distinct_known_commands)
        try:
            df.loc[(user, i), 'new_commands_usage_count'] = reduce((lambda x,y:x+y),[chunk.count(new_command)for new_command in (set(chunk) - distinct_known_commands)] )
        except TypeError :
            df.loc[(user, i), 'new_commands_usage_count'] = 0

df.loc[df['new_commands']>0,'new_commands'].hist()


# Impersonators' commands
# Number of appearances of commands used by impersonators

for user in commands.keys():
    for i, chunk in enumerate(commands[user]):
        df.loc[(user, i), 'impersonator_unique_commands'] = len(set(chunk) & commands_only_impersonators_use)
        df.loc[(user, i), 'impersonator_commands'] = len(set(chunk) & impersonators_commands_set)

pass
# Duplicate commands
# Number of appearances of series of duplicate commands (for certain lengthes)
# Based on the hypothesis that benign use is not characterized by long serieses of duplicate commands

# look for series of long duplicates of commands:
for user in commands.keys():
    for i,chunk in enumerate(commands[user]):
        # based on https://stackoverflow.com/questions/16733236/find-the-length-of-the-longest-consecutive-series-of-numbers?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        df.loc[(user, i), 'longest_duplicate_series'] = max(sum(1 for i in g) for k,g in groupby(chunk))
        df.loc[(user, i), '>13_duplicates'] = sum(sum(1 for i in g)>=13 for k,g in groupby(chunk))
        df.loc[(user, i), '>12_duplicates'] = sum(sum(1 for i in g)>=12 for k,g in groupby(chunk))
        df.loc[(user, i), '>11_duplicates'] = sum(sum(1 for i in g)>=11 for k,g in groupby(chunk))
        df.loc[(user, i), '>10_duplicates'] = sum(sum(1 for i in g)>=10 for k,g in groupby(chunk))
        df.loc[(user, i), '>3_duplicates'] = sum(sum(1 for i in g)>=3 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>9_duplicates'] = sum(sum(1 for i in g)>=9 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>8_duplicates'] = sum(sum(1 for i in g)>=8 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>7_duplicates'] = sum(sum(1 for i in g)>=7 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>6_duplicates'] = sum(sum(1 for i in g)>=6 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>5_duplicates'] = sum(sum(1 for i in g)>=5 for k,g in groupby(chunk))
    #     df.loc[(USER, i), '>4_duplicates'] = sum(sum(1 for i in g)>=4 for k,g in groupby(chunk))

df.head()