import datetime
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


from xgboost import XGBClassifier
from globals import get_now_string

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

plt.style.use('ggplot')


import preprocessing

from consts import *
from globals import commands, commands_only_impersonators_use, impersonators_commands_set


pd.options.display.max_rows = 101
pd.options.display.max_columns = 101



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

if ENABLE_COMMON_COMMANDS:
    for user in commands.keys():
        for i, chunk in enumerate(commands[user]):
            for com in common_commands:
                df.loc[(user, i), com] = chunk.count(com)

df.fillna(0, inplace=True)  # in case a command did not appear in the chunk, the cell will contain null, fill these nulls with zeros
df.head()


### Common Commands per user:

avg_diffrent_commands_per_user = pd.Series()
if ENABLE_COMMON_COMMANDS_PER_USER:
    for user in commands.keys():
        sum_diffrent_commands_per_user = 0
        for i, chunk in enumerate(commands[user]):
            sum_diffrent_commands_per_user = sum_diffrent_commands_per_user + len(set(commands[user][i]))

        avg_diffrent_commands_per_user[user] = float(sum_diffrent_commands_per_user) / len(commands[user])

    for user in commands.keys():
        for i, chunk in enumerate(commands[user]):
            # if some chunk has too high coverage
            df.loc[(user, i), 'common_commands_per_user'] = len(set(commands[user][i])) / avg_diffrent_commands_per_user[user]
            # if some chunk has too low coverage # this doesn't help
            #df.loc[(user, i), 'common_commands_per_user_reverse'] = avg_diffrent_commands_per_user[user] / len(set(commands[user][i]))


### Usage of unknown commands:
#### Number of appearances of unknown commands which where not appear in the benign chunks

distinct_known_commands = set()
for user in commands.keys():
    for chunk in commands[user][:BENIGN]:
        for command in chunk:
            distinct_known_commands.add(command)

# new feature - number of new commands in a chunk (new = did not appear in the first 50 chunks)
if ENABLE_NEW_COMMANDS:
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
if ENABLE_IMPERSONATOR_COMMANDS:
    for user in commands.keys():
        for i, chunk in enumerate(commands[user]):
            df.loc[(user, i), 'impersonator_unique_commands'] = len(set(chunk) & commands_only_impersonators_use)
            df.loc[(user, i), 'impersonator_commands'] = len(set(chunk) & impersonators_commands_set)



# Duplicate commands
# Number of appearances of series of duplicate commands (for certain lengthes)
# Based on the hypothesis that benign use is not characterized by long serieses of duplicate commands

if ENABLE_DUPLICATE_COMMANDS:
# look for series of long duplicates of commands:
    for user in commands.keys():
        for i, chunk in enumerate(commands[user]):
            df.loc[(user, i), 'longest_duplicate_series'] = max(sum(1 for i in g) for k,g in groupby(chunk))
            df.loc[(user, i), '>13_duplicates_count'] = sum(sum(1 for i in g)>=13 for k,g in groupby(chunk))
            df.loc[(user, i), '>12_duplicates_count'] = sum(sum(1 for i in g)>=12 for k,g in groupby(chunk))
            df.loc[(user, i), '>11_duplicates_count'] = sum(sum(1 for i in g)>=11 for k,g in groupby(chunk))
            df.loc[(user, i), '>10_duplicates_count'] = sum(sum(1 for i in g)>=10 for k,g in groupby(chunk))
            df.loc[(user, i), '>3_duplicates_count'] = sum(sum(1 for i in g)>=3 for k, g in groupby(chunk))
        #     df.loc[(USER, i), '>9_duplicates_count'] = sum(sum(1 for i in g)>=9 for k,g in groupby(chunk))
        #     df.loc[(USER, i), '>8_duplicates_count'] = sum(sum(1 for i in g)>=8 for k,g in groupby(chunk))
        #     df.loc[(USER, i), '>7_duplicates_count'] = sum(sum(1 for i in g)>=7 for k,g in groupby(chunk))
        #     df.loc[(USER, i), '>6_duplicates_count'] = sum(sum(1 for i in g)>=6 for k,g in groupby(chunk))
        #     df.loc[(USER, i), '>5_duplicates_count'] = sum(sum(1 for i in g)>=5 for k,g in groupby(chunk))
        #     df.loc[(USER, i), '>4_duplicates_count'] = sum(sum(1 for i in g)>=4 for k,g in groupby(chunk))

df.head()



### Repeated Patterns
# Number of different command patterns that appeared at least 5 times (for each lengthes)
# Based on the hypothesis that benign use is not characterized by many constant repeated patterns

min_length = 3
max_length = 7
appearance_threshold = 5

def sub_lists(L, min_length, max_length):
    result_list = []
    for current_list_length in range(min_length, max_length):
        for start_index in range(len(L) - current_list_length + 1):
            result_list.append(L[start_index:start_index+current_list_length])
    return result_list

if ENABLE_REPEATED_PATTERNS:
    for user in commands.keys():
        for i, chunk in enumerate(commands[user]):
            count = {l: 0 for l in range(min_length, max_length)}
            L = chunk

            for sub in sub_lists(L, min_length=min_length, max_length=max_length):
                sub = list(sub)

                occurances = [1 if L[i:i + len(sub)] == sub else 0 for i in
                              range(len(L) - len(sub))]  # mark matches to the pattern

                # Slice the list into slots, equal to the length of the pattern to avoid counting overlapping patters
                occurances = sum(1 for i in range(0, len(occurances), len(sub)) if sum(occurances[i:i + len(sub)]) > 0)

                if occurances > appearance_threshold:
                    count[len(sub)] += 1  # occurances

            for length, count_l in count.items():
                df.loc[(user, i), 'Repeated_Patterns_%d' % length] = count_l

    df.fillna(0, inplace=True)

    # de-correlate the features, remove the patterns of length 4 from the count of patterns of length 3, etc.
    df['Repeated_Patterns_3'] = df['Repeated_Patterns_3'] - df['Repeated_Patterns_4']
    df['Repeated_Patterns_4'] = df['Repeated_Patterns_4'] - df['Repeated_Patterns_5']
    df['Repeated_Patterns_5'] = df['Repeated_Patterns_5'] - df['Repeated_Patterns_6']
    del df['Repeated_Patterns_4']
    del df['Repeated_Patterns_6']


df.to_csv('data/df_backup_{}.csv'.format(get_now_string()), index=True)
