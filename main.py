from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('ggplot')

from consts import BENIGN, TRAIN_USER_COUNT, USER_COUNT, TOTAL_SEGMENTS
from globals import commands, challengeToFill, get_data_features


df = get_data_features()


## Normalize Features:
def normalize(df):
    Min = -1
    Max = 1
    result = df.copy()
    for feature_name in df.columns[1:]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        result[feature_name] = result[feature_name] * (Max - Min) + Min
    return result

normalize_df = normalize(df)
normalize_df.fillna(0, inplace=True)

normalize_df.reset_index(level=0, inplace=True)
normalize_df = normalize_df.set_index(['User', 'Chunk'])


# Correlations between the features
plt.figure(figsize=(15,10))
sns.heatmap(normalize_df.corr())
plt.xticks(rotation=85)
plt.title('Correlations between the features')


# Adding Labels to the dataframe:
if 'label' in normalize_df.columns:
    normalize_df.drop('label', axis=1, inplace=True)

for i in range(BENIGN):
    for user in commands.keys():
        normalize_df.loc[(user, i), 'label'] = 0

challengeToFill = challengeToFill.set_index('id')

for user_i in range(TRAIN_USER_COUNT):
    user = "User%d" % user_i
    normalize_df.loc[user, 'label'] = challengeToFill.loc[user].tolist()


test_idx = pd.isnull(normalize_df['label'])
train_idx = ~test_idx

X_train = normalize_df[train_idx].copy()
y_train = X_train.pop('label')
X_test = normalize_df[test_idx].copy()
y_test = X_test.pop('label')


over_sampler = SMOTE()
X, y_train = over_sampler.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X, columns=X_train.columns)


normalize_df.hist(column='label')
print( 'As can be noticed, the data is strongely inbalanced')

pass

