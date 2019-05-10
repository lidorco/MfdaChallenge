from time import time

import joblib
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
import numpy as np
from functools import partial

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from multiprocessing import Pool

plt.style.use('ggplot')

from consts import BENIGN, TRAIN_USER_COUNT, USER_COUNT, TOTAL_SEGMENTS, COMPUTE_CLASSIFIER
from globals import commands, challengeToFill, get_data_features, get_classifiers, get_now_string


#from build_features import df
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



normalize_df.hist(column='label')
print( 'As can be noticed, the data is strongely inbalanced')



train_idx = normalize_df.loc[normalize_df['label'].notnull()]
test_idx = normalize_df[~normalize_df.index.isin(train_idx.index)]

X_train = train_idx.copy()
y_train = X_train.pop('label')
X_test = test_idx.copy()
y_test = X_test.pop('label')


# More of Data Exploration:
masqueraders = train_idx[y_train==1]
benign = train_idx[y_train==0]

for feature in normalize_df.columns:
    plt.figure()
    plt.title(feature)
    plt.hist([benign[feature],masqueraders[feature]], color=['blue','red'], normed=True)

normalize_df.head()

skb = SelectKBest(k=30)
skb.fit(X_train,y_train)
cols = pd.Series(normalize_df.columns.tolist()[:-1])[skb.get_support()].tolist()
print(cols)
X_train = X_train[cols]
X_test  = X_test[cols]



## Fit classifiers

def get_dataset_for_user(df, user, min_i=0, max_i=50):
    idx = [('User%s'%user, i) for i in range(min_i, max_i)]
    return df.loc[idx].copy()

def get_preds_for_user(df, user, clfs, pairs, min_i=50, max_i=150):
    df_to_pred = get_dataset_for_user(df, user, min_i=min_i, max_i=max_i)
    preds = pd.DataFrame()
    for pair in pairs:
        if pair[0]==user:
            preds[str(pair[1])] = clfs[pair].predict_proba(df_to_pred)[:,0]
    return preds

def fit_classifiers(df, classifier):
    pairs = [(i,j) for i in range(40) for j in range(40) if i!=j]
    clfs_paired = {}
    for pair in pairs:
        i,j = pair

        i_df = get_dataset_for_user(df, i, min_i=0, max_i=BENIGN)
        j_df = get_dataset_for_user(df, j, min_i=0, max_i=BENIGN)

        i_df['label'] = 0
        j_df['label'] = 1

        train_df = i_df.append(j_df)
        y = train_df.pop('label')
        clfs_paired[pair] = classifier()
        clfs_paired[pair].fit(train_df, y)
    return clfs_paired



seed = 40
RandomForestClassifier42 = partial(RandomForestClassifier, n_estimators=30, n_jobs=-1, random_state=seed)
RandomForestClassifier42.__name__ = 'RandomForest'
SVC42 = partial(SVC, probability=True,random_state=seed)
SVC42.__name__ = 'SVC'
GradientBoostingClassifier42 = partial(GradientBoostingClassifier, random_state=seed)
GradientBoostingClassifier42.__name__ = 'GradientBoostingClassifier'
GaussianProcessClassifier42 = partial(GaussianProcessClassifier, random_state=seed)
GaussianProcessClassifier42.__name__ = 'GaussianProcessClassifier'
XGBClassifier42 = partial(XGBClassifier, seed=seed)
XGBClassifier42.__name__ = 'XGBClassifier'
AdaBoostClassifier42 = partial(AdaBoostClassifier, random_state=seed)
AdaBoostClassifier42.__name__ = 'AdaBoostClassifier'
XGBClassifier()

classifiers = [SVC42,
               AdaBoostClassifier42,
               RandomForestClassifier42,
               GradientBoostingClassifier42,
               GaussianProcessClassifier42,
               XGBClassifier]

clfs = dict()

if COMPUTE_CLASSIFIER:
    for classifier in classifiers:
        t1 = time()
        print("Started", classifier.__name__, 'in', t1)
        clfs[classifier.__name__] = fit_classifiers(normalize_df, classifier)
        t2 = time()
        print('Finished', classifier.__name__, 'in', t2, 'Total time', t2 - t1, 'sec.')
    #joblib.dump(clfs, 'data/clfs_{}.bin'.format(get_now_string()))
else:
    clsf = get_classifiers() # TODO: make this work


normalize_df.drop('label', axis=1, inplace=True)

pairs = [(i, j) for i in range(40) for j in range(40) if i != j]
classifications = pd.DataFrame([])
for user in range(10):
    tmp_df = pd.DataFrame([])
    for classifier in clfs:
        tmp_df[classifier] = get_preds_for_user(normalize_df, user, clfs[classifier], pairs, 50, 150).mean(1)
    tmp_df['label'] = challengeToFill.T['User%d' % user].tolist()[50:]
    classifications = classifications.append(tmp_df)


normalize_df.head()

classifications.head()

label = classifications.pop('label')
classifications = 1 - classifications


regressor = Ridge()
regressor.fit(classifications, label)



fig, ax = plt.subplots(figsize=(10,5))
plt.bar(
    np.arange(len(regressor.coef_)),
    regressor.coef_,
    width=0.5
)

for i, v in enumerate(regressor.coef_):
    if v < 0:
        ax.text(i - 0.15, v - 0.05, '%0.3f'%v, fontweight='bold')
    else:
        ax.text(i - 0.15, v + 0.01, '%0.3f'%v, fontweight='bold')


plt.xticks(np.arange(len(regressor.coef_)), classifications.columns)
plt.xticks(rotation=25)
plt.tight_layout()
#plt.savefig('classifier_weights.png', dpi=500)

#### Evaluation of the results


classifications['predicted_label'] = regressor.predict(classifications)
classifications.loc[classifications['predicted_label'] < 0, 'predicted_label'] = 0

classifications['real_label'] = label


fpr, tpr, thresholds = metrics.roc_curve(classifications['real_label'], classifications['predicted_label'])

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.plot([[0,0], [1,1]], '--')
plt.legend(['ROC curve'])
plt.title('AUC=%f'%metrics.auc(fpr, tpr))
#plt.savefig('roc_curve.jpg', dpi=1000)


# Find optimal Threshold (according to the given scoring scheme)
scores = []
for threshold in thresholds:
    misDetection = len(classifications[(classifications['real_label']==1) & (classifications['predicted_label']<threshold)])
    falseAlarm   = len(classifications[(classifications['real_label']==0) & (classifications['predicted_label']>threshold)])
    scores.append(-misDetection*9-falseAlarm)

plt.plot(thresholds, scores)
optimal_theshold = thresholds[scores.index(max(scores))]
plt.axvline(x=optimal_theshold, color='blue', linestyle=':', alpha=0.75)


plt.title('Scores VS thresholds')
plt.xlabel('Threshold')
plt.ylabel('Decreased points')
plt.figure()
#plt.savefig('thresholds_scores.jpg', dpi=1000)


classifications.loc[classifications['predicted_label']>optimal_theshold, 'predicted_label'] = 1
classifications.loc[classifications['predicted_label']<1, 'predicted_label'] = 0

print("False Negatives (masquraders classified as benign):", classifications[(classifications['predicted_label']==0)
                                                                             & (classifications['real_label']==1)].shape[0]/100.0)
print('False Positives (benign classified as masquraders):',classifications[(classifications['predicted_label']==1)
                                                                            & (classifications['real_label']==0)].shape[0]/900.0)

### Creating the submission file:

challengeToFill = pd.read_csv('data/partial_labels.csv').set_index('id')

for user in range(10, 40):
    classifications = pd.DataFrame([])
    for classifier in clfs:
        classifications[classifier] = get_preds_for_user(normalize_df, user, clfs[classifier], pairs, 0, 150).mean(1)

    classifications = 1 - classifications
    preds = regressor.predict(classifications)
    preds[:50] = 0
    preds[preds > optimal_theshold] = 1
    preds[preds < 1] = 0

    challengeToFill.loc['User%d' % user] = preds



submission_df = pd.DataFrame({'id': ["User{}_{}-{}".format(x, y*100, y*100+100) for x in range(10,40) for y in range(50, 150)],
                              'label': [ challengeToFill.loc[user][chunk] for user in challengeToFill.index[10:] for chunk in challengeToFill.columns[50:] ]})

submission_df.to_csv('data/prediction_result.csv', index=False)

pass
