# -*- coding: utf-8 -*-
import os
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
import pandas as pd
import csv

def ks_score(df):
    pos = df['true_label'].value_counts()[0]
    neg = df['true_label'].value_counts()[1]
    df = df.sort_values(by='prob_neg')
    cur_pos=0;
    cur_neg=0;
    rs = 0
    for index, row in df.iterrows():
        if row['true_label']==1:
            cur_neg = cur_neg+1;
        else:
            cur_pos = cur_pos+1;
        rs = max(rs, cur_pos*1.0 / pos - cur_neg*1.0 / neg)
    return rs

def ks_scoring_func(y, y_pred):
    df = pd.DataFrame()

    df['y_pred'] = y_pred[:,1]
    df['y'] = y
    pos = df['y'].value_counts()[0]
    neg = df['y'].value_counts()[1]
    df = df.sort_values(by='y_pred')
    #print df
    cur_pos = 0;
    cur_neg = 0;
    rs = 0
    for index, row in df.iterrows():
        if row['y'] == 1:
            cur_neg = cur_neg + 1;
        else:
            cur_pos = cur_pos + 1;
        rs = max(rs, cur_pos * 1.0 / pos - cur_neg * 1.0 / neg)
    return rs

train_x_raw = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/user_info_train.txt", header=None)
train_y = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/overdue_train.txt", header=None)

test_x = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/user_info_test.txt", header=None)
train_x_raw.columns = ['id', 'gender', 'pro', 'edu', 'marry', 'hukou']
test_x.columns = ['id', 'gender', 'pro', 'edu', 'marry', 'hukou']
train_y.columns = ['id', 'overdue']
t = pd.merge(train_x_raw, train_y, on=['id','id'], how='inner')
# print train_x
# print train_y
# print(t.loc[t['id'] == 55571])
#del train_x['id']
#onehot_dict = train_x
train_x = train_x_raw.drop('id', 1)
onehot_dict = train_x.append(test_x.drop('id',1));
enc = preprocessing.OneHotEncoder()
enc.fit(onehot_dict)
array = enc.transform(train_x)
array = preprocessing.PolynomialFeatures().fit_transform(array.toarray())
label = t['overdue'].values
t = t.drop('overdue',1)
n =0
for i in label:
    if i==1:
        n=n+1
print n
newid = True
oldid = -1
salary_num =0
cur=0
arr = [[0.0 for col in range(5)] for row in range(len(train_x))]
with open("/Users/ericzhou.zc/Downloads/credit/train/bank_detail_train.txt", "rb") as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        if int(row[0]) == oldid:
            newid = False
            arr[cur][2] = arr[cur][2] + 1
            if (row[2] == '0'):  # income
                arr[cur][1] = arr[cur][1] + 1
            arr[cur][3] = arr[cur][3] + float(row[3])
            if (int(row[4]) == 1):
                salary_num = salary_num + 1
                arr[cur][4] = float(row[3]) + arr[cur][4]
        else:
            # save.todo
            if oldid != -1:
                arr[cur][0] = oldid
                if salary_num != 0:
                    arr[cur][4] = arr[cur][4] / salary_num
                cur = cur+1
            # create new

            oldid = int(row[0])
            salary_num = 0
            newid = True

train_bank_x = pd.DataFrame(arr, columns=['id','in_num','trade_num','volumn', 'salary'])
print "merge begin"
t = pd.merge(t, train_bank_x, on=['id','id'], how='left')
t = t.fillna(0)
t = t.drop('id',1)
print t
print label
array = t
#readProfileFile("/Users/ericzhou.zc/Downloads/credit/train/user_info_train.txt")
lr = LogisticRegression(penalty='l2')
#, class_weight={0:0.5,1:3})#, class_weight = 'balanced')

my_scorer = metrics.make_scorer(ks_scoring_func, greater_is_better=True, needs_proba=True)
print cross_validation.cross_val_score(lr, array, label, cv=10, scoring=my_scorer)
# model_selection.cross_val_predict


lr.fit(array, label)
rs = pd.DataFrame(lr.predict(array))
#rs.columns=['id', 'label']
print("lr predict status:")
print(rs[0].value_counts())
print (lr.score(array, label))
print("lr predict prob array:")
result = pd.DataFrame(lr.predict_proba(array))
result.columns=['prob_pos', 'prob_neg']
result['true_label']=label
result['id']=train_x_raw['id']
# trues = result.loc[result['true_label']==1]
# print trues.loc[trues['prob_neg']>0.5]
#print result





ttt = ks_score(result)
print ttt