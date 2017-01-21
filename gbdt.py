# -*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import data_processing_new as dp
import pandas as pd
from gbdt_lr import GbdtLR
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

print "hdhhd"
train_x = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/input.txt")
test_x = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/input.txt")
tmp2 = test_x[['id','gender', 'pro', 'edu', 'marry', 'hukou']];
onehot_dict = train_x[['id','gender', 'pro', 'edu', 'marry', 'hukou']].append(tmp2, ignore_index=True)
test_x=test_x.drop(['gender', 'pro', 'edu', 'marry', 'hukou'],1)
train_x=train_x.drop(['gender', 'pro', 'edu', 'marry', 'hukou'],1)
label_tmp = onehot_dict['id']
onehot_dict=onehot_dict.drop('id',1)
enc = OneHotEncoder()
enc.fit(onehot_dict)
t = pd.DataFrame(enc.transform(onehot_dict).toarray())
t['id']=label_tmp
train_x = pd.merge(train_x, t, on=['id','id'], how='left')
test_x = pd.merge(test_x, t, on=['id','id'], how='left')


#neg = train_x.loc[train_x['overdue']==1]
#train_x = train_x.append(neg,ignore_index=True)
label = train_x['overdue']
train_x = train_x.drop('overdue',1).drop('id',1)

#print train_x.columns

ids = test_x['id']
test_x = test_x.drop('id',1)
# print test_x.columns
# train_x, label = dp.get_train_data(browse=True)
# gbdt = ExtraTreeClassifier()
# print cross_validation.cross_val_score(gbdt, train_x, label, cv=5, scoring='roc_auc')

#gbdt = GbdtLR()
# print cross_validation.cross_val_score(gbdt, train_x, label, cv=10, scoring='roc_auc')
#print cross_validation.cross_val_score(gbdt, train_x, label, cv=10, scoring='roc_auc')
#rf = RandomForestClassifier()
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_x, label, test_size=0.33, random_state=42)
#gbdt.fit(train_x, label)

# model = SelectFromModel(gbdt.gbdt, prefit=True)
# train_x = model.transform(train_x)
# print cross_validation.cross_val_score(gbdt, train_x, label, cv=5, scoring='roc_auc')
# print gbdt.score(X_test, y_test)
#
# rs = pd.DataFrame(gbdt.predict_proba(train_x))
# rs['true_label'] = label
# x = pd.DataFrame()
# x['imp'] = gbdt.feature_importances_
# x['name'] = train_x.columns
# print x
# print rs

lr = LogisticRegression();
rf = RandomForestClassifier();
print cross_validation.cross_val_score(lr, train_x, label, cv=5, scoring='roc_auc')
# lr_train_x = gbdt.apply(train_x)[:,:,0]
# onehot = OneHotEncoder()
# lr_train_x = onehot.fit_transform(lr_train_x)
#
# print cross_validation.cross_val_score(lr, lr_train_x, label, cv=10, scoring='roc_auc')
#
lr.fit(train_x, label)
# print lr.predict_proba(lr_train_x)


# print("gbdt predict status:")
# rs = gbdt.predict(test_x)
# test_x['rs'] = rs
# #print test_x.iloc[:,[0,6]]
#
#test_x, ids = dp.get_test_data(browse=True)
#
# gbdt.fit(train_x, label)
lr_test_x = lr.predict_proba(test_x)
rs = pd.DataFrame(lr_test_x[:,1])
rs.columns=['probability']
rs['userid']=ids
rs.to_csv("/Users/ericzhou.zc/Downloads/credit/test/gbdt_lr_rs_onehotlr.csv", columns=['userid','probability'], index=False)
