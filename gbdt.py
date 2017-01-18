# -*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import data_processing_new as dp
import pandas as pd
from gbdt_lr import GbdtLR
from sklearn.ensemble import RandomForestClassifier

print "hdhhd"

train_x, label = dp.get_train_data(browse=True)
gbdt = GradientBoostingClassifier()
#print cross_validation.cross_val_score(gbdt, train_x, label, cv=10, scoring='roc_auc')
#gbdt = GbdtLR()
print cross_validation.cross_val_score(gbdt, train_x, label, cv=10, scoring='roc_auc')
#rf = RandomForestClassifier()

gbdt.fit(train_x, label)
print gbdt.score(train_x, label)

rs = pd.DataFrame(gbdt.predict_proba(train_x))
rs['true_label'] = label
x = pd.DataFrame()
x['imp'] = gbdt.feature_importances_
x['name'] = train_x.columns
print x
print rs

# lr = LogisticRegression();
# lr_train_x = gbdt.apply(train_x)[:,:,0]
# onehot = OneHotEncoder()
# lr_train_x = onehot.fit_transform(lr_train_x)
#
# print cross_validation.cross_val_score(lr, lr_train_x, label, cv=10, scoring='roc_auc')
#
# lr.fit(lr_train_x, label)
# print lr.predict_proba(lr_train_x)


# print("gbdt predict status:")
# rs = gbdt.predict(test_x)
# test_x['rs'] = rs
# #print test_x.iloc[:,[0,6]]

test_x, ids = dp.get_test_data(browse=True)
lr_test_x = gbdt.predict_proba(test_x)
rs = pd.DataFrame(lr_test_x[:,1])
rs.columns=['probability']
rs['userid']=ids
rs.to_csv("/Users/ericzhou.zc/Downloads/credit/test/gbdt_lr_rs_.csv", columns=['userid','probability'], index=False)
