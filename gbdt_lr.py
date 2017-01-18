# -*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import data_processing as dp
import pandas as pd

class GbdtLR:
    def __init__(self, l = 0.01):
        self.l = l
        self.gbdt = GradientBoostingClassifier()
        self.lr = LogisticRegression();
        self.onehot = OneHotEncoder()

    def predict(self, X):
        t = self.gbdt.apply(X)[:, :, 0]
        t = self.onehot.transform(t)
        return self.lr.predict(t)

    def predict_proba(self, X):
        t = self.gbdt.apply(X)[:, :, 0]
        t = self.onehot.transform(t)
        return self.lr.predict_proba(t)

    def fit(self, X, y, **kwargs):
        self.gbdt.fit(X, y)
        lr_train_x = self.gbdt.apply(X)[:, :, 0]
        self.onehot.fit(lr_train_x)
        self.lr.fit(self.onehot.transform(lr_train_x), y)

    def get_params(self, deep = False):
        return {'l':self.l}

    def score(self, X, y):
        t = self.gbdt.apply(X)[:, :, 0]
        t = self.onehot.transform(t)
        return self.lr.score(t, y)