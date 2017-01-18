# -*- coding: utf-8 -*-
from sklearn import metrics
import csv
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.decomposition import NMF
from scipy import sparse
import numpy as np


def ks_scoring_func(y, y_pred):
    df = pd.DataFrame()
    df['y_pred'] = y_pred[:, 1]
    df['y'] = y
    pos = df['y'].value_counts()[0]
    neg = df['y'].value_cdataounts()[1]
    df = df.sort_values(by='y_pred')
    # print df
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


def ks_scoring_func(y, y_pred, i=1):
    df = pd.DataFrame()
    df['y_pred'] = y_pred[:, i]
    df['y'] = y
    pos = df['y'].value_counts()[0]
    neg = df['y'].value_counts()[1]
    df = df.sort_values(by='y_pred')
    # print df
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


my_scorer = metrics.make_scorer(ks_scoring_func, greater_is_better=True, needs_proba=True)


def get_user_vector_for_browse():
    train_x_browse = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/browse_history_train.txt", header=None)
    train_x_browse.columns = ['id', 'time', 'bid', 'btype']
    test_x_browse = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/browse_history_test.txt", header=None)
    test_x_browse.columns = ['id', 'time', 'bid', 'btype']
    combine_x_browse = train_x_browse.append(test_x_browse)
    rows = combine_x_browse['id']
    cols = combine_x_browse['bid']
    m = sparse.coo_matrix((np.ones_like(rows), (rows, cols)))
    svd = NMF(n_components=3)
    # both train & test user vector
    user_vectors = svd.fit_transform(m)
    return pd.DataFrame(user_vectors, columns=['emb0', 'emb1', 'emb2'])

# user_vectors=[]
user_vectors = get_user_vector_for_browse()


def get_profile(one_hot_for_categorial=False, file='train'):
    if file == 'train':
        file = "/Users/ericzhou.zc/Downloads/credit/train/user_info_train.txt"
    else:
        file = "/Users/ericzhou.zc/Downloads/credit/test/user_info_test.txt"
    train_x_profile = pd.read_csv(file, header=None)
    train_x_profile.columns = ['id', 'gender', 'pro', 'edu', 'marry', 'hukou']
    if one_hot_for_categorial == True:
        onehot_dict = train_x_profile.drop('id', 1)  # .append(test_x.drop('id',1));
        enc = pp.OneHotEncoder()
        enc.fit(onehot_dict)
        t = pd.DataFrame(enc.transform(onehot_dict).toarray())
        t['id'] = train_x_profile['id']
        train_x_profile = t;

    return train_x_profile


def get_label():
    train_y = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/overdue_train.txt", header=None)
    train_y.columns = ['id', 'overdue']
    return train_y

def get_bank_detail_period(file='train'):
    if file == 'train':
        file = "/Users/ericzhou.zc/Downloads/credit/train/bank_detail_train.txt"
        file_loan = "/Users/ericzhou.zc/Downloads/credit/train/loan_time_train.txt"
    else:
        file = "/Users/ericzhou.zc/Downloads/credit/test/bank_detail_test.txt"
        file_loan = "/Users/ericzhou.zc/Downloads/credit/test/loan_time_test.txt"
    train_bank_x = pd.read_csv(file, header=None)
    train_bank_x.columns = ['id', 'time', 'trade_type', 'volumn', 'salary_tag']

    loan_time = pd.read_csv(file_loan, header=None)
    loan_time.columns = ['id', 'loan_time']

    train_bank_x['upper'] = train_bank_x['time'].values + 86400 * 31
    train_bank_x = pd.merge(train_bank_x, loan_time, on=['id', 'id'], how='left')
    train_bank_x = train_bank_x[train_bank_x.apply(lambda x: x['upper'] >= x['loan_time'] >= x['time'], axis=1)]

    income_x = train_bank_x.loc[train_bank_x['trade_type'] == 0]
    income_x = income_x.groupby('id');
    income_x = income_x.agg({
        'time': {'perios_income_num': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean', 'sum'],
    })
    outcome_x = train_bank_x.loc[train_bank_x['trade_type'] == 1]
    outcome_x = outcome_x.groupby('id');
    outcome_x = outcome_x.agg({
        'time': {'otheroutcome_num': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean', 'sum'],
    })
    income_x.columns = ['_'.join(col).strip() for col in income_x.columns.values]
    outcome_x.columns = ['_'.join(col).strip() for col in outcome_x.columns.values]
    tmp = pd.merge(income_x, outcome_x, left_index=True, right_index=True, how='outer')
    tmp.fillna(0)
    tmp['net_outcome'] = outcome_x['volumn_sum'] - income_x['volumn_sum']
    return tmp

def get_bank_detail(file='train'):
    if file == 'train':
        fp= "/Users/ericzhou.zc/Downloads/credit/train/bank_detail_train.txt"
    else:
        fp = "/Users/ericzhou.zc/Downloads/credit/test/bank_detail_test.txt"
    train_bank_x = pd.read_csv(fp, header=None)
    train_bank_x.columns = ['id', 'time', 'trade_type', 'volumn', 'salary_tag']

    ##工资收入
    salary_x = train_bank_x.loc[train_bank_x['salary_tag'] == 1];
    salary_x = salary_x.groupby('id');
    train_bank_salary = salary_x.agg({
        'time': {'tradenum': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean', 'sum', 'std'],
    })
    train_bank_salary.columns = ['_'.join(col).strip() for col in train_bank_salary.columns.values]

    ##其他收入和支出
    other_x = train_bank_x.loc[train_bank_x['salary_tag'] == 0];

    income_x = other_x.loc[train_bank_x['trade_type'] == 0]
    income_x = income_x.groupby('id');
    income_x = income_x.agg({
        'time': {'otherincome_num': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean'],
    })
    outcome_x = other_x.loc[train_bank_x['trade_type'] == 1]
    outcome_x = outcome_x.groupby('id');
    outcome_x = outcome_x.agg({
        'time': {'otheroutcome_num': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean'],
    })
    income_x.columns = ['_'.join(col).strip() for col in income_x.columns.values]
    outcome_x.columns = ['_'.join(col).strip() for col in outcome_x.columns.values]
    other = pd.merge(income_x, outcome_x, left_index=True, right_index=True, how='outer')
    # period_vector = get_bank_detail_period(file)
    # tmp = pd.merge(train_bank_salary, other, left_index=True, right_index=True, how='outer')

    return pd.merge(train_bank_salary, other, left_index=True, right_index=True, how='outer')

def get_loan_time(file='train'):
    if file == 'train':
        file = "/Users/ericzhou.zc/Downloads/credit/train/loan_time_train.txt"
    else:
        file = "/Users/ericzhou.zc/Downloads/credit/test/loan_time_test.txt"
    loan_time =  pd.read_csv(file, header=None)
    loan_time.columns = ['id','loan_time']

def get_bill_vec(train_x_credit):
    grouped_credit = train_x_credit.groupby('id')
    threshold = -3

    print "generating credit_detail_aggregation"
    # cust = lambda g: sum(train_x_credit.ix[g.index]['last_pay'] - train_x_credit.ix[g.index]['last_to_pay'] <threshold)
    cust = lambda g: sum(train_x_credit.ix[g.index, 'last_pay'] - train_x_credit.ix[g.index, 'last_to_pay'] < threshold) \
                     * 1.0 / (1 + sum(train_x_credit.ix[g.index, 'last_to_pay'] > 0))

    grouped_credit_x = grouped_credit.agg({
        'time': lambda x: len(x),
        'bank_id': ['count', lambda x: x.nunique()],
        'last_to_pay': ['max', 'mean', 'count', 'std','sum'],
        'last_pay': ['min', 'max', 'mean', cust, 'std','sum'],
        'credit_max': ['max'],
        'cur_min_to_pay': ['max', 'mean', 'std'],
        'bill_num': ['max', 'mean', 'std'],
        'cur_to_pay': ['max', 'mean', 'std'],
        'cur_available': ['max', 'mean', 'std'],
        'cash_num': ['max', 'mean', 'std']})

    tmp = grouped_credit.last_pay.agg(
        {'pay_less0': lambda x: sum(x < 0) * 1.0 / (1 + sum(train_x_credit.ix[x.index, 'last_to_pay'] > 0)),
         'pay_more': lambda x: sum(
             train_x_credit.ix[x.index, 'last_pay'] -
             train_x_credit.ix[x.index,
                               'last_to_pay'] > 4) * 1.0 / (1 + sum(train_x_credit.ix[x.index, 'last_to_pay'] > 0))})

    grouped_credit_x['pay_less0'] = tmp['pay_less0']
    grouped_credit_x['pay_more'] = tmp['pay_more']

    # 'status':[{'distinct_s':lambda x: len(x.unique())}]})
    grouped_credit_x.columns = ['_'.join(col).strip() for col in grouped_credit_x.columns.values]
    return grouped_credit_x

def get_bill_data_period(train_x_credit, loan_time, period=60):
    train_x_credit['lower'] = train_x_credit['time'].values - 86400 * period
    train_x_credit['upper'] = train_x_credit['time'].values + 86400 * period
    train_x_credit = pd.merge(train_x_credit, loan_time, on=['id', 'id'], how='left')

    lower = train_x_credit[train_x_credit.apply(lambda x: x['time'] >= x['loan_time'] >= x['lower'], axis=1)]
    lower = lower.drop('lower', 1).drop('upper', 1)
    vec2 = get_bill_vec(lower)

    upper = train_x_credit[train_x_credit.apply(lambda x: x['upper'] >= x['loan_time'] >= x['time'], axis=1)]
    upper = upper.drop('upper', 1).drop('lower', 1)
    vec3 = get_bill_vec(upper)

    return pd.merge(vec2, vec3, left_index=True, right_index=True, how='outer')

def get_bill_data(file='train'):
    if file == 'train':
        file = "/Users/ericzhou.zc/Downloads/credit/train/bill_detail_train.txt"
        file_loan = "/Users/ericzhou.zc/Downloads/credit/train/loan_time_train.txt"
    else:
        file = "/Users/ericzhou.zc/Downloads/credit/test/bill_detail_test.txt"
        file_loan = "/Users/ericzhou.zc/Downloads/credit/test/loan_time_test.txt"

    # 读取信用卡记录
    train_x_credit = pd.read_csv(file, header=None)
    train_x_credit.columns = ['id', 'time', 'bank_id', 'last_to_pay', 'last_pay', 'credit_max', 'cur_bill_left',
                              'cur_min_to_pay',
                              'bill_num', 'cur_to_pay', 'adjust_num', 'profit', 'cur_available', 'cash_num', 'status']
    vec1 = get_bill_vec(train_x_credit)
    threshold = -2.5

    loan_time = pd.read_csv(file_loan, header=None)
    loan_time.columns = ['id', 'loan_time']

    tmp1 = get_bill_data_period(train_x_credit=train_x_credit, loan_time=loan_time, period=60)
    tmp2 = get_bill_data_period(train_x_credit=train_x_credit, loan_time=loan_time, period=30)
    tmp3 = get_bill_data_period(train_x_credit=train_x_credit, loan_time=loan_time, period=90)

    tmp = pd.merge(tmp1, tmp2, left_index=True, right_index=True,how='outer')
    tmp = pd.merge(tmp, tmp3, left_index=True, right_index=True, how='outer')
    return pd.merge(tmp, vec1, left_index=True, right_index=True,how='outer')


def get_train_data(one_hot_for_categorial=False, browse=False):
    profile = get_profile(one_hot_for_categorial=one_hot_for_categorial,file='train')
    bank = get_bank_detail(file='train')
    bill = get_bill_data(file='train')
    label = get_label()

    t = pd.merge(profile, bank, right_index=True, left_on='id', how='left')
    t = pd.merge(t, bill, right_index=True, left_on='id', how='left')
    t = pd.merge(t, label, on=['id','id'], how='left')
    if browse == True:
        t = pd.merge(t, user_vectors, right_index=True, left_on='id', how='left')
    t = t.fillna(0)

    # 归一化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # t[['in_num','trade_num', 'volumn', 'salary']] = t[['in_num','trade_num', 'volumn', 'salary']].apply(
    #                           lambda x: min_max_scaler.fit_transform(x))

    t.to_csv("/Users/ericzhou.zc/Downloads/credit/train/input.txt")

    t = t.drop('id', 1)
    label = t['overdue']
    t = t.drop('overdue', 1)
    return t, label


def get_test_data(one_hot_for_categorial=False, browse=False):
    profile = get_profile(one_hot_for_categorial=one_hot_for_categorial,file='test')
    bank = get_bank_detail(file='test')
    bill = get_bill_data(file='test')

    t = pd.merge(profile, bank, right_index=True, left_on='id', how='left')
    t = pd.merge(t, bill, right_index=True, left_on='id', how='left')
    if browse == True:
        t = pd.merge(t, user_vectors, right_index=True, left_on='id', how='left')
    t = t.fillna(0)

    # 归一化browse
    # min_max_scaler = preprocessing.MinMaxScaler()
    # t[['in_num','trade_num', 'volumn', 'salary']] = t[['in_num','trade_num', 'volumn', 'salary']].apply(
    #                           lambda x: min_max_scaler.fit_transform(x))

    t.to_csv("/Users/ericzhou.zc/Downloads/credit/test/input.txt")

    ids = t['id']
    t = t.drop('id', 1)
    return t, ids
