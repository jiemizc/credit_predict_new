# -*- coding: utf-8 -*-
from sklearn import metrics
import csv
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.decomposition  import NMF
from scipy import sparse
import numpy as np
def cut(df, colname, bin):
    arr = df[colname]
    df[colname] = pd.cut(arr, bin).labels
    return df

def ks_scoring_func(y, y_pred):
    df = pd.DataFrame()
    df['y_pred'] = y_pred[:,1]
    df['y'] = y
    pos = df['y'].value_counts()[0]
    neg = df['y'].value_cdataounts()[1]
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

def ks_scoring_func(y, y_pred, i=1):
    df = pd.DataFrame()
    df['y_pred'] = y_pred[:,i]
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

my_scorer = metrics.make_scorer(ks_scoring_func, greater_is_better=True, needs_proba=True)

def get_train_data(one_hot_for_categorial = False):
    train_x_profile = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/user_info_train.txt", header=None)
    train_y = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/overdue_train.txt", header=None)

    train_x_profile.columns = ['id', 'gender', 'pro', 'edu', 'marry', 'hukou']
    train_y.columns = ['id', 'overdue']

    #为了调整label的顺序是按照profile的id顺序
    t = pd.merge(train_x_profile, train_y, on=['id','id'], how='inner')
    label = t['overdue'].values

    train_bank_x = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/bank_detail_train.txt", header=None)
    train_bank_x.columns = ['id','time','trade_type','volumn','salary_tag']
    grouped_bank_x = train_bank_x.groupby('id')
    print "generating bank_detail_aggregation"

    train_bank = grouped_bank_x.agg({
        'time': {'tradenum':lambda x: len(x)},
        'volumn': ['min', 'max', 'mean', 'sum'],
        'salary_tag': ['sum']
    })
    tmp = grouped_bank_x.trade_type.agg({'income_num':lambda x:sum(x==1), 'outcome_num':lambda x : sum(x==0)})
    train_bank['income_num'] = tmp['income_num']
    train_bank['outcome_num'] = tmp['outcome_num']

    train_bank.columns = ['_'.join(col).strip() for col in train_bank.columns.values]
    print "merge profile with bank begin"
    t = pd.merge(t, train_bank, right_index=True, left_on='id', how='left')

    #读取信用卡记录
    train_x_credit = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/bill_detail_train.txt", header=None)
    train_x_credit.columns = ['id', 'time', 'bank_id', 'last_to_pay', 'last_pay', 'credit_max', 'cur_bill_left', 'cur_min_to_pay',
                              'bill_num', 'cur_to_pay', 'adjust_num', 'profit', 'cur_available', 'cash_num', 'status']
    threshold = -2.5
    grouped_credit = train_x_credit.groupby('id')

    print "generating credit_detail_aggregation"
    #cust = lambda g: sum(train_x_credit.ix[g.index]['last_pay'] - train_x_credit.ix[g.index]['last_to_pay'] <threshold)
    cust = lambda g: sum(train_x_credit.ix[g.index]['last_pay'] - train_x_credit.ix[g.index]['last_to_pay'] < threshold)\
                    *1.0 / (1+sum(train_x_credit.ix[g.index]['last_to_pay']>0))
    grouped_credit_x = grouped_credit.agg({
        'bank_id': ['count', lambda x: x.nunique()],
        'last_to_pay': ['min', 'max', 'mean', 'count', 'std'],
        'last_pay': ['min', 'max', 'mean', 'std',cust],
        'credit_max': ['max','min', 'mean', 'std'],
        'cur_bill_left': ['min', 'max', 'mean', 'std'],
        'cur_min_to_pay': ['min', 'max', 'mean', 'std'],
        'bill_num': ['sum','min','mean', 'std'],
        'cur_to_pay': ['min', 'max', 'mean', 'std'],
        'cur_available': ['min', 'max', 'mean', 'std'],
        # 'status':[{'distinct_snum':lambda x:x.nunique()}],
        'cash_num':['min', 'max', 'mean', 'std']})

    grouped_credit_x.columns = ['_'.join(col).strip() for col in grouped_credit_x.columns.values]
    print grouped_credit_x
    print "merge credit_detail_aggregation"
    t = pd.merge(t, grouped_credit_x, right_index=True, left_on='id', how='left')
    # for id, group in grouped_credit:
    #     violate = 0
    #     for row in group.itertuples:
    #         if row['last_pay'] - row['last_to_pay'] < -1:
    #             violate = violate + 1

    print "reading browse_history"

    train_x_browse = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/train/browse_history_train.txt", header=None)
    train_x_browse.columns = ['id','time','bid','btype']

    test_x_browse = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/browse_history_test.txt", header=None)
    test_x_browse.columns = ['id', 'time', 'bid', 'btype']
    combine_x_browse = train_x_browse.append(test_x_browse);
    onehot_dict = pd.DataFrame(combine_x_browse['bid'])
    enc = pp.OneHotEncoder()
    enc.fit(onehot_dict)
    tt = pd.DataFrame(enc.transform(onehot_dict).toarray())
    fsize = enc.active_features_.size
    columns=[]
    for i in range(0, fsize):
        columns.append('bid_'+str(i));
    tt['id'] = combine_x_browse['id']
    t = pd.merge(t, tt, on=['id','id'], how='left')
    t = t.fillna(0)
    t = t.drop('id', 1)

    # 归一化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # t[['in_num','trade_num', 'volumn', 'salary']] = t[['in_num','trade_num', 'volumn', 'salary']].apply(
    #                           lambda x: min_max_scaler.fit_transform(x))

    print "training data"
    print t
    train_x = t
    return train_x

def get_test_data(one_hot_for_categorial=False):
    train_x_profile = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/user_info_test.txt", header=None)
    train_x_profile.columns = ['id', 'gender', 'pro', 'edu', 'marry', 'hukou']
    t = train_x_profile
    if one_hot_for_categorial == True:
        onehot_dict = train_x_profile.drop('id', 1)  # .append(test_x.drop('id',1));
        enc = pp.OneHotEncoder()
        enc.fit(onehot_dict)
        t = pd.DataFrame(enc.transform(onehot_dict).toarray())
        t['id'] = train_x_profile['id']

    # oldid = -1
    # salary_num = 0
    # cur = 0
    # bank_arr = [[0.0 for col in range(5)] for row in range(len(train_x_profile))]
    # with open("/Users/ericzhou.zc/Downloads/credit/test/bank_detail_test.txt", "rb") as csvfile:
    #     datareader = csv.reader(csvfile)
    #     for row in datareader:
    #         if int(row[0]) == oldid:
    #             newid = False
    #             bank_arr[cur][2] = bank_arr[cur][2] + 1
    #             if (row[2] == '0'):  # income
    #                 bank_arr[cur][1] = bank_arr[cur][1] + 1
    #             bank_arr[cur][3] = bank_arr[cur][3] + float(row[3])
    #             if (int(row[4]) == 1):
    #                 salary_num = salary_num + 1
    #                 bank_arr[cur][4] = float(row[3]) + bank_arr[cur][4]
    #         else:
    #             # save.todo
    #             if oldid != -1:
    #                 bank_arr[cur][0] = oldid
    #                 if salary_num != 0:
    #                     bank_arr[cur][4] = bank_arr[cur][4] / salary_num
    #                 cur = cur + 1
    #             # create new
    #
    #             oldid = int(row[0])
    #             salary_num = 0
    #             newid = True
    #
    #
    # train_bank_x = pd.DataFrame(bank_arr, columns=['id', 'in_num', 'trade_num', 'volumn', 'salary'])
    # print "merge profile with bank begin"
    # t = pd.merge(train_x_profile, train_bank_x, on=['id', 'id'], how='left')

    train_bank_x = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/bank_detail_test.txt", header=None)
    train_bank_x.columns = ['id', 'time', 'trade_type', 'volumn', 'salary_tag']
    grouped_bank_x = train_bank_x.groupby('id')

    print "generating bank_detail_aggregation"

    train_bank = grouped_bank_x.agg({
        'time': {'tradenum': lambda x: len(x)},
        'volumn': ['min', 'max', 'mean', 'sum'],
        'salary_tag': ['sum']
    })
    tmp = grouped_bank_x.trade_type.agg({'income_num': lambda x: sum(x == 1), 'outcome_num': lambda x: sum(x == 0)})
    train_bank['income_num'] = tmp['income_num']
    train_bank['outcome_num'] = tmp['outcome_num']
    print "merge profile with bank begin"
    t = pd.merge(t, train_bank, right_index=True, left_on='id', how='left')
    # 读取信用卡记录
    train_x_credit = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/bill_detail_test.txt", header=None)
    train_x_credit.columns = ['id', 'time', 'bank_id', 'last_to_pay', 'last_pay', 'credit_max', 'cur_bill_left',
                              'cur_min_to_pay',
                              'bill_num', 'cur_to_pay', 'adjust_num', 'profit', 'cur_available', 'cash_num', 'status']
    threshold = -1
    grouped_credit = train_x_credit.groupby('id')

    #cust = lambda g: sum(train_x_credit.ix[g.index]['last_pay'] - train_x_credit.ix[g.index]['last_to_pay'] < threshold)
    cust = lambda g: sum(train_x_credit.ix[g.index]['last_pay'] - train_x_credit.ix[g.index]['last_to_pay'] < threshold)\
                    *1.0 / (1+sum(train_x_credit.ix[g.index]['last_to_pay']>0))
    grouped_credit_x = grouped_credit.agg({
        'bank_id': ['count', {'distinct_bnum': lambda x: x.nunique()}],
        'last_to_pay': ['min', 'max', 'mean', 'count', 'std'],
        'last_pay': ['min', 'max', 'mean', 'std', {'overduenum':cust}],
        'credit_max': ['max', 'min', 'mean', 'std'],
        'cur_bill_left': ['min', 'max', 'mean', 'std'],
        'cur_min_to_pay': ['min', 'max', 'mean', 'std'],
        'bill_num': ['sum', 'min', 'mean', 'std'],
        'cur_to_pay': ['min', 'max', 'mean', 'std'],
        'cur_available': ['min', 'max', 'mean', 'std'],
        'cash_num': ['min', 'max', 'mean', 'std']})

    print grouped_credit_x

    t = pd.merge(t, grouped_credit_x, right_index=True, left_on='id', how='left')
    # for id, group in grouped_credit:
    #     violate = 0
    #     for row in group.itertuples:
    #         if row['last_pay'] - row['last_to_pay'] < -1:
    #             violate = violate + 1


    print "reading browse_history"

    train_x_browse = pd.read_csv("/Users/ericzhou.zc/Downloads/credit/test/browse_history_test.txt", header=None)
    train_x_browse.columns = ['id', 'time', 'bid', 'btype']
    rows = train_x_browse['id']
    cols = train_x_browse['bid']
    m = sparse.coo_matrix((np.ones_like(rows), (rows, cols)))
    svd = NMF(n_components=3)
    user_vectors = svd.fit_transform(m)
    #user_vectors[user_vectors<1e-4]=0
    #user_vectors = pp.normalize(user_vectors)
    item_vectors = svd.components_
    ufeature = pd.DataFrame(user_vectors, columns=['emb0', 'emb1', 'emb2'])

    print "merging browse_history"
    t = pd.merge(t, ufeature, right_index=True, left_on='id', how='left')
    t = t.fillna(0)
    ids = t['id']
    t = t.drop('id', 1)

    # 归一化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # t[['in_num','trade_num', 'volumn', 'salary']] = t[['in_num','trade_num', 'volumn', 'salary']].apply(
    #                           lambda x: min_max_scaler.fit_transform(x))

    print "test data:"
    print t

    train_x = t
    return train_x, ids