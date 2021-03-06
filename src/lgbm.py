#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-13, 14:34

@Description:

@Update Date: 17-10-13, 14:34
"""

import lightgbm as lgb
from util import *
from sklearn.model_selection import train_test_split


def main():
    # train = load_train()
    round_n = 100
    lr = 0.02
    train, test = load_train_and_test()
    f = list(train.columns)
    # print f
    # exit(1)
    f.remove("msno")
    f.remove("is_churn")
    # f.remove("registration_init_time")
    # f.remove("expiration_date")

    train_new_in_members, train_new_not_in_members, \
    test_new_in_members, test_new_not_in_members = load_not_in_transactions()

    # 对1600个单独训练
    train_m = train[np.in1d(train.msno, train_new_in_members.msno.unique())]
    test_m = test[np.in1d(test.msno, test_new_in_members.msno.unique())]
    x_train_m = train_m[members_feature_columns].values.astype(float)
    y_train_m = train_m["is_churn"].values.astype(int)
    x_test_m = test_m[members_feature_columns].values.astype(float)
    xgb_train_m = lgb.Dataset(x_train_m, label=y_train_m)
    xgb_test_m = lgb.Dataset(x_test_m)
    watch_list = [(xgb_train_m, "train")]
    xgb_pars_m = {'min_child_weight': 5,
                  'eta': 0.05,
                  'colsample_bytree': 1,
                  'max_depth': 5,
                  'subsample': 0.8,
                  'lambda': 0,
                  'alpha': 0,
                  'gamma': 0,
                  'nthread': -1, 'booster': 'gbtree', 'silent': 1, 'seed': 2017,
                  'eval_metric': 'logloss', 'objective': 'binary:logistic'}
    bst = lgb.train(xgb_pars_m, xgb_train_m, round_n, watch_list, early_stopping_rounds=2)
    p1 = bst.predict(xgb_test_m)
    test_m["is_churn"] = p1
    print p1.mean(), p1.max(), p1.min()
    print p1.shape
    print (p1 > 0.5).sum()

    print "rest train"
    train = train[~np.in1d(train.msno, train_new_in_members.msno.unique())]
    train = train[~np.in1d(train.msno, train_new_not_in_members.msno.unique())]
    test = test[~np.in1d(test.msno, test_new_in_members.msno.unique())]
    test = test[~np.in1d(test.msno, test_new_not_in_members.msno.unique())]
    # print f
    # print
    # print train.columns
    train = train.fillna(0)
    test = test.fillna(0)
    f = filter_feature(train)
    x = train[f].values.astype(float)
    y = train["is_churn"].values.astype(int)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
    xgb_train = lgb.Dataset(x_train, label=y_train)
    xgb_valid = lgb.Dataset(x_valid, label=y_valid)
    # print len(f)
    # print test.columns
    xgb_test = lgb.Dataset(test[f].values.astype(float))
    xgb_alltrain = lgb.Dataset(x, label=y)
    watch_list = [(xgb_train, "train"), (xgb_valid, "valid")]
    min_child_weight = 10
    cb = 1
    md = 7
    ss = 1
    la = 1
    al = 0
    ga = 0
    xgb_pars = {'min_child_weight': min_child_weight,
                'eta': lr,
                'colsample_bytree': cb,
                'max_depth': md,
                'subsample': ss,
                'lambda': la,
                'alpha': al,
                'gamma': ga,
                'nthread': -1, 'booster': 'gbtree', 'silent': 1,
                'eval_metric': 'logloss', 'objective': 'binary:logistic'}
    bst = lgb.train(xgb_pars, xgb_alltrain, round_n, watch_list, early_stopping_rounds=2)
    p = bst.predict(xgb_test)
    test["is_churn"] = p
    print p.mean(), p.max(), p.min()
    print p.shape
    print (p > 0.5).sum()

    test_new_not_in_members["is_churn"] = 0.5

    test = pd.concat([test_m, test, test_new_not_in_members])
    save_path = "../result/xgb_test_f{}_{}r_{}lr_{}mcw_{}cb_{}md_{}ss_{}la_{}al_{}ga.csv".format(len(f),
                                                                                                 round_n,
                                                                                                 lr,
                                                                                                 min_child_weight,
                                                                                                 cb,
                                                                                                 md,
                                                                                                 ss,
                                                                                                 la,
                                                                                                 al,
                                                                                                 ga)
    save_result(test, save_path)
    print save_path


if __name__ == '__main__':
    main()