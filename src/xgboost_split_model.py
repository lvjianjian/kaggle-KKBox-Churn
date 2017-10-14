#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-27, 13:17

@Description:

@Update Date: 17-9-27, 13:17
"""
from util import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
import gc


# 明天工作：
#   调整split_size
#   抽取特征

def main():
    round_n = 150
    train = preprocess_transactions(load_train(), "20170131", "../data/cache/train_do_trans2.csv")
    test = preprocess_transactions(load_test(), "20170228", "../data/cache/test_do_trans2.csv")
    train_logs = preprocess_user_logs(None, None, "20170131", "../data/cache/train_do_logs.csv")
    if "is_churn" in train_logs.columns:
        train_logs = train_logs.drop("is_churn", axis=1)
    test_logs = preprocess_user_logs(None, None, "20170228", "../data/cache/test_do_logs.csv")
    if "is_churn" in test_logs.columns:
        test_logs = test_logs.drop("is_churn", axis=1)
    train = pd.merge(train, train_logs, on="msno", how="left")
    test = pd.merge(test, test_logs, on="msno", how="left")
    members = load_members()
    # members["expiration_date_dayofyear"] = pd.to_datetime(members.expiration_date).dt.dayofyear
    # members["expiration_date_month"] = pd.to_datetime(members.expiration_date).dt.month
    # members["expiration_date_year"] = pd.to_datetime(members.expiration_date).dt.year
    # members["expiration_date_values"] = pd.to_datetime(members.expiration_date).dt.value / (10 ** 13)
    train = pd.merge(train, members, on="msno", how="left")
    test = pd.merge(test, members, on="msno", how="left")
    train["gender"] = np.where(train.gender == "male", 1, np.where(train.gender == "female", 0, train.gender))
    test["gender"] = np.where(test.gender == "male", 1, np.where(test.gender == "female", 0, test.gender))
    f = list(train.columns)
    f.remove("msno")
    f.remove("is_churn")
    # f.remove("registration_init_time")
    # f.remove("expiration_date")

    train_new_in_members, train_new_not_in_members, \
    test_new_in_members, test_new_not_in_members = load_not_in_transactions()

    # 对1600个单独训练
    train_m = train[np.in1d(train.msno, train_new_in_members.msno.unique())]
    test_m = test[np.in1d(test.msno, test_new_in_members.msno.unique())]
    x_train_m = train_m[["city", "bd", "gender", "registered_via",
                         "registration_init_time", "expiration_date"]].values.astype(float)
    y_train_m = train_m["is_churn"].values.astype(int)
    x_test_m = test_m[
        ["city", "bd", "gender", "registered_via", "registration_init_time", "expiration_date"]].values.astype(float)
    xgb_train_m = xgb.DMatrix(x_train_m, label=y_train_m)
    xgb_test_m = xgb.DMatrix(x_test_m)
    watch_list = [(xgb_train_m, "train")]
    xgb_pars_m = {'min_child_weight': 10,
                  'eta': 0.02,
                  'colsample_bytree': 1,
                  'max_depth': 5,
                  'subsample': 0.8,
                  'lambda': 0,
                  'alpha': 0,
                  'gamma': 0,
                  'nthread': -1, 'booster': 'gbtree', 'silent': 1,
                  'eval_metric': 'logloss', 'objective': 'binary:logistic'}
    bst = xgb.train(xgb_pars_m, xgb_train_m, round_n, watch_list, early_stopping_rounds=2)
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
    split_size = 2
    train_split = split_origin_ration(train, split_size=split_size)
    predicts = []
    for i in xrange(len(train_split)):
        train = train_split[0]

        x = train[f].values.astype(float)
        y = train["is_churn"].values.astype(int)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
        xgb_test = xgb.DMatrix(test[f].values, label=test["is_churn"])
        xgb_alltrain = xgb.DMatrix(x, label=y)
        watch_list = [(xgb_train, "train"), (xgb_valid, "valid")]
        xgb_pars = {'min_child_weight': 28,
                    'eta': 0.02,
                    'colsample_bytree': 0.8,
                    'max_depth': 7,
                    'subsample': 0.8,
                    'lambda': 1,
                    'alpha': 0,
                    'gamma': 0,
                    'nthread': -1, 'booster': 'gbtree', 'silent': 1,
                    'eval_metric': 'logloss', 'objective': 'binary:logistic'}
        bst = xgb.train(xgb_pars, xgb_alltrain, round_n, watch_list, early_stopping_rounds=2)
        p = bst.predict(xgb_test)

        print p.mean(), p.max(), p.min()
        print p.shape
        print (p > 0.5).sum()
        predicts.append(p)
        del bst
        gc.collect()

    p = np.mean(predicts, axis=0)
    test["is_churn"] = p
    test_new_not_in_members["is_churn"] = 0.5
    test = pd.concat([test_m, test, test_new_not_in_members])

    save_result(test, "../result/xgb_split_or_{}_test_f{}_{}r.csv".format(split_size,len(f), round_n))


if __name__ == '__main__':
    main()
