#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-21, 14:15

@Description:

@Update Date: 17-9-21, 14:15
"""
from util import *


def easy_test1():
    # 基于一种简单假设,
    # 如果在train中流失的客户我们认为在下个月(test)中依然流失,设置为0.95
    # 如果在train中没有流失的客户我们认为在下个月(test)中依然没有流失,设置为0.95
    # 其他情况我们设置为0.1(大多数人不会流失的假设)
    train = load_train()
    test = load_test()
    train_churn_msno = train[train.is_churn == 1].msno
    train_not_churn_msno = train[train.is_churn == 0].msno
    test["is_churn"] = 0.1
    test.loc[np.in1d(test.msno, train_churn_msno), "is_churn"] = 0.95
    test.loc[np.in1d(test.msno, train_not_churn_msno), "is_churn"] = 0.05
    save_result(test, "../result/easy_test1.csv")


def find_cancel_user():
    transactions = load_transactions(in_train_test=False, iterator=False)
    msno_uni = transactions[transactions.is_cancel == 1].msno.unique()
    print "exist cancel user unique size:", len(msno_uni)
    find = transactions[np.in1d(transactions.msno, msno_uni)]
    find = find.sort_values(by=["msno", "transaction_date"])
    find.to_csv("../data/exist_cancel_user_transactions.csv")
    others = transactions[~np.in1d(transactions.msno, msno_uni)]
    print "no exist cancel user unique size:", others.msno.unique().shape[0]
    others = others.sort_values(by=["msno", "transaction_date"])
    others.to_csv("../data/no_exist_cancel_user_transactions.csv")


if __name__ == '__main__':
    # easy_test1()
    find_cancel_user()
