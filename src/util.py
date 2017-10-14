#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-9-20, 22:49

@Description:

@Update Date: 17-9-20, 22:49
"""

import pandas as pd
import numpy as np
import os
import time
import h5py
import sys
import gc

clip_value = 10 ** -15


def load_train():
    train = pd.read_csv("../data/train.csv")
    return train


def load_test():
    test = pd.read_csv("../data/test.csv")
    return test


members_feature_columns = ["city", "bd", "gender", "registered_via", "registration_init_time", "expiration_date"]


def load_members(in_train_test=True):
    if in_train_test:
        members = pd.read_csv("../data/part_members_1.csv")
    else:
        members = pd.read_csv("../data/members.csv")

    members["city_is_one"] = np.where(members.city == 1, 2, 1)
    members["age_is_zero"] = np.where(members.bd == 0, 2, 1)
    members["age_is_neg"] = np.where(members.bd < 0, 2, 1)
    members["age_larger_100"] = np.where(members.bd > 100, 2, 1)
    members["age_between_10_80"] = np.where((members.bd > 10) & (members.bd < 80), 2, 1)
    members["gender_is_nan"] = np.where(pd.isnull(members.gender), 2, 1)
    members["register_via_is_7"] = np.where(members.registered_via == 7, 2, 1)
    members["register_via_is_9"] = np.where(members.registered_via == 9, 2, 1)
    members["register_via_is_13"] = np.where(members.registered_via == 13, 2, 1)
    init_time = pd.to_datetime(members.registration_init_time.astype(str))
    members["register_init_year"] = init_time.dt.year
    ex_time = pd.to_datetime(members.expiration_date.astype(str))
    members["expiration_year"] = ex_time.dt.year

    def append(columns, name):
        if name not in columns:
            columns.append(name)

    append(members_feature_columns, "age_is_zero")
    append(members_feature_columns, "age_is_neg")
    append(members_feature_columns, "city_is_one")
    append(members_feature_columns, "age_larger_100")
    append(members_feature_columns, "age_between_10_80")
    append(members_feature_columns, "gender_is_nan")
    append(members_feature_columns, "register_via_is_7")
    append(members_feature_columns, "register_via_is_9")
    append(members_feature_columns, "register_via_is_13")
    append(members_feature_columns, "register_init_year")
    append(members_feature_columns, "expiration_year")
    return members


def load_transactions(in_train_test=True, iterator=True):
    if in_train_test:
        transactions = pd.read_csv("../data/part_transactions_1.csv", iterator=iterator)
    else:
        transactions = pd.read_csv("../data/transactions.csv", iterator=iterator)
    return transactions


def filter(filter_fn, save_fn, filter_ids):
    print "filter:" + filter_fn
    save_fn = save_fn.replace(".csv", "_{}.csv")
    data = pd.read_csv(filter_fn, iterator=True)
    loop = True
    chunkSize = 5000000
    maxRecordSize = 30 * chunkSize
    file_index = 1
    chunks = []
    size = 0
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunk = chunk[np.in1d(chunk.msno, filter_ids)]
            chunks.append(chunk)
            size += chunk.shape[0]
            if (size >= maxRecordSize):
                df = pd.concat(chunks, ignore_index=True)
                df.to_csv(save_fn.format(file_index), index=False)
                chunks = []
                size = 0
                file_index += 1
        except StopIteration:
            loop = False
            print "Iteration is stopped."
    if (len(chunks) > 0):
        df = pd.concat(chunks, ignore_index=True)
        df.to_csv(save_fn.format(file_index), index=False)
    return file_index


def save_result(result, path):
    if isinstance(result, np.ndarray):
        pd.DataFrame(result, columns=["msno", "is_churn"]).to_csv(path, index=None)
    elif isinstance(result, pd.DataFrame):
        result[["msno", "is_churn"]].to_csv(path, index=None)


def logloss(predict, real):
    predict = np.clip(predict, clip_value, 1 - clip_value)
    return -np.mean(real * np.log(predict) + (1 - real) * np.log(1 - predict))


def load_part_transactions(part_index=-1):
    part_dfs = []
    split_file_num = 10  # 10为分割文件数量
    assert part_index < split_file_num
    if os.path.exists("../data/cache/part_transactions_0.csv"):
        print "read cache"
        if part_index < 0:
            for i in xrange(split_file_num):
                part_dfs.append(pd.read_csv("../data/cache/part_transactions_{}.csv".format(i)))
        else:
            part_dfs.append(pd.read_csv("../data/cache/part_transactions_{}.csv".format(part_index)))
    else:
        transactions = load_transactions(iterator=False)
        uni = transactions.msno.unique()
        msno_size = len(uni)
        each_size = int(round(float(msno_size) / split_file_num))
        for i in xrange(split_file_num - 1):
            _msno = np.asarray(uni[i * each_size:(i + 1) * each_size]).astype(np.str)
            _part = transactions[np.in1d(transactions.msno, _msno)]
            _part.to_csv("../data/cache/part_transactions_{}.csv".format(i), index=False)
            if part_index < 0:
                part_dfs.append(_part)
            else:
                if part_index == i:
                    part_dfs.append(_part)
        # rest
        _msno = np.asarray(uni[(i + 1) * each_size:]).astype(np.str)
        _part = transactions[np.in1d(transactions.msno, _msno)]
        _part.to_csv("../data/cache/part_transactions_{}.csv".format(i + 1), index=False)
        if part_index < 0:
            part_dfs.append(_part)
        else:
            if part_index == i + 1:
                part_dfs.append(_part)
    return part_dfs


def load_train_transactions(iterator=True):
    train_trans = pd.read_csv("../data/train_transactions.csv", iterator=iterator)
    return train_trans


def load_test_transactions(iterator=True):
    test_trans = pd.read_csv("../data/test_transactions.csv", iterator=iterator, index_col=0)
    return test_trans


def load_user_logs(iterator=True):
    t_logs = pd.read_csv("../data/user_logs.csv", iterator=iterator)
    return t_logs


def load_train_logs(iterator=True):
    t_logs = pd.read_csv("../data/train_logs.csv", iterator=iterator, index_col=0)
    return t_logs


def load_test_logs(iterator=True):
    t_logs = pd.read_csv("../data/test_logs.csv", iterator=iterator, index_col=0)
    return t_logs


def preprocess_trans_part(_part, f_names):
    _part = _part.sort_values(["msno", "transaction_date"])
    count_max = None
    for f_name in f_names:
        count_max_f = _part.groupby(["msno", f_name], as_index=False)[f_name]. \
            agg({"c": "count"}).groupby("msno").max().reset_index()[["msno", f_name]]
        count_max_f.columns = [["msno", "max_count_{}".format(f_name)]]
        if count_max is None:
            count_max = count_max_f
        else:
            count_max = pd.merge(count_max, count_max_f, on="msno", how="left")
    return_feature = count_max
    latest_f = _part.groupby(["msno"], as_index=False).apply(lambda x: x.iloc[-1])
    latest_f = latest_f[["msno"] + f_names]
    latest_f.columns = ["msno"] + ["latest_{}".format(f_name) for f_name in f_names]

    sh1 = _part[["msno"] + f_names].shift(1)
    sh1 = sh1[sh1.msno == _part.msno]
    latest_2_f = sh1.groupby(["msno"], as_index=False).apply(lambda x: x.iloc[-1])

    sh2 = _part[["msno"] + f_names].shift(2)
    sh2 = sh2[sh2.msno == _part.msno]
    latest_3_f = sh2.groupby(["msno"], as_index=False).apply(lambda x: x.iloc[-1])

    # new_f = []
    for i in xrange(len(f_names)):
        index = np.intersect1d(latest_f.index, latest_2_f.index)
        latest_f.loc[index, "last_{}_12_change".format(f_names[i])] = \
            (latest_f.loc[index, "latest_{}".format(f_names[i])] != \
             latest_2_f.loc[index, f_names[i]]).astype(int)
        index = np.intersect1d(latest_f.index, latest_3_f.index)
        latest_f.loc[index, "last_{}_13_change".format(f_names[i])] = \
            (latest_f.loc[index, "latest_{}".format(f_names[i])] != \
             latest_3_f.loc[index, f_names[i]]).astype(int)
        index = np.intersect1d(latest_2_f.index, latest_3_f.index)
        latest_f.loc[index, "last_{}_23_change".format(f_names[i])] = \
            (latest_2_f.loc[index, f_names[i]] != \
             latest_3_f.loc[index, f_names[i]]).astype(int)
        # new_f.append("last_{}_12_change".format(f_names[i]))
        # new_f.append("last_{}_13_change".format(f_names[i]))
        # new_f.append("last_{}_23_change".format(f_names[i]))

    latest_f = latest_f.fillna(0)
    l = list(latest_f.columns)
    l.remove("msno")
    latest_f = pd.concat([latest_f["msno"], latest_f[l].astype(int)], axis=1)

    for i in xrange(len(f_names)):
        latest_f["last_{}_123_exist_change".format(f_names[i])] = \
            (latest_f["last_{}_13_change".format(f_names[i])] | \
             latest_f["last_{}_23_change".format(f_names[i])] | \
             latest_f["last_{}_12_change".format(f_names[i])]).astype(int)

    return_feature = pd.merge(return_feature, latest_f, on="msno", how="left")
    start_f = _part.groupby(["msno"], as_index=False).apply(lambda x: x.iloc[0])
    start_f = start_f[["msno"] + f_names]
    start_f.columns = ["msno"] + ["start_{}".format(f_name) for f_name in f_names]
    return_feature = pd.merge(return_feature, start_f, on="msno", how="left")
    return return_feature


def preprocess_transactions(data, explore_last_time, cache_path=None):
    start = time.time()
    cache_path = cache_path.replace(".csv", "_{}.csv".format(explore_last_time))
    if os.path.exists(cache_path):
        print cache_path, " cache exist"
        data = pd.read_csv(cache_path)
    else:
        if data is None:
            print "cache not exist and data is none!"
            return
        # all_preprocess_size = 0
        trans_all_times = []
        trans_cancel_times = []
        trans_no_cancel_times = []
        tran_start_to_ends = []
        trans_is_autonews = []
        trans_not_autonews = []
        trans_plan_days_maxs = []
        trans_plan_days_mins = []
        trans_plan_days_medians = []
        return_features = []
        last_cancel_to_ends = []
        last_no_cancel_to_ends = []
        last_auto_to_ends = []
        last_no_auto_to_ends = []

        trans_is_mounth_ends = []
        trans_is_month_starts = []
        exp_is_month_starts = []
        exp_is_month_ends = []
        part_dfs = load_part_transactions()
        for _printi, _part in enumerate(part_dfs):
            _part = _part[_part.transaction_date <= int(explore_last_time)]
            print "preprocess:", _printi
            trans_all_time = _part[["msno", "is_cancel"]].groupby("msno").count().reset_index()
            trans_all_time.columns = ["msno", "trans_all_time"]
            trans_all_times.append(trans_all_time)

            trans_cancel_time = _part.loc[_part.is_cancel == 1, ["msno", "is_cancel"]].groupby(
                    "msno").count().reset_index()
            trans_cancel_time.columns = ["msno", "trans_cancel_time"]
            trans_cancel_times.append(trans_cancel_time)

            trans_no_cancel_time = _part.loc[_part.is_cancel == 0, ["msno", "is_cancel"]].groupby(
                    "msno").count().reset_index()
            trans_no_cancel_time.columns = ["msno", "trans_no_cancel_time"]
            trans_no_cancel_times.append(trans_no_cancel_time)

            tran_start_to_end = _part[["msno", "transaction_date"]].groupby("msno").min().reset_index()
            tran_start_to_end["tran_start_to_end"] = tran_start_to_end.transaction_date.map(
                    lambda x: (pd.Timestamp(str(explore_last_time)) - pd.Timestamp(str(x))).days)
            tran_start_to_end = tran_start_to_end.drop("transaction_date", axis=1)
            tran_start_to_ends.append(tran_start_to_end)

            trans_is_autonew = _part.loc[_part.is_auto_renew == 1, ["msno", "is_auto_renew"]].groupby(
                    "msno").count().reset_index()
            trans_is_autonew.columns = ["msno", "is_auto_renew_count"]
            trans_is_autonews.append(trans_is_autonew)

            trans_not_autonew = _part.loc[_part.is_auto_renew == 0, ["msno", "is_auto_renew"]].groupby(
                    "msno").count().reset_index()
            trans_not_autonew.columns = ["msno", "not_auto_renew_count"]
            trans_not_autonews.append(trans_not_autonew)

            trans_plan_days_max = _part[["msno", "payment_plan_days"]].groupby("msno").max().reset_index()
            trans_plan_days_max.columns = ["msno", "trans_plan_days_max"]
            trans_plan_days_maxs.append(trans_plan_days_max)

            trans_plan_days_min = _part[["msno", "payment_plan_days"]].groupby("msno").min().reset_index()
            trans_plan_days_min.columns = ["msno", "trans_plan_days_min"]
            trans_plan_days_mins.append(trans_plan_days_min)

            trans_plan_days_median = _part[["msno", "payment_plan_days"]].groupby("msno").median().reset_index()
            trans_plan_days_median.columns = ["msno", "trans_plan_days_median"]
            trans_plan_days_medians.append(trans_plan_days_median)

            return_feature = preprocess_trans_part(_part, ["payment_method_id", "payment_plan_days",
                                                           "plan_list_price", "actual_amount_paid"])
            return_features.append(return_feature)

            last_cancel = _part[_part.is_cancel == 1].sort_values(["msno", "transaction_date"]). \
                groupby("msno", as_index=False).apply(lambda x: x.iloc[-1])[["msno", "transaction_date"]]
            last_cancel["last_cancel_to_end"] = last_cancel.transaction_date.map(
                    lambda x: (pd.Timestamp(str(explore_last_time)) - pd.Timestamp(str(x))).days)
            last_cancel = last_cancel[["msno", "last_cancel_to_end"]]
            last_cancel_to_ends.append(last_cancel)

            last_no_cancel_to_end = _part[_part.is_cancel == 0].sort_values(["msno", "transaction_date"]). \
                groupby("msno", as_index=False).apply(lambda x: x.iloc[-1])[["msno", "transaction_date"]]
            last_no_cancel_to_end["last_no_cancel_to_end"] = last_no_cancel_to_end.transaction_date.map(
                    lambda x: (pd.Timestamp(str(explore_last_time)) - pd.Timestamp(str(x))).days)
            last_no_cancel_to_end = last_no_cancel_to_end[["msno", "last_no_cancel_to_end"]]
            last_no_cancel_to_ends.append(last_no_cancel_to_end)

            last_auto_to_end = _part[_part.is_auto_renew == 1].sort_values(["msno", "transaction_date"]). \
                groupby("msno", as_index=False).apply(lambda x: x.iloc[-1])[["msno", "transaction_date"]]
            last_auto_to_end["last_auto_to_end"] = last_auto_to_end.transaction_date.map(
                    lambda x: (pd.Timestamp(str(explore_last_time)) - pd.Timestamp(str(x))).days)
            last_auto_to_end = last_auto_to_end[["msno", "last_auto_to_end"]]
            last_auto_to_ends.append(last_auto_to_end)

            last_no_auto_to_end = _part[_part.is_auto_renew == 0].sort_values(["msno", "transaction_date"]). \
                groupby("msno", as_index=False).apply(lambda x: x.iloc[-1])[["msno", "transaction_date"]]
            last_no_auto_to_end["last_no_auto_to_end"] = last_no_auto_to_end.transaction_date.map(
                    lambda x: (pd.Timestamp(str(explore_last_time)) - pd.Timestamp(str(x))).days)
            last_no_auto_to_end = last_no_auto_to_end[["msno", "last_no_auto_to_end"]]
            last_no_auto_to_ends.append(last_no_auto_to_end)

            _part["trans_is_month_end"] = pd.to_datetime(_part.transaction_date.astype(str)).dt.is_month_end
            _part["trans_is_month_start"] = pd.to_datetime(_part.transaction_date.astype(str)).dt.is_month_start
            _part["exp_is_month_end"] = pd.to_datetime(_part.membership_expire_date.astype(str)).dt.is_month_end
            _part["exp_is_month_start"] = pd.to_datetime(_part.membership_expire_date.astype(str)).dt.is_month_start

            trans_is_month_end = _part.loc[_part.trans_is_month_end == 1, ["msno", "trans_is_month_end"]]. \
                groupby("msno").count().reset_index()
            trans_is_month_end.columns = ["msno", "trans_is_month_end"]
            trans_is_mounth_ends.append(trans_is_month_end)

            trans_is_month_start = _part.loc[_part.trans_is_month_start == 1, ["msno", "trans_is_month_start"]]. \
                groupby("msno").count().reset_index()
            trans_is_month_start.columns = ["msno", "trans_is_month_start"]
            trans_is_month_starts.append(trans_is_month_start)

            exp_is_month_end = _part.loc[_part.exp_is_month_end == 1, ["msno", "exp_is_month_end"]]. \
                groupby("msno").count().reset_index()
            exp_is_month_end.columns = ["msno", "exp_is_month_end"]
            exp_is_month_ends.append(exp_is_month_end)

            exp_is_month_start = _part.loc[_part.exp_is_month_start == 1, ["msno", "exp_is_month_start"]]. \
                groupby("msno").count().reset_index()
            exp_is_month_start.columns = ["msno", "exp_is_month_start"]
            exp_is_month_starts.append(exp_is_month_start)

        trans_all_time = pd.concat(trans_all_times)
        trans_cancel_time = pd.concat(trans_cancel_times)
        trans_no_cancel_time = pd.concat(trans_no_cancel_times)
        tran_start_to_end = pd.concat(tran_start_to_ends)
        trans_is_autonew = pd.concat(trans_is_autonews)
        trans_not_autonew = pd.concat(trans_not_autonews)
        trans_plan_days_max = pd.concat(trans_plan_days_maxs)
        trans_plan_days_min = pd.concat(trans_plan_days_mins)
        trans_plan_days_median = pd.concat(trans_plan_days_medians)
        return_feature = pd.concat(return_features)
        last_cancel_to_end = pd.concat(last_cancel_to_ends)
        last_no_cancel_to_end = pd.concat(last_no_cancel_to_ends)
        last_auto_to_end = pd.concat(last_auto_to_ends)
        last_no_auto_to_end = pd.concat(last_no_auto_to_ends)
        exp_is_month_start = pd.concat(exp_is_month_starts)
        exp_is_month_end = pd.concat(exp_is_month_ends)
        trans_is_month_start = pd.concat(trans_is_month_starts)
        trans_is_mounth_end = pd.concat(trans_is_mounth_ends)
        data = pd.merge(data, trans_all_time, how="left", on="msno")
        data = pd.merge(data, trans_cancel_time, how="left", on="msno")
        data = pd.merge(data, trans_no_cancel_time, how="left", on="msno")
        data = pd.merge(data, trans_is_autonew, how="left", on="msno")
        data = pd.merge(data, trans_not_autonew, how="left", on="msno")
        data = pd.merge(data, trans_plan_days_max, how="left", on="msno")
        data = pd.merge(data, trans_plan_days_min, how="left", on="msno")
        data = pd.merge(data, trans_plan_days_median, how="left", on="msno")
        data = pd.merge(data, return_feature, how="left", on="msno")
        data = pd.merge(data, exp_is_month_start, how="left", on="msno")
        data = pd.merge(data, exp_is_month_end, how="left", on="msno")
        data = pd.merge(data, trans_is_month_start, how="left", on="msno")
        data = pd.merge(data, trans_is_mounth_end, how="left", on="msno")
        data.fillna(0, inplace=True)
        data = pd.merge(data, tran_start_to_end, how="left", on="msno")
        data = pd.merge(data, last_cancel_to_end, how="left", on="msno")
        data = pd.merge(data, last_no_cancel_to_end, how="left", on="msno")
        data = pd.merge(data, last_auto_to_end, how="left", on="msno")
        data = pd.merge(data, last_no_auto_to_end, how="left", on="msno")
        data.fillna(0, inplace=True)
        if cache_path is not None:
            data.to_csv(cache_path, index=None)

    data["is_autonew_freq"] = data.is_auto_renew_count / data.trans_all_time
    data["not_autonew_freq"] = data.not_auto_renew_count / data.trans_all_time
    data["cancel_freq"] = (data.trans_cancel_time) / data.trans_all_time
    data["no_cancel_freq"] = (data.trans_no_cancel_time) / data.trans_all_time
    data["exp_is_month_start_freq"] = (data.exp_is_month_start) / data.trans_all_time
    data["exp_is_month_end_freq"] = (data.exp_is_month_end) / data.trans_all_time
    data["trans_is_month_start_freq"] = (data.trans_is_month_start) / data.trans_all_time
    data["trans_is_month_end_freq"] = (data.trans_is_month_end) / data.trans_all_time
    data["exist_cancel"] = (data.trans_all_time != data.trans_no_cancel_time).astype(np.int)

    print "preprocess transactions take time {} s".format(time.time() - start)
    return data


def preprocess_transactions2(data, transactions, explore_last_time, cache_path=None):
    t = transactions
    start = time.time()
    cache_path = cache_path.replace(".csv", "_{}.csv".format(explore_last_time))
    if os.path.exists(cache_path):
        print cache_path, " cache exist"
        data = pd.read_csv(cache_path)
    else:
        if data is None:
            print "cache not exist and data is none!"
            return
        t = t.sort_values(by=["msno", "transaction_date"])
        trans_date_max = t.groupby("msno", as_index=False)["transaction_date"].agg({"max"}).reset_index()
        trans_date_max = trans_date_max.fillna(explore_last_time)
        tmdt = pd.to_datetime(trans_date_max["max"].astype(int).astype(str)).dt
        trans_date_max["trans_date_max_is_month_start"] = tmdt.is_month_start
        trans_date_max["trans_date_max_is_month_end"] = tmdt.is_month_end
        trans_date_max.drop("max", axis=1, inplace=True)
        data = pd.merge(data, trans_date_max, on="msno", how="left")

        exp_date_max = t.groupby("msno", as_index=False)["membership_expire_date"].agg({"max"}).reset_index()
        exp_date_max = exp_date_max.fillna(explore_last_time)
        tmdt = pd.to_datetime(exp_date_max["max"].astype(int).astype(str)).dt
        exp_date_max["exp_date_max_is_month_start"] = tmdt.is_month_start
        exp_date_max["exp_date_max_is_month_end"] = tmdt.is_month_end
        exp_date_max.drop("max", axis=1, inplace=True)
        data = pd.merge(data, exp_date_max, on="msno", how="left")

        t["plan_actual_noequal"] = (t.plan_list_price != t.actual_amount_paid)
        _x = t.groupby(["msno", "plan_actual_noequal"], as_index=False).count()[
            ["msno", "plan_actual_noequal", "transaction_date"]]
        plan_actual_noequal = _x[_x["plan_actual_noequal"] == 1][["msno", "transaction_date"]]
        plan_actual_noequal.columns = ["msno", "plan_actual_noequal_count"]
        plan_actual_equal = _x[_x["plan_actual_noequal"] == 0][["msno", "transaction_date"]]
        plan_actual_equal.columns = ["msno", "plan_actual_equal_count"]
        data = pd.merge(data, plan_actual_equal, on="msno", how="left")
        data = pd.merge(data, plan_actual_noequal, on="msno", how="left")
        data.fillna(0)

        data["plan_actual_equal_count_freq"] = data["plan_actual_equal_count"] / (
            data["plan_actual_equal_count"] + data["plan_actual_noequal_count"])
        data["plan_actual_noequal_count_freq"] = data["plan_actual_noequal_count"] / (
            data["plan_actual_equal_count"] + data["plan_actual_noequal_count"])

        sh1 = t.shift(1)
        t_sh1_share_index = (t.msno == sh1.msno).index
        _t = t[np.in1d(t.index, t_sh1_share_index)]
        _sh1 = sh1[np.in1d(sh1.index, t_sh1_share_index)]

        f_names = ["payment_method_id", "payment_plan_days",
                   "plan_list_price", "actual_amount_paid"]

        for f_name in f_names:
            f_new = "{}_noeq".format(f_name)
            _t[f_new] = _t[f_name] != _sh1[f_name]
            _x = _t.groupby(["msno", f_new], as_index=False).count()[["msno", f_new, "payment_method_id"]]
            _x_noequal = _x[_x[f_new] == 1][["msno", "payment_method_id"]]
            _x_noequal.columns = ["msno", "{}_noeq_count".format(f_name)]
            _x_equal = _x[_x[f_new] == 0][["msno", "payment_method_id"]]
            _x_equal.columns = ["msno", "{}_eq_count".format(f_name)]
            data = pd.merge(data, _x_noequal, on="msno", how="left")
            data = pd.merge(data, _x_equal, on="msno", how="left")
            data.fillna(0)
            data["{}_noeq_freq".format(f_name)] = data["{}_noeq_count".format(f_name)] / (
                data["{}_eq_count".format(f_name)] + data["{}_noeq_count".format(f_name)])
            data["{}_eq_freq".format(f_name)] = data["{}_eq_count".format(f_name)] / (
                data["{}_eq_count".format(f_name)] + data["{}_noeq_count".format(f_name)])

        _sh1 = _sh1.dropna()
        _t = _t.dropna()
        share_index = np.intersect1d(_sh1.index, _t.index)
        _t = _t[np.in1d(_t.index, share_index)]
        _sh1 = _sh1[np.in1d(_sh1.index, share_index)]
        f_names = ["transaction_date", "membership_expire_date"]
        for f_name in f_names:
            f_new = "{}_diff".format(f_name)
            _t[f_new] = (pd.to_datetime(_t[f_name].astype(int).astype(str)) - pd.to_datetime(
                    _sh1[f_name].astype(int).astype(str))).map(lambda x: x.days)
            _x = _t.groupby(["msno"], as_index=False)[f_new].agg(["max", "min", "mean", "median"]).reset_index()
            _x.columns = ["msno", "{}_max".format(f_name), "{}_min".format(f_name), "{}_mean".format(f_name),
                          "{}_median".format(f_name)]
            data = pd.merge(data, _x, on="msno", how="left")
            data.fillna(0)

        if cache_path is not None:
            if "is_churn" in data.columns:
                data.drop("is_churn", axis=1, inplace=True)
            data.to_csv(cache_path, index=None)
    print "preprocess trans2 feature take {} s".format(time.time() - start)
    return data


def preprocess_user_logs(data, logs, explore_last_time, cache_path=None):
    start = time.time()
    cache_path = cache_path.replace(".csv", "_{}.csv".format(explore_last_time))
    if os.path.exists(cache_path):
        print cache_path, " cache exist"
        data = pd.read_csv(cache_path)
    else:
        if data is None:
            print "cache not exist and data is none!"
            return
        date = pd.to_datetime(logs.date.astype(int).astype(str))
        last1_end = pd.Timestamp(explore_last_time)
        last1_start = last1_end - pd.Timedelta("30d")
        for _l in xrange(2):
            last1_start = last1_start - _l * pd.Timedelta("30d")
            last1_end = last1_end - _l * pd.Timedelta("30d")
            g = logs[(date >= last1_start) & (date <= last1_end)].groupby("msno")
            last1_mean = g.mean().drop("date", axis=1).reset_index()
            last1_max = g.max().drop("date", axis=1).reset_index()
            last1_min = g.min().drop("date", axis=1).reset_index()
            last1_median = g.median().drop("date", axis=1).reset_index()
            for df, name in [(last1_mean, "mean"), (last1_max, "max"), (last1_min, "min"), (last1_median, "median")]:
                origin_columns = df.columns
                new_columns = [origin_columns[0]]
                for i in xrange(1, len(origin_columns)):
                    new_columns.append("last_{}_month_{}_{}".format(_l + 1, name, origin_columns[i]))
                df.columns = new_columns
                data = pd.merge(data, df, how="left", on="msno")
        data.fillna(0)
        if cache_path is not None:
            if "is_churn" in data.columns:
                data.drop("is_churn", axis=1, inplace=True)
            data.to_csv(cache_path, index=None)
    print "preprocess user logs feature take {} s".format(time.time() - start)
    return data


def preprocess_user_logs2(data, logs, explore_last_time, cache_path=None):
    start = time.time()
    cache_path = cache_path.replace(".csv", "_{}.csv".format(explore_last_time))
    if os.path.exists(cache_path):
        print cache_path, " cache exist"
        data = pd.read_csv(cache_path)
    else:
        if data is None:
            print "cache not exist and data is none!"
            return
        date = pd.to_datetime(logs.date.astype(int).astype(str))
        last1_end = pd.Timestamp(explore_last_time)
        last1_start = last1_end - pd.Timedelta("7d")
        for _l in xrange(2):
            last1_start = last1_start - _l * pd.Timedelta("7d")
            last1_end = last1_end - _l * pd.Timedelta("7d")
            g = logs[(date >= last1_start) & (date <= last1_end)].groupby("msno")
            last1_mean = g.mean().drop("date", axis=1).reset_index()
            last1_max = g.max().drop("date", axis=1).reset_index()
            last1_min = g.min().drop("date", axis=1).reset_index()
            last1_median = g.median().drop("date", axis=1).reset_index()
            for df, name in [(last1_mean, "mean"), (last1_max, "max"), (last1_min, "min"), (last1_median, "median")]:
                origin_columns = df.columns
                new_columns = [origin_columns[0]]
                for i in xrange(1, len(origin_columns)):
                    new_columns.append("last_{}_week_{}_{}".format(_l + 1, name, origin_columns[i]))
                df.columns = new_columns
                data = pd.merge(data, df, how="left", on="msno")
        data.fillna(0)
        if cache_path is not None:
            if "is_churn" in data.columns:
                data.drop("is_churn", axis=1, inplace=True)
            data.to_csv(cache_path, index=None)
    print "preprocess user logs feature take {} s".format(time.time() - start)
    return data


def preprocess_user_logs3(data, logs, explore_last_time, cache_path=None):
    start = time.time()
    cache_path = cache_path.replace(".csv", "_{}.csv".format(explore_last_time))
    if os.path.exists(cache_path):
        print cache_path, " cache exist"
        data = pd.read_csv(cache_path)
    else:
        if data is None:
            print "cache not exist and data is none!"
            return
        date = pd.to_datetime(logs.date.astype(int).astype(str))
        last1_end = pd.Timestamp(explore_last_time) - pd.Timedelta("305d")
        last1_start = last1_end - pd.Timedelta("305d") - pd.Timedelta("30d")
        for _l in xrange(2):
            last1_start = last1_start - _l * pd.Timedelta("365d")
            last1_end = last1_end - _l * pd.Timedelta("365d")
            g = logs[(date >= last1_start) & (date <= last1_end)].groupby("msno")
            last1_mean = g.mean().drop("date", axis=1).reset_index()
            last1_max = g.max().drop("date", axis=1).reset_index()
            last1_min = g.min().drop("date", axis=1).reset_index()
            last1_median = g.median().drop("date", axis=1).reset_index()
            for df, name in [(last1_mean, "mean"), (last1_max, "max"), (last1_min, "min"), (last1_median, "median")]:
                origin_columns = df.columns
                new_columns = [origin_columns[0]]
                for i in xrange(1, len(origin_columns)):
                    new_columns.append("last_{}_year_{}_{}".format(_l + 1, name, origin_columns[i]))
                df.columns = new_columns
                data = pd.merge(data, df, how="left", on="msno")
        data.fillna(0)
        if cache_path is not None:
            if "is_churn" in data.columns:
                data.drop("is_churn", axis=1, inplace=True)
            data.to_csv(cache_path, index=None)
    print "preprocess user logs feature take {} s".format(time.time() - start)
    return data


def load_not_in_transactions():
    if os.path.exists("../data/cache/train_new_not_in_members.csv"):
        train_new_not_in_members = pd.read_csv("../data/cache/train_new_not_in_members.csv")
        train_new_in_members = pd.read_csv("../data/cache/train_new_in_members.csv")
        test_new_not_in_members = pd.read_csv("../data/cache/test_new_not_in_members.csv")
        test_new_in_members = pd.read_csv("../data/cache/test_new_in_members.csv")
    else:
        train = load_train()
        test = load_test()
        transactions = load_transactions(iterator=False)
        transactions_train = transactions[transactions.transaction_date <= 20170131]
        msno_in_trans = transactions.msno.unique()
        msno_in_trans_train = transactions_train.msno.unique()
        train_new = train[~np.in1d(train.msno, msno_in_trans_train)]
        test_new = test[~np.in1d(test.msno, msno_in_trans)]
        members = load_members()
        msno_in_members = members.msno.unique()
        train_new_not_in_members = train_new[~np.in1d(train_new.msno, msno_in_members)]
        train_new_in_members = train_new[np.in1d(train_new.msno, msno_in_members)]
        test_new_not_in_members = test_new[~np.in1d(test_new.msno, msno_in_members)]
        test_new_in_members = test_new[np.in1d(test_new.msno, msno_in_members)]

        train_new_not_in_members.to_csv("../data/cache/train_new_not_in_members.csv",
                                        index=False, )
        train_new_in_members.to_csv("../data/cache/train_new_in_members.csv",
                                    index=False)
        test_new_not_in_members.to_csv("../data/cache/test_new_not_in_members.csv",
                                       index=False)
        test_new_in_members.to_csv("../data/cache/test_new_in_members.csv",
                                   index=False)

    return train_new_in_members, train_new_not_in_members, test_new_in_members, test_new_not_in_members


def split(train, split_size=10, not_churn_and_churn_ratio=2):
    train_churn = train[train.is_churn == 1]
    train_not_churn = train[train.is_churn == 0]
    churn_size = train_churn.shape[0]
    not_churn_size = train_not_churn.shape[0]
    random_chooose_churn_size = not_churn_size / split_size / not_churn_and_churn_ratio
    # print not_churn_size / split_size
    # print random_chooose_churn_size
    assert churn_size > random_chooose_churn_size
    train_split = []
    for i in xrange(split_size):
        _train_n_c = train_not_churn.iloc[i * churn_size: (i + 1) * churn_size]
        _train_c = train_churn.sample(n=random_chooose_churn_size)
        _train = pd.concat([_train_n_c, _train_c])
        _train = _train.sample(n=_train.shape[0])
        train_split.append(_train)
    return train_split


def split_origin_ration(train, split_size=10):
    train_churn = train[train.is_churn == 1]
    train_not_churn = train[train.is_churn == 0]
    ratio = int(train_not_churn.shape[0] / train_churn.shape[0])
    return split(train, split_size, ratio)


def load_train_and_test():
    def merge(part1, part2):
        if part1 is None:
            return part2
        else:
            if part2 is None:
                return part1
            else:
                return pd.merge(part1, part2, on="msno", how="left")

    train = preprocess_transactions(None, "20170131", "../data/cache/train_do_trans1.csv")
    test = preprocess_transactions(None, "20170228", "../data/cache/test_do_trans1.csv")
    train_logs = preprocess_user_logs(None, None, "20170131", "../data/cache/train_do_logs1.csv")
    if "is_churn" in train_logs.columns:
        train_logs = train_logs.drop("is_churn", axis=1)
    test_logs = preprocess_user_logs(None, None, "20170228", "../data/cache/test_do_logs1.csv")
    if "is_churn" in test_logs.columns:
        test_logs = test_logs.drop("is_churn", axis=1)
    train = merge(train, train_logs)
    test = merge(test, test_logs)

    train2 = preprocess_transactions2(None, None, "20170131", "../data/cache/train_do_trans2.csv")
    test2 = preprocess_transactions2(None, None, "20170228", "../data/cache/test_do_trans2.csv")
    train = merge(train, train2)
    test = merge(test, test2)

    train2 = preprocess_user_logs2(None, None, "20170131", "../data/cache/train_do_logs2.csv")
    test2 = preprocess_user_logs2(None, None, "20170228", "../data/cache/test_do_logs2.csv")
    train = merge(train, train2)
    test = merge(test, test2)

    # train2 = preprocess_user_logs3(None, None, "20170131", "../data/cache/train_do_logs3.csv")
    # test2 = preprocess_user_logs3(None, None, "20170228", "../data/cache/test_do_logs3.csv")
    # train = merge(train, train2)
    # test = merge(test, test2)

    members = load_members()
    members["expiration_date_dayofyear"] = pd.to_datetime(members.expiration_date).dt.dayofyear
    members["expiration_date_month"] = pd.to_datetime(members.expiration_date).dt.month
    members["expiration_date_year"] = pd.to_datetime(members.expiration_date).dt.year
    train = merge(train, members)
    test = merge(test, members)
    train["gender"] = np.where(train.gender == "male", 1, np.where(train.gender == "female", 0, train.gender))
    test["gender"] = np.where(test.gender == "male", 1, np.where(test.gender == "female", 0, test.gender))
    return train, test


def filter_feature(train, threshold=0.8):
    useful_features = list(train.columns)
    if "is_churn" in useful_features:
        useful_features.remove("is_churn")
    if "msno" in useful_features:
        useful_features.remove("msno")
    choose_features = []
    all_num = train.shape[0]
    for f in useful_features:
        # print f
        vs = train[f].unique()
        if len(vs) > 8:
            choose_features.append(f)
            continue
        flag = True
        for v in vs:
            if float(train[train[f] == v][f].count()) / all_num > threshold:
                flag = False
                break
        if flag:
            choose_features.append(f)
    return choose_features


def filter_by_corrcoef(features, train, threshold=0.95):
    v = train[features].copy()
    v = v.fillna(0).values.astype(float)
    f = list(np.copy(features))
    x = np.corrcoef(v, rowvar=0)
    x = np.abs(x)
    f_num = len(features)
    for i in range(f_num):
        if features[i] in f:
            # 将线性相关强的删除
            for j in range(i + 1, f_num):
                if x[i][j] > threshold:
                    f.remove(features[j])

    return f


def feature_engineering():
    train = load_train()
    test = load_test()
    train_trans = load_train_transactions(False)
    preprocess_transactions(train, "20170131", "../data/cache/train_do_trans1.csv")
    preprocess_transactions2(train, train_trans, "20170131", "../data/cache/train_do_trans2.csv")
    del train_trans
    gc.collect()
    test_trans = load_test_transactions(False)
    preprocess_transactions(test, "20170228", "../data/cache/test_do_trans1.csv")
    preprocess_transactions2(test, test_trans, "20170228", "../data/cache/test_do_trans2.csv")
    del test_trans
    gc.collect()
    train_logs = load_train_logs(False)
    train_logs = train_logs.dropna()
    train_logs.date = train_logs.date.astype(int)
    preprocess_user_logs(load_train(), train_logs, "20170131", "../data/cache/train_do_logs1.csv")
    preprocess_user_logs2(load_train(), train_logs, "20170131", "../data/cache/train_do_logs2.csv")
    preprocess_user_logs3(load_train(), train_logs, "20170131", "../data/cache/train_do_logs3.csv")
    del train_logs
    gc.collect()
    test_logs = load_test_logs(False)
    test_logs = test_logs.dropna()
    test_logs.date = test_logs.date.astype(int)
    preprocess_user_logs(load_test(), test_logs, "20170228", "../data/cache/test_do_logs1.csv")
    preprocess_user_logs2(load_test(), test_logs, "20170228", "../data/cache/test_do_logs2.csv")
    preprocess_user_logs3(load_test(), test_logs, "20170228", "../data/cache/test_do_logs3.csv")
    del test_logs
    gc.collect()


if __name__ == '__main__':
    # test = pd.read_csv("../data/sample_submission_zero.csv")
    # test["is_churn"] = 0.1
    # save_result(test,"../result/all_0.1.csv")
    # train = pd.read_csv("../data/train.csv")
    # ids = np.union1d(train.msno, test.msno)
    # filter("../data/user_logs.csv", "../data/part_user_logs.csv", ids)
    # preprocess_transactions(load_train(), "20170131", "../data/cache/train_do_trans.csv")

    # feature_engineering()
    train = load_train()
    test = load_test()
    train_trans = load_train_transactions(False)
    preprocess_transactions(train, "20170131", "../data/cache/train_do_trans1.csv")
    preprocess_transactions(test, "20170228", "../data/cache/test_do_trans1.csv")
