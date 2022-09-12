import numpy as np
from src.utils.pattern_classes import ChartParamsNew, ChartParams


def patterns_mean_std_calculate(targets_df, price_change_interval, min_price_change):
    targets_list = list(targets_df[f'{price_change_interval - 1}'])   # todo: added "round(,3)"
    mean = np.mean(targets_list)
    std = np.std(targets_list)
    # el > 0 - first condition is to define action of prediction. The condition abs(el) > min_price_change is
    # to estimate what prediction value is significant and what value can be defined as null
    positive_targets = [el for el in targets_list if el > 0]   # & (abs(el) > min_price_change)
    # positive_targets_1 = [el for el in targets_list if el > 0]
    positive_count = len(positive_targets)
    positive_mean = np.mean(positive_targets)
    negative_targets = [el for el in targets_list if el < 0]   # & (abs(el) > min_price_change))
    # negative_targets_1 = [el for el in targets_list if el < 0]
    negative_count = len(negative_targets)
    negative_mean = np.mean(negative_targets)
    pos_neg_ratio = [round(positive_count / (positive_count + negative_count) * 100),
                     round(negative_count / (positive_count + negative_count) * 100)] if (
                                            positive_count + negative_count) != 0 else [0, 0]
    null_targets = [el for el in targets_list if abs(el) <= min_price_change]
    nulls_count = len(null_targets)
    return ChartParamsNew(len(targets_list), mean, std,
                          positive_count, positive_mean, negative_count, negative_mean, nulls_count,
                          pos_neg_ratio, targets_list)


def patterns_mean_std_calculate_old(targets_df, price_change_interval):
    targets_list = list(targets_df[f'{price_change_interval - 1}'])
    mean = np.mean(targets_list)
    std = np.std(targets_list)
    positive_targets = [el for el in targets_list if el > 0]
    positive_count = len(positive_targets)
    positive_mean = np.mean(positive_targets)
    negative_targets = [el for el in targets_list if el < 0]
    negative_count = len(negative_targets)
    negative_mean = np.mean(negative_targets)
    pos_neg_ratio = [round(positive_count / (positive_count + negative_count) * 100),
                     round(negative_count / (positive_count + negative_count) * 100)] if (
                                            positive_count + negative_count) != 0 else [0, 0]
    action = 'BUY' if pos_neg_ratio[0] > 70 else ('SELL' if pos_neg_ratio[1] > 70 else '-')
    change = 'not much up' if (action == 'BUY') & (0.005 < mean * 100 < 1) else (
        'not much down' if (action == 'SELL') & (-1 < mean * 100 < -0.005) else (
            'jump' if (action == 'BUY') & (mean * 100 >= 1) else (
                'fall' if (action == 'SELL') & (mean * 100 < -1) else '-')))
    return ChartParams(len(targets_list), mean, std, positive_count, positive_mean, negative_count, negative_mean,
                       pos_neg_ratio, action, change, targets_list)


def patterns_mean_std_calculate_new(price_changes_pct_df, price_change_interval, min_price_change):
    targets_list = round(price_changes_pct_df[price_change_interval], 3).to_list()   # todo: added "round(,3)"
    mean = np.mean(targets_list)
    std = np.std(targets_list)
    # el > 0 - first condition is to define action of prediction. The condition abs(el) > min_price_change is
    # to estimate what prediction value is significant and what value can be defined as null
    positive_targets = [el for el in targets_list if el > 0]   # & (abs(el) > min_price_change)
    # positive_targets_1 = [el for el in targets_list if el > 0]
    positive_count = len(positive_targets)
    positive_mean = np.mean(positive_targets)
    negative_targets = [el for el in targets_list if el < 0]   # & (abs(el) > min_price_change))
    # negative_targets_1 = [el for el in targets_list if el < 0]
    negative_count = len(negative_targets)
    negative_mean = np.mean(negative_targets)
    pos_neg_ratio = [round(positive_count / (positive_count + negative_count) * 100),
                     round(negative_count / (positive_count + negative_count) * 100)] if (
                                            positive_count + negative_count) != 0 else [0, 0]
    null_targets = [el for el in targets_list if abs(el) <= min_price_change]
    nulls_count = len(null_targets)
    return ChartParamsNew(len(targets_list), mean, std,
                          positive_count, positive_mean, negative_count, negative_mean, nulls_count,
                          pos_neg_ratio, targets_list)
