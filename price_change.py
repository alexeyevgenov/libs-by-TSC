import warnings
warnings.filterwarnings("ignore")
import pandas as pd


def generate_price_change(df, window, price_change_threshold):
    # PRICE_CHANGE FEATURE GENERATION
    train_data = df.copy()
    train_data['price_change'] = 0
    for row in range(train_data.shape[0] - window - 1):
        low_threshold = train_data.close.iloc[row] * (1 - price_change_threshold)
        high_threshold = train_data.close.iloc[row] * (1 + price_change_threshold)
        for win in range(window):
            if train_data['low'].iloc[row + win+1] < low_threshold:
                train_data['price_change'].iloc[row] = -1
                break
            if train_data['high'].iloc[row + win+1] > high_threshold:
                train_data['price_change'].iloc[row] = 1
                break

    return train_data['price_change']

def generate_max_price_change(df, window, price_change_threshold):
    # PRICE_CHANGE FEATURE GENERATION
    train_data = df.copy()
    train_data['price_change'] = 0
    for row in range(train_data.shape[0] - window - 1):
        low_threshold = train_data.close.iloc[row] * (1 - price_change_threshold)
        high_threshold = train_data.close.iloc[row] * (1 + price_change_threshold)
        max_low_difference = max_high_difference = 0
        # low_difference = high_difference = 0
        for win in range(window):
            if train_data['low'].iloc[row + win+1] < low_threshold:
                low_difference = abs(train_data['low'].iloc[row + win + 1] - low_threshold)
                max_low_difference = low_difference if low_difference > max_low_difference else max_low_difference
            # else:
            #     low_difference = 0

            if train_data['high'].iloc[row + win+1] > high_threshold:
                high_difference = train_data['high'].iloc[row + win+1] - high_threshold
                max_high_difference = high_difference if high_difference > max_high_difference else max_high_difference
            # else:
            #     high_difference = 0

            # if low_difference == high_difference == 0:
            #     continue

            # max_low_difference = low_difference if low_difference > max_low_difference else max_low_difference
            # max_high_difference = high_difference if high_difference > max_high_difference else max_high_difference

        train_data['price_change'].iloc[row] = 1 if max_high_difference > max_low_difference \
            else -1 if max_high_difference < max_low_difference else 0

    return train_data['price_change']

def compute_price_diff(row, price_change_threshold):
    diff = row.close - row.open
    if abs(diff) < price_change_threshold:
        return 0
    elif diff > 0:
        return 1
    else:
        return -1


def day_price_change(df, price_change_threshold):
    train_data = df.copy()
    train_data['price_change'] = train_data.apply(lambda x: compute_price_diff(x, price_change_threshold), axis=1 )
    return train_data['price_change']



#if __name__ == '__main__':

    #df_ohlc = pd.read_csv('data/eurusd_4100_days.csv')
    #df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])
    #df_ohlc = df_ohlc.set_index('date')
    #window = 1
    #price_change_threshold = 0.0000000001

    #target = day_price_change(df_ohlc, price_change_threshold)
    #print(target)