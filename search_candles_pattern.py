import numpy as np
import pandas as pd
import datetime
import pickle
from src.lib.market_profile import MarketProfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from src.utils.pattern_classes import Sample
from src.utils.volume_preparation import market_profile_lib
from src.backtest.pattern import Patterns
from src.utils.time_utils import get_prev_business_day

scaler = StandardScaler()


def ohlc_scale(data, n_candles):
    result_df = data.copy()
    for shft in range(1, n_candles):
        result_df = pd.concat([result_df, data.shift(-shft)], axis=1)
    result_df = result_df.dropna()
    scaled_result = scaler.fit_transform(result_df)
    result_df_scaled = pd.DataFrame(scaled_result, index=result_df.index)
    return result_df_scaled


def train_pattern_old(scan_data, data, n_candles, ticker, error_threshold):
    patterns = []
    targets = {}
    price_change_interval_init = 5

    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern = scaler.fit_transform(pattern_ohlc).flatten()
    for i_row in range(data.shape[0] + 1 - n_candles - price_change_interval_init):
        sample_ohlc = data[i_row: (i_row + n_candles)]
        sample = scaler.fit_transform(sample_ohlc).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1
            targets_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval - 1
                # Localize data by date
                target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
                # Calculate various targets
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                targets_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = targets_list
    print(f"{len(patterns)} {scan_data} patterns found in {ticker} for {n_candles} candles")
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(f'.output/{scan_data}/patterns/{today_date}/targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_pattern(scan_data, data, n_candles, ticker, error_threshold, today_date, output_folder):
    patterns = []
    targets = {}
    price_change_interval_init = 5

    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern = scaler.fit_transform(pattern_ohlc).flatten()
    for i_row in range(data.shape[0] + 1 - n_candles - price_change_interval_init):
        sample_ohlc = data[i_row: (i_row + n_candles)]
        sample = scaler.fit_transform(sample_ohlc).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1
            targets_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval - 1
                # Localize data by date
                target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
                # Calculate various targets
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                targets_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = targets_list
    print(f"{len(patterns)} {scan_data} patterns found in {ticker} for {n_candles} candles")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(
        f'{output_folder}{scan_data}/patterns/{today_date}/targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_pattern_new(scan_data, data, n_candles, ticker, error_threshold, today_date, output_folder, mute):
    general_columns = data.columns
    patterns = []
    targets = {}
    price_change_interval_init = 5

    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern_date = pattern_ohlc.index[-1]
    pattern = scaler.fit_transform(pattern_ohlc).flatten()
    samples_ohlc = data[:data.shape[0] - price_change_interval_init]
    flattened_samples_ohlc = samples_ohlc.copy()
    # DATE OF THE PATTERN IS THE DATE OF LAST PATTERN ELEMENT
    # rename columns as last pattern candle elements. We will add new columns from left side to last element of pattern
    flattened_samples_ohlc.columns = [el + f'_{n_candles}' for el in flattened_samples_ohlc.columns]
    for i in range(1, n_candles):
        shifted_samples_ohlc = samples_ohlc.shift(i)
        flattened_samples_ohlc = shifted_samples_ohlc[general_columns].join(flattened_samples_ohlc)
        flattened_samples_ohlc.columns = [el + f'_{n_candles - i}' if el in general_columns else el for el in
                                          flattened_samples_ohlc.columns]

    # add targets to df
    list_of_values = []
    samples_ohlc['close'].rolling(price_change_interval_init).apply(
        lambda x: list_of_values.append(x.values) or 0, raw=False)
    samples_ohlc.loc[:len(list_of_values), 'list_of_closes'] = pd.Series(list_of_values).values
    samples_ohlc['target'] = samples_ohlc.apply(
        lambda x: (x['list_of_closes'] - x['open']) / x['open'], axis=1)
    # shift being used to stay last pattern candle and target on the same line
    flattened_samples_ohlc['target'] = samples_ohlc['target'].shift(-1)
    flattened_samples_ohlc = flattened_samples_ohlc.dropna()

    samples_ohlc = flattened_samples_ohlc.drop(columns=['target'])
    samples_ohlc_scaled = samples_ohlc.apply(lambda x: scaler.fit_transform(x.to_numpy().reshape(
        n_candles, len(general_columns))).flatten(), axis=1).to_frame(
        name='scaled_ohlc')  # todo: i want to scale all df without apply lambda row by row
    samples_ohlc_scaled['mse'] = samples_ohlc_scaled['scaled_ohlc'].apply(lambda x: mean_squared_error(pattern, x))

    mse_ok_df = samples_ohlc_scaled[samples_ohlc_scaled['mse'] < error_threshold]
    for ind, row in mse_ok_df.iterrows():
        ind_loc = data.index.get_loc(ind)
        sample_ohlc = data.iloc[ind_loc - n_candles + 1: ind_loc + 1]
        patterns.append(Sample(ind, sample_ohlc, row['mse']))
        targets[f'{ind}'] = [el for el in flattened_samples_ohlc['target'].loc[ind]]

    if not mute:
        print(f"{len(patterns)} {scan_data} patterns found in {ticker} for {n_candles} candles")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(
        f'{output_folder}{scan_data}/patterns/{pattern_date}/targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_pattern_new2(scan_data, data, n_candles, ticker, error_threshold, last_pattern_date, mute, save_path):
    price_change_interval_max = 5

    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern_date = pattern_ohlc.index[-1]
    if not pattern_date == last_pattern_date:
        print(f'Dates mismatch. Last data day {pattern_date}. Previous business day {last_pattern_date}. '
              f'\nType {scan_data}. N_candles {n_candles}. Ticker {ticker}')
        return
    pattern = scaler.fit_transform(pattern_ohlc.to_numpy().reshape(-1, 1))
    ohlc_data = data[:data.shape[0] - price_change_interval_max]

    # DATE OF THE PATTERN IS THE DATE OF LAST PATTERN ELEMENT
    # rename columns as last pattern candle elements. We will add new columns from left side to last element of pattern
    last_columns_group = ohlc_data.copy()
    flattened_ohlc_data = last_columns_group.add_suffix(f'_{n_candles}')
    for i in range(1, n_candles):
        shifted_columns = ohlc_data.shift(i).add_suffix(f'_{n_candles - i}')
        flattened_ohlc_data = shifted_columns.join(flattened_ohlc_data)

    # add price changes to df
    list_of_values = []
    ohlc_data['close'].rolling(price_change_interval_max).apply(
        lambda x: list_of_values.append(x.values) or 0, raw=False)
    ohlc_data.loc[:len(list_of_values), 'list_of_closes'] = pd.Series(list_of_values).values
    ohlc_data['price_change_pct'] = ohlc_data.apply(
        lambda x: (x['list_of_closes'] - x['open']) / x['open'], axis=1)

    # shift aligns last pattern candle and price change
    price_changes = ohlc_data['price_change_pct'].shift(-1)
    list_of_closes = ohlc_data['list_of_closes'].shift(-1)
    flattened_ohlc_data = flattened_ohlc_data.dropna()

    # MSE calculation
    scaled_flattened_ohlc_data = flattened_ohlc_data.apply(lambda x: scaler.fit_transform(x.to_numpy().reshape(-1, 1)),
                                                           axis=1).to_frame(name='scaled_ohlc')
    mse = scaled_flattened_ohlc_data['scaled_ohlc'].apply(lambda x: mean_squared_error(pattern, x)).rename("mse/similarity")

    flattened_ohlc_data = flattened_ohlc_data.join(mse).join(price_changes).join(list_of_closes)
    flattened_ohlc_data['n_candles'] = n_candles
    flattened_ohlc_data['pattern_type'] = scan_data
    flattened_ohlc_data['ticker'] = ticker
    flattened_ohlc_data['prototype_pattern_date'] = last_pattern_date
    flattened_ohlc_data = flattened_ohlc_data.dropna()

    if not mute:
        flattened_ohlc_data_filtered = flattened_ohlc_data[flattened_ohlc_data['mse/similarity'] <= error_threshold]
        print(f"{flattened_ohlc_data_filtered.shape[0]} {scan_data} patterns found in {ticker} for {n_candles} candles")
    pickle.dump(flattened_ohlc_data, open(f'{save_path}', 'wb'))


def market_profile_calculate(data, volume_profile_window):
    market_profile_dict = {}
    for i_row in range(data.shape[0] - volume_profile_window):
        data_piece = data[i_row:i_row + volume_profile_window].reset_index()
        start_index = data_piece.index[0]
        end_index = data_piece.index[-1]
        data_piece.columns = data_piece.columns.str.capitalize()
        mp = MarketProfile(data_piece)
        mp_slice = mp[start_index:end_index]
        poc = mp_slice.poc_price
        vah = mp_slice.value_area[1]
        val = mp_slice.value_area[0]
        index_next_row = data.reset_index().iloc[i_row + volume_profile_window].date
        market_profile_dict[f'{index_next_row}'] = [poc, vah, val]
    market_profile_df = pd.DataFrame.from_dict(market_profile_dict, columns=['poc', 'vah', 'val'], orient='index')
    columns_names = list(market_profile_df.columns)
    market_profile_df = market_profile_df.reset_index()
    market_profile_df.columns = ['date'] + columns_names
    # market_profile_df.to_csv(market_profile_df, index=False)
    return market_profile_df


def market_profile_patterns_search_old(data, n_candles, ticker, error_threshold, market_profile_filename, scan_data):
    # print(f'\nSearch {scan_data.upper()} patterns for {ticker}. {n_candles} candles')
    patterns = []
    targets = {}

    market_profile_data = pd.read_csv(market_profile_filename)
    market_profile_data['date'] = pd.to_datetime(market_profile_data['date']).dt.date
    market_profile_data = market_profile_data.set_index('date')
    all_df = data.join(market_profile_data)  # pd.concat([data, market_profile_data], axis=1)
    all_df['close_poc'] = all_df['close'] / all_df['poc']
    all_df['vah_poc'] = all_df['vah'] / all_df['poc']
    all_df['val_poc'] = all_df['val'] / all_df['poc']
    prepared_df = all_df[['close_poc', 'vah_poc', 'val_poc']].dropna()

    if data.empty:
        print('No market profile data to create patterns')
        return

    pattern = scaler.fit_transform(prepared_df[-n_candles:]).flatten()

    for i_row in range(prepared_df.shape[0] + 1 - n_candles - 5):
        sample_data = prepared_df[i_row: i_row + n_candles:]
        sample = scaler.fit_transform(sample_data).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')

        if mse < error_threshold:
            sample_index = sample_data.index[0]
            sample_ohlc = data.loc[sample_data.index]
            # Calculate target
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1

            targets_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval - 1
                target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                targets_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = targets_list
    print(f"{len(patterns)} patterns found in {ticker} for {n_candles} candles")
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(f'.output/market_profile/patterns/{today_date}/'
                      f'targets_{ticker}_{scan_data}_{n_candles}_candles.csv')

    return patterns


def market_profile_patterns_search(data, n_candles, ticker, error_threshold, market_profile_filename, scan_data,
                                   today_date, output_folder):
    # print(f'\nSearch {scan_data.upper()} patterns for {ticker}. {n_candles} candles')
    patterns = []
    targets = {}

    market_profile_data = pd.read_csv(market_profile_filename)
    market_profile_data['date'] = pd.to_datetime(market_profile_data['date']).dt.date
    market_profile_data = market_profile_data.set_index('date')
    all_df = data.join(market_profile_data)  # pd.concat([data, market_profile_data], axis=1)
    all_df['close_poc'] = all_df['close'] / all_df['poc']
    all_df['vah_poc'] = all_df['vah'] / all_df['poc']
    all_df['val_poc'] = all_df['val'] / all_df['poc']
    prepared_df = all_df[['close_poc', 'vah_poc', 'val_poc']].dropna()

    if data.empty:
        print('No market profile data to create patterns')
        return

    pattern = scaler.fit_transform(prepared_df[-n_candles:]).flatten()

    for i_row in range(prepared_df.shape[0] + 1 - n_candles - 5):
        sample_data = prepared_df[i_row: i_row + n_candles:]
        sample = scaler.fit_transform(sample_data).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')

        if mse < error_threshold:
            sample_index = sample_data.index[0]
            sample_ohlc = data.loc[sample_data.index]
            # Calculate target
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1

            targets_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval - 1
                target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                targets_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = targets_list
    print(f"{len(patterns)} patterns found in {ticker} for {n_candles} candles")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(f'{output_folder}market_profile/patterns/{today_date}/'
                      f'targets_{ticker}_{scan_data}_{n_candles}_candles.csv')

    return patterns


def market_profile_patterns_search_new(data, n_candles, ticker, error_threshold, market_profile_filename, scan_data,
                                       today_date, output_folder, mute):
    general_columns = ['close_poc', 'vah_poc', 'val_poc']
    patterns = []
    targets = {}
    price_change_interval_init = 5

    market_profile_data = pd.read_csv(market_profile_filename)
    market_profile_data['date'] = pd.to_datetime(market_profile_data['date']).dt.date
    market_profile_data = market_profile_data.set_index('date')
    all_df = data.join(market_profile_data)  # pd.concat([data, market_profile_data], axis=1)
    all_df['close_poc'] = all_df['close'] / all_df['poc']
    all_df['vah_poc'] = all_df['vah'] / all_df['poc']
    all_df['val_poc'] = all_df['val'] / all_df['poc']
    prepared_df = all_df[general_columns].dropna()

    if data.empty:
        if not mute:
            print('No market profile data to create patterns')
        return

    pattern = scaler.fit_transform(prepared_df[-n_candles:]).flatten()

    samples_ohlc = data[:data.shape[0] - price_change_interval_init]
    mprofiles = prepared_df[:prepared_df.shape[0] - price_change_interval_init]
    flattened_samples_ohlc = mprofiles
    for i in range(1, n_candles):
        shifted_samples_ohlc = mprofiles.shift(-i)
        flattened_samples_ohlc = flattened_samples_ohlc.join(shifted_samples_ohlc[general_columns], rsuffix=f'_{i}')

    # add targets to df
    # shifted_samples_ohlc = samples_ohlc.shift(-1)
    list_of_values = []
    samples_ohlc['close'].rolling(price_change_interval_init).apply(
        lambda x: list_of_values.append(x.values) or 0, raw=False)
    samples_ohlc.loc[:len(list_of_values), 'list_of_closes'] = pd.Series(list_of_values).values
    samples_ohlc['target'] = samples_ohlc.apply(
        lambda x: (x['list_of_closes'] - x['open']) / x['open'], axis=1)
    flattened_samples_ohlc['target'] = samples_ohlc['target']
    index = flattened_samples_ohlc.index
    flattened_samples_ohlc = flattened_samples_ohlc.dropna()
    flattened_samples_ohlc = flattened_samples_ohlc.set_index(index[-flattened_samples_ohlc.shape[0]:])

    samples_ohlc = flattened_samples_ohlc.drop(columns=['target'])
    samples_ohlc_scaled = samples_ohlc.apply(lambda x: scaler.fit_transform(x.to_numpy().reshape(
        n_candles, len(general_columns))).flatten(), axis=1).to_frame(
        name='scaled_mp')  # todo: i want to scale all df without apply lambda row by row
    samples_ohlc_scaled['mse'] = samples_ohlc_scaled['scaled_mp'].apply(lambda x: mean_squared_error(pattern, x))

    mse_ok_df = samples_ohlc_scaled[samples_ohlc_scaled['mse'] < error_threshold]
    for ind, row in mse_ok_df.iterrows():
        ind_loc = data.index.get_loc(ind)
        sample_ohlc = data.iloc[ind_loc - n_candles + 1:ind_loc + 1]
        patterns.append(Sample(ind, sample_ohlc, row['mse']))
        targets[f'{ind}'] = [el for el in flattened_samples_ohlc['target'].loc[ind]]

    if not mute:
        print(f"{len(patterns)} patterns found in {ticker} for {n_candles} candles")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(f'{output_folder}market_profile/patterns/{today_date}/'
                      f'targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_pattern_old_(data, n_candles, error_threshold, price_change_interval):
    patterns = []
    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern = scaler.fit_transform(pattern_ohlc).flatten()
    for i_row in range(data.shape[0] - n_candles - price_change_interval):
        sample_ohlc = data[i_row: (i_row + n_candles)]
        sample = scaler.fit_transform(sample_ohlc).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1
            end_day_row_number = next_day_row_number + price_change_interval - 1
            # Localize data by date
            target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
            # Calculate various targets
            target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                target_ohlc.open[0] != 0 else 1
            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))  # target_close_open))
    return patterns


def train_pattern_with_volume_old(data, n_candles, error_threshold, ticker):
    print(f'\nSearch OHLC & VOLUME patterns for {ticker}. {n_candles} candles')
    patterns = []
    targets = {}
    price_change_interval_init = 5

    data_ohlc = data.iloc[:, :4]
    data_volume = data.iloc[:, 4:]

    # Take n_candles pattern
    pattern_ohlc_scaled = scaler.fit_transform(data_ohlc[-n_candles:])
    pattern_volume_scaled = scaler.fit_transform(data_volume[-n_candles:])
    pattern = np.c_[pattern_ohlc_scaled, pattern_volume_scaled].flatten()

    for i_row in range(data_ohlc.shape[0] - n_candles - price_change_interval_init):
        sample_ohlc = data_ohlc[i_row: (i_row + n_candles)]
        sample_ohlc_scaled = scaler.fit_transform(sample_ohlc)

        sample_volume = data_volume[i_row: (i_row + n_candles)]
        sample_volume_scaled = scaler.fit_transform(sample_volume)

        sample = np.c_[sample_ohlc_scaled, sample_volume_scaled].flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data_ohlc.index.get_loc(sample_ohlc.index[-1]) + 1

            target_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval_init - 1
                # Localize data by date
                target_ohlc = data_ohlc.loc[data_ohlc.index[next_day_row_number]:data_ohlc.index[end_day_row_number]]
                # Calculate various targets
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                target_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = target_list
    print(f"{len(patterns)} patterns found in {ticker}")
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(
        f'.output/ohlc_n_volume/patterns/{today_date}/targets_{ticker}_ohlc_n_volume_{n_candles}_candles.csv')
    return patterns


def train_pattern_with_volume(data, n_candles, error_threshold, ticker, today_date, output_folder):
    print(f'\nSearch OHLC & VOLUME patterns for {ticker}. {n_candles} candles')
    patterns = []
    targets = {}
    price_change_interval_init = 5

    data_ohlc = data.iloc[:, :4]
    data_volume = data.iloc[:, 4:]

    # Take n_candles pattern
    pattern_ohlc_scaled = scaler.fit_transform(data_ohlc[-n_candles:])
    pattern_volume_scaled = scaler.fit_transform(data_volume[-n_candles:])
    pattern = np.c_[pattern_ohlc_scaled, pattern_volume_scaled].flatten()

    for i_row in range(data_ohlc.shape[0] - n_candles - price_change_interval_init):
        sample_ohlc = data_ohlc[i_row: (i_row + n_candles)]
        sample_ohlc_scaled = scaler.fit_transform(sample_ohlc)

        sample_volume = data_volume[i_row: (i_row + n_candles)]
        sample_volume_scaled = scaler.fit_transform(sample_volume)

        sample = np.c_[sample_ohlc_scaled, sample_volume_scaled].flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data_ohlc.index.get_loc(sample_ohlc.index[-1]) + 1

            target_list = []
            for price_change_interval in range(1, 6):
                end_day_row_number = next_day_row_number + price_change_interval_init - 1
                # Localize data by date
                target_ohlc = data_ohlc.loc[data_ohlc.index[next_day_row_number]:data_ohlc.index[end_day_row_number]]
                # Calculate various targets
                target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                    target_ohlc.open[0] != 0 else 1
                target_list.append(target_close_open)

            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
            targets[f'{sample_index}'] = target_list
    print(f"{len(patterns)} patterns found in {ticker}")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(
        f'{output_folder}ohlc_n_volume/patterns/{today_date}/targets_{ticker}_ohlc_n_volume_{n_candles}_candles.csv')
    return patterns


def train_market_profile(data, n_candles, error_threshold, price_change_interval, volume_profile_window):
    patterns = []
    # Take n_candles pattern
    pattern_ohlc = data[-n_candles:]
    pattern_poc, pattern_vah, pattern_val = market_profile_lib(data=data.reset_index(),
                                                               end_date=pattern_ohlc.index[0],
                                                               volume_profile_window=volume_profile_window)
    pattern_ohlc['close_poc'] = pattern_ohlc['close'] / pattern_poc
    pattern_ohlc['close_vah'] = pattern_ohlc['close'] / pattern_vah
    pattern_ohlc['close_val'] = pattern_ohlc['close'] / pattern_val
    pattern = scaler.fit_transform(pattern_ohlc[['close_poc', 'close_vah', 'close_val']]).flatten()
    for i_row in range(volume_profile_window,
                       data.shape[0] - n_candles - price_change_interval - volume_profile_window):
        sample_ohlc = data[i_row: (i_row + n_candles)]
        sample_poc, sample_vah, sample_val = market_profile_lib(data=data.reset_index(),
                                                                end_date=pattern_ohlc.index[0],
                                                                volume_profile_window=volume_profile_window)
        sample_ohlc['close_poc'] = sample_ohlc['close'] / sample_poc
        sample_ohlc['close_vah'] = sample_ohlc['close'] / sample_vah
        sample_ohlc['close_val'] = sample_ohlc['close'] / sample_val
        sample = scaler.fit_transform(sample_ohlc[['close_poc', 'close_vah', 'close_val']]).flatten()

        mse = mean_squared_error(pattern, sample)
        # print(f'\nfound pattern with mse = {mse}')
        if mse < error_threshold:
            # sample price change
            sample_index = sample_ohlc.index[0]
            # Calculate start and end dates for sample ohlc data
            next_day_row_number = data.index.get_loc(sample_ohlc.index[-1]) + 1
            end_day_row_number = next_day_row_number + price_change_interval - 1
            # Localize data by date
            target_ohlc = data.loc[data.index[next_day_row_number]:data.index[end_day_row_number]]
            # Calculate various targets
            target_close_open = (target_ohlc.close[-1] - target_ohlc.open[0]) / target_ohlc.open[0] if \
                target_ohlc.open[0] != 0 else 1
            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, mse))
    return patterns


def train_rsi_patterns_old(data, n_candles, ticker, similarity, scan_data, rsi_period):
    import tulipy as ti
    patterns = []
    targets = {}
    height = n_candles

    rsi = pd.Series(ti.rsi(data['close'].to_numpy(), rsi_period), name='rsi',
                    index=range(rsi_period, len(data)))
    data_with_rsi = data.reset_index().join(rsi).dropna().set_index('date')

    pattern_class = Patterns(height=height, width=n_candles, similarity=similarity, signal='rsi')
    data_with_rsi = pattern_class.prepare_patterns(data_with_rsi)

    # Take n_candles pattern
    pattern = data_with_rsi[-1:]
    pattern_weights = pattern_class.weights_matrix(pattern['codes'].item())

    list_of_indexes = data_with_rsi.index[:(n_candles - 1)].to_list()
    for ind, row in data_with_rsi[(n_candles - 1): -1].iterrows():
        list_of_indexes.append(ind)
        list_of_indexes = list_of_indexes[-n_candles:]

        actual_similarity = pattern_class.get_similarity(pattern_weights, row['codes'])
        if actual_similarity >= similarity:
            # sample price change
            sample_index = list_of_indexes[0]
            sample_ohlc = data_with_rsi[['open', 'high', 'low', 'close', 'rsi']].loc[
                          list_of_indexes[0]: list_of_indexes[-1]]
            targets_list = row[[col for col in row.index if 'price_change' in col]].to_list()
            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, actual_similarity))
            targets[f'{sample_index}'] = targets_list
    print(f"{len(patterns)} {scan_data} patterns found in {ticker} for {n_candles} candles")
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(f'.output/{scan_data}/patterns/{today_date}/targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_rsi_patterns(data, n_candles, ticker, similarity, scan_data, rsi_period, today_date, output_folder, mute):
    import tulipy as ti
    patterns = []
    targets = {}
    height = n_candles

    rsi = pd.Series(ti.rsi(data['close'].to_numpy(), rsi_period), name='rsi',
                    index=range(rsi_period, len(data)))
    data_with_rsi = data.reset_index().join(rsi).dropna().set_index('date')

    pattern_class = Patterns(height=height, width=n_candles, similarity=similarity, signal='rsi')
    data_with_rsi = pattern_class.prepare_patterns(data_with_rsi)

    # Take n_candles pattern
    pattern = data_with_rsi[-1:]
    pattern_weights = pattern_class.weights_matrix(pattern['codes'].item())

    list_of_indexes = data_with_rsi.index[:(n_candles - 1)].to_list()
    for ind, row in data_with_rsi[(n_candles - 1): -1].iterrows():
        list_of_indexes.append(ind)
        list_of_indexes = list_of_indexes[-n_candles:]

        actual_similarity = pattern_class.get_similarity(pattern_weights, row['codes'])
        if actual_similarity >= similarity:
            # sample price change
            sample_index = list_of_indexes[0]
            sample_ohlc = data_with_rsi[['open', 'high', 'low', 'close', 'rsi']].loc[
                          list_of_indexes[0]: list_of_indexes[-1]]
            targets_list = row[[col for col in row.index if 'price_change' in col]].to_list()
            # write sample to list
            patterns.append(Sample(sample_index, sample_ohlc, actual_similarity))
            targets[f'{sample_index}'] = targets_list

    if not mute:
        print(f"{len(patterns)} {scan_data} patterns found in {ticker} for {n_candles} candles")
    targets_df = pd.DataFrame.from_dict(targets, orient='index')
    targets_df.index = targets_df.index.rename('date')
    targets_df.to_csv(
        f'{output_folder}{scan_data}/patterns/{today_date}/targets_{ticker}_{scan_data}_{n_candles}_candles.csv')
    return patterns


def train_close_pct_patterns(scan_data, data, n_candles, ticker, similarity_threshold, last_pattern_date, mute,
                             save_path, height, bottom_grid_level, upper_grid_level):
    price_change_interval_max = 5

    # Take n_candles pattern
    data['previous_close'] = data['close'].shift(1)
    data['close_pct_change'] = np.log(data['close'] / data['previous_close'])
    data = data.dropna()
    pattern = data['close_pct_change'][-n_candles:]
    pattern_date = pattern.index[-1]
    if not pattern_date == last_pattern_date:
        print(f'Dates mismatch. Last data day {pattern_date}. Previous business day {last_pattern_date}. '
              f'\nType close_pct. N_candles {n_candles}. Ticker {ticker}')
        return
    # Get data without prototype pattern
    addition_shift = n_candles - price_change_interval_max if n_candles > price_change_interval_max else 0
    ohlc_data = data[:data.shape[0] - price_change_interval_max - addition_shift]
    # Get codes of patterns
    pattern_class = Patterns(height=height, width=n_candles, similarity=similarity_threshold, signal='close_pct_change')
    ohlc_data = pattern_class.prepare_close_pct_patterns(ohlc_data, bottom_grid_level, upper_grid_level)
    ohlc_calc_df = ohlc_data.copy()

    # add price changes to df
    list_of_values = []
    ohlc_calc_df['close'].rolling(price_change_interval_max).apply(
        lambda x: list_of_values.append(x.values) or 0, raw=False)
    ohlc_calc_df.loc[:len(list_of_values), 'list_of_closes'] = pd.Series(list_of_values).values
    ohlc_calc_df['price_change_pct'] = ohlc_calc_df.apply(
        lambda x: (x['list_of_closes'] - x['open']) / x['open'], axis=1)

    # shift aligns last pattern candle and price change
    price_changes = ohlc_calc_df['price_change_pct'].shift(-1)
    list_of_closes = ohlc_calc_df['list_of_closes'].shift(-1)
    ohlc_data = ohlc_data.dropna()

    pattern_code = pattern_class.get_code_close_pct(pattern, bottom_grid_level, upper_grid_level)
    pattern_weights = pattern_class.weights_matrix(pattern_code)

    # Get similarity
    similarity = ohlc_data.apply(lambda x: pattern_class.get_similarity(pattern_weights, x['codes']),
                                 axis=1).rename("mse/similarity")

    ohlc_data = ohlc_data.join(similarity).join(price_changes).join(list_of_closes)
    ohlc_data['n_candles'] = n_candles
    ohlc_data['pattern_type'] = scan_data
    ohlc_data['ticker'] = ticker
    ohlc_data['prototype_pattern_date'] = last_pattern_date
    ohlc_data = ohlc_data.dropna()

    if not mute:
        ohlc_data_filtered = ohlc_data[ohlc_data['mse/similarity'] >= similarity_threshold]
        print(f"{ohlc_data_filtered.shape[0]} {scan_data} patterns found in {ticker} for {n_candles} candles")
    pickle.dump(ohlc_data, open(f'{save_path}', 'wb'))
