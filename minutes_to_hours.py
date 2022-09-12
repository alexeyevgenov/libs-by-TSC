def minutes_to_hours(df):
    ohlc_dict = {'open': 'first',
                 'high': 'max',
                 'low': 'min',
                 'close': 'last',
                 'volume': 'sum'}
    data_hour = df.copy().set_index('date')
    data_hour = data_hour.resample('1H').apply(ohlc_dict)
    data_hour = data_hour.reset_index()
    data_hour = data_hour.dropna()
    return data_hour