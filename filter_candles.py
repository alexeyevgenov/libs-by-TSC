def all_candles_percentage(df_with_generated_features, percentage):
    big_candles_df = df_with_generated_features.sort_values(by=['candle_size'],
                                    ascending=False).iloc[:int(df_with_generated_features.shape[0] * percentage)]
    return big_candles_df

def max_candle_percentage(df_with_generated_features, percentage):
    big_candles_df = df_with_generated_features[df_with_generated_features['candle_size'] >   \
                                                percentage*df_with_generated_features['candle_size'].max()]
    return big_candles_df

def select_upper_outliers(df_with_generated_features):
    Q1 = df_with_generated_features.candle_size.quantile(0.25)
    Q3 = df_with_generated_features.candle_size.quantile(0.75)
    IQR = Q3 - Q1
    mx_disp = Q3 + 1.5 * IQR
    big_candles_df = df_with_generated_features[df_with_generated_features.candle_size > mx_disp]
    return big_candles_df, mx_disp

def candle_mean_plus_std(df_with_generated_features):
    big_candle_threshold = df_with_generated_features.candle_size.mean() + df_with_generated_features.candle_size.std()
    big_candles = df_with_generated_features[df_with_generated_features.candle_size > big_candle_threshold]
    return big_candles, big_candle_threshold
