import numpy as np
import tulipy as ti

import warnings
warnings.filterwarnings("ignore")

class FeatureGenerator:
    def __init__(self):
        self.data = None
        self.columns = []
        self.required_bars = 18
        self.date = []

    @staticmethod
    def candle_attitude(frame):
        if (frame.high == frame.prev_high) & (frame.low == frame.prev_low):
            return 'equal'
        elif (frame.high > frame.prev_high) & (frame.low >= frame.prev_low):
            return 'higher'
        elif (frame.high <= frame.prev_high) & (frame.low < frame.prev_low):
            return 'lower'
        elif ((frame.high <= frame.prev_high) & (frame.low > frame.prev_low)) | \
                ((frame.high < frame.prev_high) & (frame.low >= frame.prev_low)):
            return 'inside'
        elif (frame.high > frame.prev_high) & (frame.low < frame.prev_low):
            return 'outside'

    def generate(self, df):
        _initial_columns = df.columns
        data = df.copy()
        # Candle size, body size, lower and upper shadows, define color of candle and shadows ratio to candle size
        data['candle_size'] = (data['high'] - data['low'])        #/ frame['open']
        data['body_size'] = abs(data['close'] - data['open'])       #/ frame['open']
        data['lower_shadow'] = data.apply(lambda x: min(x['close'], x['open']) - x['low'], axis=1)
        data['upper_shadow'] = data.apply(lambda x: x['high'] - max(x['close'], x['open']), axis=1)
        data['color'] = np.sign(data['close'] - data['open'])
        data['body_candle_size'] = data['body_size'] / data['candle_size']
        data['lower_shadow_size'] = data['lower_shadow'] / data['candle_size']
        data['upper_shadow_size'] = data['upper_shadow'] / data['candle_size']
        # Ratios of previous candle to current
        data['prev_high'] = data.high.shift(1)
        data['prev_low'] = data.low.shift(1)
        # attitude_features
        data['attitude'] = data.apply(lambda x: self.candle_attitude(x), axis=1)
        data['is_equal'] = data['attitude'] == 'equal'
        data['is_higher'] = data['attitude'] == 'higher'
        data['is_lower'] = data['attitude'] == 'lower'
        data['is_inside'] = data['attitude'] == 'inside'
        data['is_outside'] = data['attitude'] == 'outside'
        ema9 = ti.ema(data['close'].values, 9)
        data['ema9diff'] = data['close'] - ema9
        rsi14 = ti.rsi(data['close'].values, 14)
        data['rsi14'] = np.concatenate([np.zeros(len(data) - len(rsi14)), np.array(rsi14)])
        service_columns = ['attitude', 'prev_high', 'prev_low'] + list(_initial_columns)

        # Select columns
        self.columns = [col for col in data.columns if col not in service_columns]
        self.data = data[self.columns].dropna()

        # SORT DATA BY TYPE
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
        self.categorical_columns = [cname for cname in self.data.columns if
                                    self.data[cname].nunique() < 10 and
                                    self.data[cname].dtype == "object"]
        # Select numerical columns
        self.numerical_columns = [cname for cname in self.data.columns if
                                  self.data[cname].dtype in ['int64', 'float64']]
        # Select boolean columns
        self.boolean_columns = [cname for cname in self.data.columns if
                                self.data[cname].dtype in ['bool']]

        return self.data
