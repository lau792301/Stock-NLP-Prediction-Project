import math
import time

import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Stock(object):
    def __init__(self, code):
        self.code = code
        self.data_frame = None
        self.start = None
        self.end = None

    def get_data_frame(self, start, end=datetime.now()):
        if start != self.start or end != self.end:
            self.start = start
            self.end = end
            yf.pdr_override()

            for x in range(5):
                data_frame = pdr.get_data_yahoo(self.code, start=start, end=end)
                if data_frame is not None and len(data_frame.index) > 0:
                    break

            def calc_change(row):
                index = data_frame.index.get_loc(row.name)
                if index == 0:
                    return 0
                prev_row = data_frame.iloc[index - 1]
                return (row['Close'] - prev_row['Close']) / prev_row['Close'] * 100

            def calc_volume_change(row):
                index = data_frame.index.get_loc(row.name)
                if index == 0:
                    return 0
                prev_row = data_frame.iloc[index - 1]
                return (row['Volume'] - prev_row['Volume']) / prev_row['Volume'] * 100

            data_frame['change'] = data_frame.apply(calc_change, axis=1)
            data_frame['volume_change'] = data_frame.apply(calc_volume_change, axis=1)

            self.data_frame = data_frame.rename_axis('Date').reset_index()
        return self.data_frame


## Usage
## code = "AAPL"
## stock = Stock(code)
## data_frame = stock.get_data_frame(start=datetime.now()-relativedelta(years=3), end=datetime.now())