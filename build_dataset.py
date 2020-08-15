# %%
import yfinance as yf
from pandas_datareader import data as pdr


import pandas as pd
import holidays

# %%
START_DATE = '2000-01-01'
END_DATE = '2019-12-31'
NEWS_DATA_PATH = 'consolidated_news.csv'
# %%

## Usage
## code = "AAPL"
## stock = Stock(code)
## data_frame = stock.get_data_frame(start=datetime.now()-relativedelta(years=3), end=datetime.now())

def get_finance_data(code, start_date, end_date):
    yf.pdr_override()

    data_frame = pdr.get_data_yahoo(code, start=start_date, end=end_date)

    data_frame = data_frame.rename_axis('rec_date').reset_index()
    data_frame = data_frame.rename(columns = {'Open': 'open', 'High':'high', 'Low' :'low', 'Close': 'close',
                                              'Adj Close': 'adj_close', 'Volume': 'volume'})
    #Drop Columns
    data_frame = data_frame.drop(columns = ['adj_close'])

    return data_frame  

# %%

def fill_missing(df):
    df = df.copy()
    # Fill Date data and replace missing data by preceding data
    min_date = df['rec_date'].min()
    max_date = df['rec_date'].max()
    date_range = pd.date_range(min_date, max_date)

    df = df.set_index('rec_date')
    df = df.reindex(date_range)
    df = df.fillna(method = 'ffill')
    df.index.name = 'rec_date'
    df = df.reset_index(drop = False)
    return df
        

def preprocessing(df, alpha = 0.01):
    def get_holiday(start_year, end_year, country = 'us'):
        output_date_list = []
        us_holiday = holidays.CountryHoliday('US')
        year_len = end_year - start_year + 1
        for _ in range(year_len):
            output_date_list.extend(us_holiday[f'{start_year}-01-01':f'{start_year}-12-31'])
            start_year  += 1
        return output_date_list

    df = df.copy()
    ##### Add Feature #####
    # Weekday # Monday = 1, Sunday = 7
    df['weekday'] = df.rec_date.dt.weekday + 1
    # Month
    df['month'] = df.rec_date.dt.month

    # Holiday
    us_holiday_list = get_holiday(2000, 2020)
    df['holiday'] = df['rec_date'].apply(lambda x: 1 if x in us_holiday_list else 0)

    
    def get_change(df):
        df = df.copy()
        shift_df = df.shift(1).copy()
        # Get Change
        df['close_change'] = (df['close'] - shift_df['close'])
        df['volume_change'] = (df['volume'] - shift_df['volume'])
        df = df.iloc[1:].reset_index(drop = True)
        return df
    df =  get_change(df)
    df['price_gap'] = df['high'] - df['low']
    df['is_closed_by_high'] = (df['close'] * (1 + alpha) >= df['high']).astype(int)
    df['is_closed_by_low'] = (df['close'] <= df['low'] * (1  + alpha)).astype(int)
    df['up_down'] = (df['close_change'] >= 0).astype(int)
    return df

def consoildate(stock_df, new_df):
    stock_df = stock_df.copy()
    new_df = new_df.copy()

    stock_df = stock_df[stock_df['rec_date'] < '2019-12-31']
    combined_df = stock_df.set_index('rec_date').join(new_df.set_index('rec_date'), how ='inner')
    return combined_df


def get_full_data(stock_code, new_df, start_date = START_DATE, end_date = END_DATE, save = False):
    stock_data = get_finance_data(stock_code, start_date = start_date, end_date = end_date)
    stock_data = fill_missing(stock_data)
    pre_stock_data = preprocessing(stock_data)
    full_data = consoildate(pre_stock_data, new_df)
    if save:
        full_data.to_csv(f'{stock_code}_full_dataset.csv', index = False)
    return full_data


news_data = pd.read_csv(NEWS_DATA_PATH)

get_full_data('QQQ', news_data, save  = True)
get_full_data('AAPL', news_data, save  = True)
# full_dataset = get_full_data('QQQ', news_data, save  = True)

# stock_data = get_finance_data('AAA', start_date = START_DATE, end_date = END_DATE)
# stock_data = fill_missing(stock_data)
# pre_stock_data = preprocessing(stock_data)

# %%

# %%
