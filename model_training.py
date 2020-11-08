# %%
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, UpSampling1D, TimeDistributed, Lambda, RepeatVector, Dropout
# from keras.models import Sequential
# from keras.layers import LSTM, RepeatVector, Dense, UpSampling1D, TimeDistributed, Lambda, RepeatVector, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import random
import argparse
import os

from build_dataset import get_full_data
import datetime
# %%
SEED_VALUE = 2020
## getting the arguments
parser = argparse.ArgumentParser(description='Run Cronjob')
parser.add_argument('--stock', type=str, help="stock_code of training model", required= True)
parser.add_argument('--epochs', type=int, help="Iteration of training model")
parser.add_argument('--lr', type=float, help="learing rate of training model")
args = parser.parse_args()
# Environment Variable Setting
STOCK_CODE = args.stock
EPOCHS = args.epochs if args.epochs else 100
LEARNING_RATE = args.lr if args.lr else 0.001
# %%
# N_INPUT = 7
# N_OUTPUT = 1
# EPOCHS = 50
# LEARNING_RATE = 0.001
# N_NEURONS =  100

# STOCK_CODE = 'QQQ'
# NEWS_DATA_PATH = 'news_original.csv'
# %%
# Preprocessing
def scaling(df, meta_data = False):
    df = df.copy()
    df = df.sort_values('rec_date')
    shift_df = df.shift(1).copy()
    # % Percentage Change
    # close_change, volume_change
    df['close_change'] = df['close_change'] / shift_df['close']
    df['volume_change'] = df['volume_change'] / shift_df['volume']
    df = df[1:].reset_index(drop = True).copy()
    # Min Max
    # Open, High, Low, Close, Volume, Price_gap
    meta_data_dict = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'price_gap']:
        col_info = {}
        #
        col_info['max'] = df[col].max()
        col_info['min'] = df[col].min()
        df[col] = (df[col] - col_info['min']) / (col_info['max'] - col_info['min'])
        meta_data_dict[col] = col_info
    if meta_data:
        return df, meta_data_dict
    return df

def get_feature_layer(news_col_list, news_is_cluster = False, stock = True, news = True, holiday = True, month =  True, weekday = True):
    #  FeatureColumn
    feature_columns = []
    if stock:
        # Numeric Columns
        numeric_cols = ['open', 'high', 'low', 
                        'close', 'volume', 'close_change',
                        'volume_change', 'price_gap']
        for header in numeric_cols :
            feature_columns.append(feature_column.numeric_column(header))
    if (news == True) & (news_is_cluster == False):
        for header in news_col_list:
            feature_columns.append(feature_column.numeric_column(header))
    

    # # Categorical Columns
    if (news == True) & (news_is_cluster == True):
        for header in news_col_list:
            news_type = feature_column.categorical_column_with_vocabulary_list(
                        header, [1,2,3])
            news_type_one_hot = feature_column.indicator_column(news_type)
            feature_columns.append(news_type_one_hot)
    if month:
    # Month
        month_type = feature_column.categorical_column_with_vocabulary_list(
                    'month', [month for month in range(1,13)])  #1 -12
        month_type_one_hot = feature_column.indicator_column(month_type)
        feature_columns.append(month_type_one_hot)

    if weekday:
    # Weekday
        weekday_type = feature_column.categorical_column_with_vocabulary_list(
                    'weekday', [weekday for weekday in range(1,8)])  # 1 - 7
        weekday_type_one_hot = feature_column.indicator_column(weekday_type)
        feature_columns.append(weekday_type_one_hot)

    if holiday:
    # Holiday
        holiday_type = feature_column.categorical_column_with_vocabulary_list(
                    'holiday', [1])
        holiday_type_one_hot = feature_column.indicator_column(holiday_type)
        feature_columns.append(holiday_type_one_hot)

    if stock:
        # is_closed_by_high
        is_closed_by_high_type = feature_column.categorical_column_with_vocabulary_list(
                    'is_closed_by_high', [1])
        is_closed_by_high_type_one_hot = feature_column.indicator_column(is_closed_by_high_type)
        feature_columns.append(is_closed_by_high_type_one_hot)

        # is_closed_by_low
        is_closed_by_low_type = feature_column.categorical_column_with_vocabulary_list(
                    'is_closed_by_low', [1])
        is_closed_by_low_type_one_hot = feature_column.indicator_column(is_closed_by_low_type)
        feature_columns.append(is_closed_by_low_type_one_hot)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns, trainable = False)
    return feature_layer

def build_data(x_value, y_value ,n_input, n_output):
    X, Y = list(), list()
    in_start = 0
    data_len = len(x_value)
    # step over the entire history one time step at a time
    for _ in range(data_len):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        if out_end <= data_len:
            x_input = x_value[in_start:in_end] # e.g. t0-t3
            X.append(x_input)
            y_output = y_value[in_end:out_end] # e.g. t4-t5
            Y.append(y_output)
        # move along one time step
        in_start += 1
    return np.array(X), np.array(Y)            

def get_meta_index(df, n_input): #, n_ouput  = N_OUTPUT):
    df = df.copy().reset_index(drop = True)
    meta_dict = {}
    # Offline Index List
    offline_index_list = df[(df['holiday'] == 1) |  (df['weekday'].isin([6,7]))].index
    offline_index_list = [index - n_input for index in offline_index_list if (index >= n_input) ]
    meta_dict['offline_index'] = offline_index_list
    # Month Index Dict of list 
    month_index_dict = {inc_month: df[df['inc_month'] == inc_month].index - 30 for inc_month in df['inc_month'].unique()}
    meta_dict['month_index'] = month_index_dict
    return  meta_dict

def filter_offline(value, offline_index_list, month_index_dict):
    filtered_inc_month_value_dict = {}
    for inc_month, index_list in month_index_dict.items():
        filtered_month_index_list = []
        for month_index in index_list:
            if month_index not  in offline_index_list:
                filtered_month_index_list.append(value[month_index])
        filtered_inc_month_value_dict[inc_month] = np.array(filtered_month_index_list)
    return  filtered_inc_month_value_dict
    
def integrate_month_value(inc_month_value_dict, split_month = 12):
    inc_month_key_list = list(inc_month_value_dict.keys())
    # Train
    train_value = []
    for inc_month in inc_month_key_list[:-split_month]:
        train_value.extend(inc_month_value_dict[inc_month])
    # Test
    test_value = []
    for inc_month in inc_month_key_list[-split_month:]:
        test_value.extend(inc_month_value_dict[inc_month])
    return np.array(train_value), np.array(test_value)

# %%
# Model Part
def build_model(n_inputs, n_features, n_outputs, n_neurons, 
                auto_encoder = True, n_repeat = 3, learning_rate = LEARNING_RATE, dropout = 0.2):
    # def self_measurement(y_true, y_pred):
    #     return tf.keras.backend.mean(y_pred)

    # define model
    model = Sequential()
    '''
    # LSTM AutoEncoder (Unsupervised Learning)
    # https://machinelearningmastery.com/lstm-autoencoders/
    https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    '''
    if auto_encoder:
        # model.add(LSTM(n_neurons, input_shape = (n_inputs, n_features), return_sequences=False))
        # model.add(RepeatVector(n_inputs))
        # model.add(LSTM(n_neurons, return_sequences=True))
        model.add(LSTM(128, input_shape = (n_inputs, n_features), return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(RepeatVector(n_inputs))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(LSTM(n_neurons, return_sequences=True))
    else:
        model.add(LSTM(n_neurons, input_shape = (n_inputs, n_features), return_sequences=True, 
                    kernel_initializer='random_uniform', bias_initializer='zeros'))
    '''
    return_sequences default: False, return single hidden value (2 dimension)
                            True, return all time step hidden value. It also return lstm1, state_h, state_c.
                        
    '''
    if  dropout > 0:
        model.add(Dropout(dropout))
    # Dimension Adjustment
    model.add(UpSampling1D(n_outputs))
    model.add(Lambda(lambda x: x[:,-n_outputs:,:]))
    # https://stackoverflow.com/questions/55532683/why-is-timedistributed-not-needed-in-my-keras-lstm
    # model.add(Dense(1))

    model.add(TimeDistributed(Dense(1, activation = 'sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros')))
    opt = tf.keras.optimizers.Adam(learning_rate= learning_rate)
    model.compile(loss="binary_crossentropy", optimizer= opt,
                metrics = ['accuracy'])#,  self_measurement])
    return model

def train_model(model, X, Y, valid_split = 0.1, shuffle = True, random_state = SEED_VALUE, epochs = EPOCHS,
                early_stop =  False):
    # Set Seed
    '''
    Keras gets its source of randomness from the NumPy random number generator.
    '''
    # https://medium.com/@bc165870081/keras-model%E7%9A%84%E5%AF%A6%E9%A9%97%E5%9C%A8%E7%8F%BE%E6%80%A7-f0c926af4634
    # os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tf.random.set_seed(SEED_VALUE)
    # tf.compat.v1.set_random_seed(SEED_VALUE)

    train_x, valid_x, train_y, valid_y =  train_test_split(X, Y, shuffle = shuffle, random_state = random_state, test_size = valid_split)
    # simple early stopping
    if early_stop:
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience = min(int(EPOCHS * 0.1), 50))
        history  = model.fit(train_x, train_y, validation_data = (valid_x, valid_y), epochs=epochs, callbacks = [es])
    else:
        history  = model.fit(train_x, train_y, validation_data = (valid_x, valid_y), epochs=epochs)
    # history = model.fit(X, Y, validation_split= valid_split, shuffle = shuffle, epochs=epochs)#, callbacks = [es])
    return model, history


# %%
def run(stock_code, news_type, news_is_cluster, n_input, n_neurons,
    scale = True,
    auto_encoder  = False, learning_rate =  LEARNING_RATE, 
    include_news = True, training_valid_split = 0.1, dropout = 0.2,
    split_month = 12):
    ###### Init ######
    output_dict = {}
    result_dict = {}
    output_dict['meta_data'] = {
        'stock_code' : stock_code,
        'news_type' : news_type,
        'news_is_cluster' : news_is_cluster,
        'scale' : scale,
        'n_neurons' : n_neurons,
        'auto_encoder': auto_encoder,
        'learning_rate' : learning_rate,
        'n_input' : n_input,
        'include_news' : include_news,
        'training_valid_split': training_valid_split,
        'split_month': split_month,
        'dropout': dropout
    }
    print(output_dict['meta_data'])
    ######  Read Data ######
    # new_df = pd.read_csv(NEWS_DATA_PATH)
    df = get_full_data(stock_code, news_type, news_is_cluster = news_is_cluster)
    df['rec_date'] = pd.to_datetime(df['rec_date'])
    df['inc_month'] = (df['rec_date'].dt.year).astype(str) + (df['rec_date'].dt.month).astype(str).apply(lambda x:x.zfill(2))
    print('Read Data Done')
    if scale:
        print('Data is scaled')
        df = scaling(df)
    print(df)
    
    ###### Preprocessing ######
    news_cols = [col for col in df.columns if ('headline'  in col) or ('abstract' in col)]
    feature_layer = get_feature_layer(news_col_list = news_cols , news = include_news)
    X, Y = feature_layer(dict(df.drop(columns = 'rec_date'))).numpy(), df['up_down'].values.reshape(-1,1)
    # Index Filtering for only online data
    meta_index_dict = get_meta_index(df, n_input)
    offline_index_list =  meta_index_dict['offline_index']
    month_index_dict = meta_index_dict['month_index']

    T_X, T_Y = build_data(X , Y, n_input = n_input, n_output = 1)
    online_monthly_x_value_dict = filter_offline(T_X, offline_index_list, month_index_dict)
    online_monthly_y_value_dict = filter_offline(T_Y, offline_index_list, month_index_dict)
    # Integration and Split For Training and Testing Dataset
    train_x, test_x = integrate_month_value(online_monthly_x_value_dict, split_month = split_month)
    train_y, test_y = integrate_month_value(online_monthly_y_value_dict, split_month = split_month)

    ###### Model ######
    # Model Pre-setting
    n_inputs, n_features  = train_x.shape[1], train_x.shape[2]
    n_outputs = train_y.shape[1]

    # Model Setup
    model = build_model(n_inputs = n_inputs,
                        n_features = n_features, n_outputs = n_outputs, auto_encoder= auto_encoder,n_repeat  = 10,
                        learning_rate = learning_rate, n_neurons = n_neurons, dropout= dropout)
    # Model Training
    model, history = train_model(model , train_x, train_y, valid_split= training_valid_split)
    output_dict['model'] = model
    output_dict['history'] = history
    # Result Measurement
    result_dict['overall_training'] = model.evaluate(train_x, train_y)[1]
    result_dict['overall_testing'] =  model.evaluate(test_x, test_y)[1]
    print('Overall Training Result', result_dict['overall_training'])
    print('Overall Testing Result', result_dict['overall_testing'])
    # By month measurement
    test_month_list = list(online_monthly_x_value_dict.keys())[-split_month:]
    for inc_month in test_month_list:
        result_dict[inc_month] = model.evaluate(online_monthly_x_value_dict[inc_month],
                                                online_monthly_y_value_dict[inc_month])[1]
        print(f'{inc_month} testing accuarcy:',  result_dict[inc_month])
    output_dict['results'] = result_dict
    return output_dict
        
# %%
# output_dict = run()
# %%
# result_df = pd.DataFrame()
# for stock_code in ['QQQ']:
#     for news_type in ['original', 'original_grouped']:
#         for include_news in [True , False]:
#             for news_is_cluster in [True, False]:
#                 for n_input in [7,14,21,30]:
#                     for dropout in [0, 0.2, 0.4]:
#                         for is_auto_encoder in [True, False]:
#                             model_result_dict = run(stock_code = stock_code, 
#                                                 news_type = news_type, news_is_cluster = news_is_cluster,
#                                                 include_news = include_news,
#                                                 auto_encoder= is_auto_encoder,
#                                                 n_input = n_input, dropout=dropout)
#                             result_dict = model_result_dict['results']
#                             meta_dict = model_result_dict['meta_data']
#                             result_dict.update(meta_dict)
#                             partial_df = pd.DataFrame([result_dict])
#                             result_df = result_df.append(partial_df)


# %%
def main(stock_code):
    overall_result_df = pd.DataFrame()
    overall_model_history = pd.DataFrame()
    for news_type in ['original', 'original_grouped', 
                        'cleaned', 'cleaned_grouped']:
        for include_news in [True , False]:
            for news_is_cluster in [True, False]:
                for n_input in [7,14,21,30]:
                    for dropout in [0, 0.2, 0.4]:
                        for n_neurons in [50, 100, 150, 200]:
                            for is_auto_encoder in [True, False]:
                                model_result_dict = run(stock_code = stock_code, 
                                                    news_type = news_type, news_is_cluster = news_is_cluster,
                                                    include_news = include_news, n_neurons = n_neurons,
                                                    auto_encoder= is_auto_encoder,
                                                    n_input = n_input, dropout=dropout)
                                # Common
                                meta_dict = model_result_dict['meta_data']
                                # Model Result
                                result_dict = model_result_dict['results']
                                result_dict.update(meta_dict)
                                result_partial_df = pd.DataFrame([result_dict])
                                overall_result_df = overall_result_df.append(result_partial_df)
                                # Model History
                                model_history = model_result_dict['history'].history
                                model_history.update(meta_dict)
                                model_partial_df = pd.DataFrame(model_history)
                                model_partial_df.index = model_partial_df.index.set_names('epochs')
                                model_partial_df = model_partial_df.reset_index(drop = False)
                                overall_model_history = overall_model_history.append(model_partial_df)
    ### Save ###
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    stock_code = stock_code.upper()
    # Result
    overall_result_df.to_csv(f'{current_time}_{stock_code}_result.csv', index = False)
    # Model History
    overall_model_history.to_csv(f'{current_time}_{stock_code}_model_history.csv', index = False)
    print('Training and Tuning are Done')

# %%
# pd.set_option('display.max_columns', None)
# columns_selection = [
#     'stock_code', 'n_input', 'include_news', 'dropout',
#     'overall_training', 'overall_testing',
#     '201901', '201902','201903','201904','201905','201906','201907','201908', '201909',
#     '201910', '201911', '201912',
#     ]
# result_df2 = result_df[columns_selection].round(2).T
# result_df2.to_csv('QQQ_interim.csv')
# %%
if __name__ == '__main__':
    print('STOCK_CODE', STOCK_CODE)
    print('EPOCHS:', EPOCHS)
    print('LEARNING_RATE:', LEARNING_RATE)
    main(STOCK_CODE)
