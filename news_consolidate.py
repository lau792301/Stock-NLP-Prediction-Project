# %%
import pickle
import datetime
import pandas as pd
# %%
PATH = 'news_dataset/news_dataset_v2.dat'
# %%
with open(PATH, "rb") as f:
    result_list = pickle.load(f)
# %%
news_list = []
for result in result_list:
    data_dict = {}
    data_dict['source'] = result['source']
    data_dict['rec_date'] = result['pub_date'].strftime('%Y-%m-%d')
    setiment_headline = result['sentiment_headline']
    data_dict['headline'] = max(setiment_headline, key = setiment_headline.get)
    sentiment_abstract = result['sentiment_abstract']
    data_dict['abstract'] = max(sentiment_abstract, key = sentiment_abstract.get)
    news_list.append(data_dict)

# %%
df = pd.DataFrame(news_list)

# %%
headline_count_df  = df.drop(columns = 'abstract').groupby(['source', 'rec_date']).headline.value_counts()
headline_count_df.name = 'count'
headline_count_df  = headline_count_df.reset_index()
headline_count_df = headline_count_df.rename(columns = {'headline': 'nature'})

# %%
consolidate_headline_df = pd.DataFrame()
consolidate_headline_df['rec_date'] = headline_count_df['rec_date'].unique()

for source in  headline_count_df['source'].unique():
    temp_headline_df = headline_count_df[headline_count_df['source'] == source].copy()

    for nature in  headline_count_df['nature'].unique():
        temp_nature_df = temp_headline_df[temp_headline_df['nature'] == nature].copy()
        temp_nature_df = temp_nature_df[['rec_date', 'count']]
        temp_nature_df = temp_nature_df.rename(columns = {'count': f'{source}_headline_{nature}'})

        consolidate_headline_df = consolidate_headline_df.set_index('rec_date').join(temp_nature_df.set_index('rec_date'), how = 'left').reset_index()

# replace NAN
consolidate_headline_df = consolidate_headline_df.fillna(0)

# %%
# %%
abstract_count_df  = df.drop(columns = 'headline').groupby(['source', 'rec_date']).abstract.value_counts()
abstract_count_df.name = 'count'
abstract_count_df  = abstract_count_df.reset_index()
abstract_count_df = abstract_count_df.rename(columns = {'abstract': 'nature'})

# %%
consolidate_abstract_df = pd.DataFrame()
consolidate_abstract_df['rec_date'] = abstract_count_df['rec_date'].unique()

for source in  abstract_count_df['source'].unique():
    temp_abstract_df = abstract_count_df[abstract_count_df['source'] == source].copy()

    for nature in  abstract_count_df['nature'].unique():
        temp_nature_df = temp_abstract_df[temp_abstract_df['nature'] == nature].copy()
        temp_nature_df = temp_nature_df[['rec_date', 'count']]
        temp_nature_df = temp_nature_df.rename(columns = {'count': f'{source}_abstract_{nature}'})

        consolidate_abstract_df = consolidate_abstract_df.set_index('rec_date').join(temp_nature_df.set_index('rec_date'), how = 'left').reset_index()

# replace NAN
consolidate_abstract_df = consolidate_abstract_df.fillna(0)

# %%
full_consolidate = consolidate_headline_df.set_index('rec_date').join(consolidate_abstract_df.set_index('rec_date')).reset_index()
full_consolidate.to_csv('news_cleaned.csv', index = False)
# %%
# Consolidation for 
# positive = positive + neutral
# negative = negative + compound
full_consolidate_2 = full_consolidate.copy()
for source in ['dj', 'nytimes', 'wsj']:
    for news_type in ['headline', 'abstract']:
        #positive
        full_consolidate_2[f'{source}_{news_type}_pos'] = full_consolidate_2[f'{source}_{news_type}_pos'] +  full_consolidate_2[f'{source}_{news_type}_neu']
        full_consolidate_2 = full_consolidate_2.drop(columns = f'{source}_{news_type}_neu')
        #negative
        full_consolidate_2[f'{source}_{news_type}_neg'] = full_consolidate_2[f'{source}_{news_type}_neg'] +  full_consolidate_2[f'{source}_{news_type}_compound']
        full_consolidate_2 = full_consolidate_2.drop(columns = f'{source}_{news_type}_compound')


# %%
full_consolidate_2.to_csv('news_cleaned_grouped.csv', index = False)

# %%
source_col = [col for col in full_consolidate_2.columns if 'dj' in col]
full_consolidate_2[source_col]

# %%
