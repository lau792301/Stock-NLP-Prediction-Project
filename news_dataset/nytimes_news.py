# %%
import requests
import json
import time
from os import path
import os
# %%
# Developer.nytimes
API_KEY = "qetbGTUT72FRwHGVnXTuWA5FTvl64hd1" #"AuhC063COzACEtIVYUCw4wkAS8cwvWRs"
DESK = 'technology'
DATA_BASE_PATH = 'nytimes_data/'
START_YEAR = 2016 #2010
END_YEAR = 2019 # 201901

BASE_URL = 'https://api.nytimes.com/svc/search/v2/articlesearch.json' + f"?api-key={API_KEY}&q={DESK}"


def get_date_month_range(start_year, end_year):
    output_daterange_list = []
    for year in range(start_year, end_year + 1):
        for month in range (1, 13):
            month = str(month).zfill(2)
            date = f'{year}{month}01'
            output_daterange_list.append(date)
    output_daterange_list.append('20200101')  # for 201912 data
    return output_daterange_list

date_range_list = get_date_month_range(START_YEAR,END_YEAR)

# %%
SELECTED_KEY = ["pub_date", "abstract", "headline", "section_name", "keywords"]
def processing(doc_dict):
    def keyword_processing(key_dict_list):
        output_dict = {}
        for key_dict in key_dict_list:
            name_field = key_dict['name']
            if  name_field not in output_dict:
                output_dict[name_field] = [] # init
                output_dict[name_field].append(key_dict["value"])
            else:
                output_dict[name_field].append(key_dict["value"])
        return output_dict
    selected_dict = {key:doc_dict[key] for key in SELECTED_KEY}
    selected_dict['ref_keywords'] = keyword_processing(selected_dict['keywords'])
    del selected_dict['keywords']
    return selected_dict


WATIING_TIME = 30 # 30 seconds
MAX_TRY = 5

for i in range(len(date_range_list) - 1):
    prefix_date = date_range_list[i]
    suffix_date = date_range_list[i + 1]
    requests_url = BASE_URL + f'&begin_date={prefix_date}&end_date={suffix_date}'
    for _ in range(MAX_TRY):
        try:
            response = requests.get(requests_url).json()['response']
            no_page = int((response['meta']['hits'] -1) / 10) # since each query only disply 10 records, and 10-1 = 9 = page 0
            print(f'prefix_date = {prefix_date}, suffix_date = {suffix_date}, number of pages = {no_page}')
            time.sleep(5)
            break
        except:
            print('Connection Issue Error')
            time.sleep(WATIING_TIME)

    for page in range(no_page + 1):
        # if page file existed then continue
        if path.exists(DATA_BASE_PATH + f'{prefix_date}_{suffix_date}_{page}.json') :
            continue
        page_data_list = []
        print(f'working {page}')
        page_request_url = requests_url + f'&page={page}'
        # Auto max try
        for _ in range(MAX_TRY):
            try:
                page_response = requests.get(page_request_url).json()['response']
                docs_list = page_response['docs']
                for doc in docs_list:
                    page_data_list.append(processing(doc))
                    # Save Page data in json
                with open(DATA_BASE_PATH + f'{prefix_date}_{suffix_date}_{page}.json', 'w') as json_file:
                    json.dump(page_data_list, json_file)
                time.sleep(5)
                break
            except:
                print('Connection Issue Error')
                time.sleep(WATIING_TIME)


# %%
# Combined all files into one signle file
# Load Data
all_file_list = os.listdir('nytimes_data')
full_data_list = []
for file_path in all_file_list:
    with open('nytimes_data/' + file_path) as json_file:
        loaded_data = json.load(json_file)
        full_data_list.extend(loaded_data)

len(full_data_list) #81589
# %%
# Clean duplicate data
full_txt_list = list(set([json.dumps(data) for data in full_data_list]))
len(full_txt_list) # 73377
cleaned_full_data_list = [json.loads(data) for data in full_txt_list]

# %%
# Save combined version
with open('201001_201912_nytimes_tech_news.json', 'w') as json_file:
    json.dump(cleaned_full_data_list, json_file)

# %%
