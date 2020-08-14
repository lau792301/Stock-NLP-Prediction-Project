# %%
import json
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# %%
def load_json(path):
    data = []
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def nytimes_analysis(nytimes, analyzer):
    result = []
    count = 0

    for news in nytimes:
        try:
            pub_date = datetime.strptime(news['pub_date'], '%Y-%m-%dT%H:%M:%S%z')
            sentiment_headline = analyzer.polarity_scores(news['headline']['main'])
            sentiment_abstract = analyzer.polarity_scores(news['abstract'])

            data = dict(
                pub_date=pub_date,
                headline=news['headline']['main'],
                abstract=news['abstract'],
                source="nytimes",
                sentiment_headline=sentiment_headline,
                sentiment_abstract=sentiment_abstract
            )
            result.append(data)
            count += 1
            if count % 1000 == 0:
                print("%s %s/%s" % ("nytimes", count, len(nytimes)))
        except Exception as ex:
            print("%s-%s" % (news, ex))

    return result


def dj_wsj_analysis(data_list, analyzer, type):
    result = []
    count = 0

    for news in data_list:
        try:
            pub_date = datetime.strptime(news['date'], '%d %B %Y')
            sentiment_headline = analyzer.polarity_scores(news['header'])
            sentiment_abstract = analyzer.polarity_scores(news['abstract'])

            data = dict(
                pub_date=pub_date,
                headline=news['header'],
                abstract=news['abstract'],
                source=type,
                sentiment_headline=sentiment_headline,
                sentiment_abstract=sentiment_abstract
            )
            result.append(data)
            count += 1
            if count % 1000 == 0:
                print("%s %s/%s" % (type, count, len(data_list)))
        except Exception as ex:
            print("%s-%s" % (news, ex))

    return result


def main():
    nytimes_path = "./data/201001_201912_nytimes_tech_news.json"
    dj_path = "./data/201001_202007_DJ_tech_news.json"
    wsj_path = "./data/201001_202007_WSJ_Tech_News.json"
    out_path = "./data/out.dat"

    analyzer = SentimentIntensityAnalyzer()
    nytimes = load_json(nytimes_path)
    dj = load_json(dj_path)
    wsj = load_json(wsj_path)

    result = []
    nytimes_s = nytimes_analysis(nytimes, analyzer)
    dj_s = dj_wsj_analysis(dj, analyzer, "dj")
    wsj_s = dj_wsj_analysis(wsj, analyzer, "wsj")

    result.extend(nytimes_s)
    result.extend(dj_s)
    result.extend(wsj_s)

    with open(out_path, 'wb') as outfile:
        pickle.dump(result, outfile)

    ## Get result
    # with open(out_path, "rb") as f:
    #     result = pickle.load(f)
# %%

if __name__ == '__main__':
    main()
