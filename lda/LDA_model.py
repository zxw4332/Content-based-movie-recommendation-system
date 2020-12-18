import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(40)
import nltk
from gensim import corpora, models

data = pd.read_pickle('movie_tags.pkl')
data['index'] = [x for x in range(data.shape[0])]

dic = {}
for x in data['tags']:
    for y in x:
        if y in dic:
            dic[y] = 1 + dic[y]
        else:
            dic[y] = 1
df = pd.DataFrame(dic, index=[0])
df = df.T.reset_index()
df.columns = ['tags', 'count']
dic_bigger_2 = {}
for x in df['tags']:
    dic_bigger_2[x] = 1
preprocessed = []
for line in data['tags']:
    new_line = []
    for word in line:
        if word in dic_bigger_2:
            new_line.append(word)
    preprocessed.append(new_line)
data['preprocessed'] = preprocessed
dictionary = gensim.corpora.Dictionary(data['preprocessed'])
count = 0
for k, v in dictionary.iteritems():
    count += 1
    if count > 10:
        break
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in data['preprocessed']]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

if __name__ ==  '__main__':
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=50, id2word=dictionary, passes=2, workers=2,minimum_probability = 0)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
    result_df = pd.DataFrame()
    result_ls = []
    for x in bow_corpus:
        y = lda_model_tfidf[x]
        new_result = []
        for n in y:
            new_result.append(n[1])
        result_ls.append(new_result)
    result_df['imdbId'] = data['movie_id']
    result_df['result'] = result_ls
    result_df.to_pickle('movie_lda_50.pkl')
    link_data = pd.read_csv('links.csv')
    sample_movie = pd.read_pickle('movie_lda_50.pkl')
    sample_data = link_data.merge(sample_movie, on = 'imdbId', how = 'right')[['movieId', 'result']]
    sample_data.to_pickle('movie_lda_50.pkl')