#import beautifulsoup4
import urllib3
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('vader_lexicon')
import pandas as pd
import csv
import re
from collections import Counter


def obtain_features(df):
    """

    """
    ss = nltk.stem.SnowballStemmer('english')
    sentences = nltk.sent_tokenize(df['text'])

    sid = SentimentIntensityAnalyzer()
    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]

    # counts
    sentence_count = len(sentences)
    word_count = 0;
    character_count = 0;
    syllable_count = 0;

    cleaned_text = []
    for sentence in sentences:
        # only extract letters/numbers out of text, remove punctuations
        letter = re.sub("\d+", " ", sentence)
        letters_only = re.sub("[^\w\s]", " ", letter)

        words = nltk.word_tokenize(letters_only.lower())
        word_count += len(words)
        # Remove stop words such as "in", "this" etc
        stops = set(nltk.corpus.stopwords.words("english"))
        meaningful_words = []
        for word in words:
            character_count += len(word)
            syllable_count += count_syllable(word)
            if not word in stops:
                meaningful_words.append(word)
                #print(sid.polarity_scores(word)['compound'])
                polarScore = sid.polarity_scores(word)['compound']

                if polarScore >= 0.5:
                    pos_word_list.append(word)
                elif polarScore <= -0.5:
                    neg_word_list.append(word)
                else:
                    neu_word_list.append(word)

        #meaningful_words = [w for w in words if not w in stops]
        # tag meaning full words based on part of speach
        tagged_text = nltk.pos_tag(meaningful_words)
        cleaned_text.extend(([(ss.stem(w[0]), w[1]) for w in tagged_text]))

    cfg_list = Counter(tag for word,tag in cleaned_text)
    word_list = Counter(word for word,tag in cleaned_text)

    # To normalize cfg
    total = sum(cfg_list.values())
    #norm_cfg = dict((word, float(cnt)/total) for word,cnt in cfg_list.items())

    # calculate automated readability index
    #ARI = 4.71*(character_count/word_count) + 0.5*(word_count/sentence_count)-21.43
    # calculate Flesch–Kincaid grade level
   # FKGL = 0.39*(word_count/sentence_count) + 11.8*(syllable_count/word_count)-15.59

    matrix =  dict(cfg_list)
    #matrix['_rARI'] = ARI
    #matrix['_rFKGL'] = FKGL
    matrix['label'] = df['Label']
    #matrix['_pos_word'] = len(pos_word_list)
    #matrix['_neg_word'] = len(neg_word_list)
    #matrix['_neu_word'] = len(neu_word_list)

    print(matrix)
    return matrix

def count_syllable(word):
    """
    count the number of syllable in a given word
    """
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def readTrainingData(location):
    """
    """

    df = pd.read_csv(location, header=0)
    # initializing empty dataframe to store processed data
    processed_data = pd.DataFrame()

    # for each article
    for i in range(0, len(df)):
        try:
            # calling function to collect features

            data = obtain_features(df.iloc[i])
            print(data)
            # stores obtained features in dictionary form to a dataframe form
            temp = pd.DataFrame.from_dict(data, orient='index').reset_index()
            temp = temp.transpose()
            temp.columns = temp.iloc[0]
            temp = temp[1:]

        except:
            continue

        # add each article to storage?
        processed_data = pd.concat([processed_data, temp], sort = True)
    processed_data = processed_data.fillna('0')

    return processed_data


def dataFiles():
    # list of sources containing data
    #bf = 'TrainingData/BuzzFeed_real_news_content.csv'
    fake_and_real_data = 'TrainingData/fake_and_real_data.csv'

    data = readTrainingData(fake_and_real_data)
    #data = readTrainingData(bf)
    #data = pd.concat([d1, d2, d3], sort = True)

    data = data.drop(data.columns[0:1], axis=1)
    print(data)
    data.to_csv('preprocessed_data_test003	.csv')



if __name__ == "__main__":
    dataFiles()
