# -*- coding: utf-8 -*-
import tweepy
import re
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import arff
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import plotly.express as px
import pickle
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import pytz

init_notebook_mode(connected=True)
import cufflinks as cf
from numpy import mean
import statistics

cf.go_offline()
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pickle

# setting up paths
path_for_cyberbullying_model = 'cyberBullying_model.sav'
path_for_hatespeech_model = 'Pickle_Hate_Tweet_Model.pkl'
path_for_count_vector = 'CountVector.sav'
path_for_twitter_groundtruth = 'twitter_groundtruth.arff'

# setting up tweepy api
api_key = "iE7uxkyppsvhnecWJqicn6tJy"
api_key_secret = "aKAGPkNzzkTlBbGQSw87w8P0QJcK37fjV82QehiIPMFWwVt70I"
access_token = "1032658815420780544-pSY9SA6RWVa0bdwX6sgWpd4cwoXl6N"
access_token_secret = "MvU74DZyj72iAf3ftCgGTr7132sWPBRAOhgkkG0k6ZCaS"

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Function: get_perimeters(userid)
# Returns dictionary
# keys are:
# is_spammer : 1 for spammer, 0 for non-spammer
# post_Interval : average time interval between 2 posts
# response_interval : average time taken to reply
# avg_number_of_hashtags : average number of hashtags used in every post
# percentage_of_capital_letter : percentage of capital letters used
# percentage_of_tweets_greater_than_120_chars : percentage of tweets which have more than 120 characters
# avg_length_of_tweets : average length of the tweets
# count_of_cyberbullying_for_posted_tweets : number of tweets classified as cyberbullying
# count_of_cyberbullying_for_liked_tweets : number of liked tweets classified as cyberbullying
# count_of_hatespeech_for_posted_tweets : number of tweets classified as hate speech
# count_of_hatespeech_for_liked_tweets : number of liked tweets classified as hate speech

# find average number of hashtags
from collections import Counter


def count_average_hashtags(list):
    count = 0
    count_hashtag = Counter()
    for element in list:
        for hast_tag in re.findall('#(\w+)', element):
            count_hashtag[hast_tag] += 1
            count += 1
    return count / len(list)


# count percentage of capital letters used
def get_percentage_of_capital_letters(list):
    count = 0
    count_letters = 0
    for element in list:
        count += len(re.findall(r'[A-Z]', element))
        count_letters += len(element)
    return (count / count_letters) * 100


# count number of tweets >= 120 characters
def get_percentage_of_tweets_greater_120(list):
    count_greater_120 = 0
    for element in list:
        if (len(element) > 120):
            count_greater_120 += 1
    return (count_greater_120 / len(list)) * 100


# count average length of tweets
def get_average_length_of_tweets(list):
    count = 0
    for element in list:
        count += len(element)
    return count / len(list)


# return list of liked tweets in format (username, tweet id, tweet text)
def get_recent_liked_tweets(user, number_of_tweets):
    liked_tweets_list = []
    for favorite in tweepy.Cursor(api.favorites, id=user, tweet_mode='extended').items(number_of_tweets):
        # liked_tweet = []
        # liked_tweet.append(str(favorite.user.screen_name))
        # liked_tweet.append(str(favorite.id))
        # liked_tweet.append(str(favorite.full_text))
        liked_tweets_list.append(favorite.full_text)
    return liked_tweets_list


def get_recent_tweets(user, number_of_tweets):
    tweet_list = []
    for status in tweepy.Cursor(api.user_timeline, id=user, tweet_mode="extended").items(number_of_tweets):
        if status.full_text.startswith("RT @") == True:
            tweet_list.append(status.retweeted_status.full_text[5:])
        else:
            tweet_list.append(status.full_text)
    return tweet_list


def tweetextract(uid):
    userid = uid
    consumer_key = 'THrn3aKGF8aHoy06qpI1xcc8M'
    consumer_secret = 'Dtqcuic6z0gX8f2tulp8SUhObOcxUrlEk4b9blwZCXXip7Cxkr'
    access_token = '39222626-urIui6mZYIgMohalBQnBq7pSiK4reZGDXDbBVxAqW'
    access_token_secret = 'hDJ2cbbBFiaSNgTKp8thHsCcuDwuJFD4Vqxin7sIfY8fr'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = api.user_timeline(screen_name=userid,
                               # 200 is the maximum allowed count
                               count=500,
                               include_rts=True,

                               # Necessary to keep full_text
                               # otherwise only the first 140 words are extracted
                               tweet_mode='extended'
                               )

    oldest_id = tweets[-1].id
    while len(tweets) <= 500:
        tweet = api.user_timeline(screen_name=userid,
                                  # 200 is the maximum allowed count
                                  count=200,
                                  include_rts=True,
                                  max_id=oldest_id - 1,

                                  tweet_mode='extended'
                                  )
        if len(tweet) == 0:
            break
        oldest_id = tweets[-1].id
        tweets.extend(tweet)

    out = []
    out.append(tweets)
    out.append(api)

    return out


def twttime(tweets):
    intertwt_time = []
    times = []
    for i in range(len(tweets) - 1):
        tweet_time = str(tweets[i].created_at)
        tweetprev_time = str(tweets[i + 1].created_at)
        a = pd.to_datetime(tweet_time)

        b = pd.to_datetime(tweetprev_time)
        intertwt_time.append(a - b)
        times.append(a)
        times.append(b)

    df = pd.DataFrame(intertwt_time, columns=['tt'])
    df['tt'] = df['tt'].values.astype(numpy.int64)
    timebtwmax = df['tt'].max()

    f = pd.DataFrame(times, columns=['tweet_time'])
    f['count'] = [1] * len(times)

    f['tweet_time'] = pd.to_datetime(f['tweet_time'], format='%Y-%m-%d %H:%M:%S')
    l = f['tweet_time'].tolist()

    f['day'] = f['tweet_time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    cumm = f.groupby(['day']).agg(numpy.sum)

    # cumm['year_response_cum']=pd.to_timedelta(cumm['response'])

    cumm['days'] = cumm.index
    cumm['days'] = pd.to_datetime(cumm['days'], errors='coerce')

    cumm['week_number'] = cumm['days'].dt.week
    maxweek = cumm['week_number'].max()
    k = cumm['week_number'].tolist()
    weekcount = []
    for i in range(maxweek):
        c = 0
        for j in range(len(k)):
            if (k[j] == i + 1):
                c += 1
        weekcount.append(c)

    twtdaymean = cumm['count'].mean()
    twtdaymin = cumm['count'].min()
    output = []
    output.append(df['tt'].tolist())
    output.append(cumm['count'].tolist())
    output.append(weekcount)
    # print(output)
    return output


def followers(userid, api):
    user = api.get_user(userid)

    # fetching the followers_count
    followers = user.followers_count
    return followers


def followee(userid, api):
    user = api.get_user(userid)
    followee = user.friends_count
    return followee


def followee_tweetcount(userid, api):
    tweet_count_followee = []
    l = []
    for friend in tweepy.Cursor(api.followers_ids, screen_name=userid).items(250):
        l.append(friend)

    i = 0
    if 100 >= len(l) > 0:
        users = api.lookup_users(l)
        for j in users:
            tweet_count = j._json['statuses_count']
            tweet_count_followee.append(tweet_count)
    else:
        while i + 100 < len(l):

            users = api.lookup_users(l[i:i + 100])
            for j in users:
                tweet_count = j._json['statuses_count']
                tweet_count_followee.append(tweet_count)
            i = i + 100

    return sum(tweet_count_followee)


def tweetscharactercount(tweets):
    count = []
    for tweet in tweets:
        twt = tweet.full_text.split(" ")
        count.append(len(twt))
    return count


def tweetsmentioncount(tweets):
    count = []
    for tweet in tweets:
        mention = tweet.entities['user_mentions']
        count.append(len(mention))
    return count


def replies(tweets):
    reply = 0
    for i in range(len(tweets)):
        if (tweets[i]._json['in_reply_to_status_id'] is not None):
            reply += 1
    return reply


def hashtags(tweets):
    count = []
    for tweet in tweets:
        hashtag = tweet.entities['hashtags']
        count.append(len(hashtag))
    return count


def url(tweets):
    count = []
    for tweet in tweets:
        url = tweet.entities['urls']
        count.append(len(url))
    return count


def numericchar(tweets):
    numericcount = []
    for tweet in tweets:
        twt = tweet.full_text
        numericcount.append(len(re.findall(r"\d", twt)))
    return numericcount


def retweet(tweets):
    rtcount = []
    for tweet in tweets:
        twt = tweet.full_text
        rtcount.append(len(re.findall(r"RT\s*@\w+", twt)))
    return rtcount


def mentionreplycount(tweets, userid, api):
    mentions = tweepy.Cursor(api.mentions_timeline, user_id=userid, count=200).items()
    mentioncount = 0
    replycount = 0
    for mention in mentions:
        mentioncount += 1
        if (mention.in_reply_to_status_id_str):
            replycount += 1
        out = []
        out.append(mentioncount)
        out.append(replycount)
        return out


import statistics


def features(tweets, userid, api):
    feat = numpy.zeros((1, 58))
    #     df_comp=df[['time_between_posts_max','follower_count','post_count_per_day_mean','total_tweet_count_of_followees','followee_count','post_count_per_day_min','characters_per_tweet_max','mentions_per_tweet_max', 'spammer']]
    ext = twttime(tweets)
    timebtwtwt = ext[0]
    twtperday = ext[1]
    twtperweek = ext[2]
    followercount = followers(userid, api)
    totalpostcountfollowees = followee_tweetcount(userid, api)
    followeecount = followee(userid, api)
    characterscount = tweetscharactercount(tweets)
    mentioncount = tweetsmentioncount(tweets)
    repliescount = replies(tweets)
    hashtagscount = hashtags(tweets)
    urlcount = url(tweets)
    numcount = numericchar(tweets)
    rtcount = retweet(tweets)
    out = mentionreplycount(tweets, userid, api)
    mentionct = out[0]
    replyct = out[1]

    urlc = 0
    for i in urlcount:
        if i > 0:
            urlc += 1

    hashtagperwordlen = []
    for i in range(len(hashtagscount)):
        hashtagperwordlen.append(numpy.float64(hashtagscount[i] / characterscount[i]))

    urlperwordlen = []
    for i in range(len(urlcount)):
        urlperwordlen.append(numpy.float64(urlcount[i] / characterscount[i]))

    feat[0][0] = numpy.float64(followercount / followeecount)
    feat[0][1] = numpy.float64(repliescount / len(tweets))
    feat[0][2] = numpy.float64(urlc / len(tweets))
    feat[0][3] = float(repliescount)
    feat[0][4] = numpy.average(hashtagperwordlen)
    feat[0][5] = float(statistics.median(hashtagperwordlen))
    feat[0][6] = float(min(hashtagperwordlen))
    feat[0][7] = float(max(hashtagperwordlen))
    feat[0][8] = numpy.average(urlperwordlen)
    feat[0][9] = float(statistics.median(urlperwordlen))
    feat[0][10] = float(min(urlperwordlen))
    feat[0][11] = float(max(urlperwordlen))
    feat[0][12] = numpy.average(characterscount)
    feat[0][13] = float(statistics.median(characterscount))
    feat[0][14] = float(min(characterscount))
    feat[0][15] = float(max(characterscount))
    feat[0][16] = numpy.average(hashtagscount)
    feat[0][17] = float(statistics.median(hashtagscount))
    feat[0][18] = float(min(hashtagscount))
    feat[0][19] = float(max(hashtagscount))
    feat[0][20] = numpy.average(mentioncount)
    feat[0][21] = float(statistics.median(mentioncount))
    feat[0][22] = float(min(mentioncount))
    feat[0][23] = float(max(mentioncount))
    feat[0][24] = numpy.average(numcount)
    feat[0][25] = float(statistics.median(numcount))
    feat[0][26] = float(min(numcount))
    feat[0][27] = float(max(numcount))
    feat[0][28] = numpy.average(urlcount)
    feat[0][29] = float(statistics.median(urlcount))
    feat[0][30] = float(min(urlcount))
    feat[0][31] = float(max(urlcount))
    feat[0][32] = numpy.average(characterscount)
    feat[0][33] = float(statistics.median(characterscount))
    feat[0][34] = float(min(characterscount))
    feat[0][35] = float(max(characterscount))
    feat[0][36] = numpy.average(rtcount)
    feat[0][37] = float(statistics.median(rtcount))
    feat[0][38] = float(min(rtcount))
    feat[0][39] = float(max(rtcount))
    feat[0][40] = float(followeecount)
    feat[0][41] = float(followercount)
    user = api.get_user(userid)
    tweetcount = user._json['statuses_count']
    feat[0][42] = float(tweetcount)
    feat[0][43] = float(mentionct)
    feat[0][44] = float(replyct)
    feat[0][45] = float(totalpostcountfollowees)
    feat[0][46] = numpy.average(timebtwtwt)
    feat[0][47] = float(statistics.median(timebtwtwt))
    feat[0][48] = float(min(timebtwtwt))
    feat[0][49] = float(max(timebtwtwt))
    feat[0][50] = numpy.average(twtperday)
    feat[0][51] = float(statistics.median(twtperday))
    feat[0][52] = float(min(twtperday))
    feat[0][53] = float(max(twtperday))
    feat[0][54] = numpy.average(twtperweek)
    feat[0][55] = float(statistics.median(twtperweek))
    feat[0][56] = float(min(twtperweek))
    feat[0][57] = float(max(twtperweek))

    #     feat[0][0]=float(timebtwmax)
    #     feat[0][1]=float(followercount)
    #     feat[0][2]=float(twtdaymean)
    #     feat[0][3]=float(postcountfollowees)
    #     feat[0][4]=float(followeecount)
    #     feat[0][5]=float(twtdaymin)
    #     feat[0][6]=float(characterstwtmax)
    #     feat[0][7]=float(mentiontwtmax)
    return feat


def spamclassifer(data_train):
    #
    df = pd.DataFrame(data_train[0])
    df['spammer'] = df['spammer'].str.decode("utf-8")
    df['spammer'] = df.spammer.map({'yes': 0, 'no': 1})
    #     df_comp=df[['time_between_posts_max','follower_count','post_count_per_day_mean','total_tweet_count_of_followees','followee_count','post_count_per_day_min','characters_per_tweet_max','mentions_per_tweet_max', 'spammer']]
    #     print(df_comp.head())

    y = df['spammer']
    train = df.drop(columns=['spammer'])

    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=13)

    # y_test=df_test['spammer']
    # test=df_test.drop(columns=['spammer'])
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression().fit(X_train, y_train)

    predict = clf.predict(X_test)
    # print("accuracy of classifier on dataset")
    # print(print(np.mean(predict == y_test)))

    #     print(Counter(df['spammer']))
    #     print(Counter(predict))
    # print(classification_report(y_test,predict))
    return clf, scaler


#
def tweeting(tweets):
    intertwt_time = []
    times = []
    for i in range(len(tweets) - 1):
        tweet_time = str(tweets[i].created_at)
        tweetprev_time = str(tweets[i + 1].created_at)
        a = pd.to_datetime(tweet_time)

        b = pd.to_datetime(tweetprev_time)
        intertwt_time.append(a - b)
        times.append(a)
        times.append(b)

    df = pd.DataFrame(intertwt_time, columns=['tt'])
    df['tt'] = df['tt'].values.astype(numpy.int64)
    #     print(pd.to_timedelta(df['tt'].min()))

    #     print(pd.to_timedelta(df['tt'].max()))

    meantwt = pd.to_timedelta(df['tt'].mean())

    #     time series plots
    f = pd.DataFrame(times, columns=['tweet_time'])
    f['count'] = [1] * len(times)

    f['tweet_time'] = pd.to_datetime(f['tweet_time'], format='%Y-%m-%d %H:%M:%S')

    f['year'] = pd.DatetimeIndex(f['tweet_time']).year
    f['month'] = f['tweet_time'].apply(lambda x: x.strftime('%Y-%m'))
    f['day'] = f['tweet_time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    f['hour'] = f['tweet_time'].apply(lambda x: x.strftime('%Y-%m-%d %H'))

    cumm = f.groupby(['year']).agg(numpy.sum)
    # cumm['year_response_cum']=pd.to_timedelta(cumm['response'])
    cumm['years'] = cumm.index

    fig = px.bar(cumm, x='years', y='count', title='Tweet Posts (each year)')
    first_val = min(cumm['years'].drop_duplicates().to_list())
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=first_val,
            dtick=1
        )
    )
    fig.write_image("plot_1.png")

    cumm = f.groupby(['month']).agg(numpy.sum)
    cumm['months'] = cumm.index

    fig = px.bar(cumm, x='months', y='count', title='Tweet Posts (each month)')
    fig.write_image("plot_2.png")

    cumm = f.groupby(['day']).agg(numpy.sum)

    cumm['days'] = cumm.index

    # cumm.iplot(kind='bar', x='days', y='count')

    return meantwt


def responseinteraction(tweets, userid, api):
    twt = []
    reply = []
    for i in range(len(tweets)):
        if (tweets[i]._json['in_reply_to_status_id'] is not None):
            #         replyids.append((tweets[i]._json['in_reply_to_status_id'],tweets[i]._json['in_reply_to_user_id']))
            #         print(tweets[i]._json['in_reply_to_status_id'])
            twt.append(tweets[i]._json['in_reply_to_status_id'])
            reply.append(tweets[i])

    if (len(reply) == 0):
        print(" user has no interaction in form of replies")
        return None

    else:
        replytwt = []
        i = 0
        while i + 100 < len(twt):
            tweet = api.statuses_lookup(twt[i:i + 100], tweet_mode='extended')

            ids = []

            for t in tweet:
                ids.append(t._json['id'])

            for rep in reply[i:i + 100]:
                if rep._json['in_reply_to_status_id'] in ids:
                    twtt = ids.index(rep._json['in_reply_to_status_id'])
                    replytwt.append((rep, tweet[twtt]))

            i = i + 100

        response = []
        reptime = []

        for i in range(len(replytwt)):
            tweet = replytwt[i][1]
            reply = replytwt[i][0]

            tweet_time = str(tweet.created_at)
            reply_time = str(reply.created_at)
            a = pd.to_datetime(tweet_time)

            b = pd.to_datetime(reply_time)

            reptime.append(a)
            response.append(b - a)

        f = pd.DataFrame(response, columns=['rt'])
        f['rt'] = f['rt'].values.astype(numpy.int64)
        responsemmin = pd.to_timedelta(f['rt'].min())
        responsemax = pd.to_timedelta(f['rt'].max())
        responsemean = pd.to_timedelta(f['rt'].mean())

        df = pd.DataFrame(response, columns=['response'])
        df['response'] = df['response'].values.astype(numpy.int64)
        df['time'] = reptime

        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')

        df['year'] = pd.DatetimeIndex(df['time']).year
        df['month'] = df['time'].apply(lambda x: x.strftime('%Y-%m'))
        df['day'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['hour'] = df['time'].apply(lambda x: x.strftime('%Y-%m-%d %H'))

        # time series plots
        cumm = df.groupby(['year']).agg(numpy.mean)
        cumm['year_response_cum'] = pd.to_timedelta(cumm['response'])
        cumm['years'] = cumm.index

        # cumm.iplot(kind='bar', x='years',y='response', text='year_response_cum')
        """fig = px.bar(cumm, x='years', y='response', title='Cumulative Tweet Response (each year)')
        first_val = min(cumm['years'].drop_duplicates().to_list())
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=first_val,
                dtick=1
            )
        )
        
        fig.write_image("plot_1.png")
        fig.show()"""

        cumm2 = df.groupby(['month']).agg(numpy.mean)
        cumm2['month_response_cum'] = pd.to_timedelta(cumm2['response'])
        cumm2['months'] = cumm2.index

        # cumm2.iplot(kind='bar', x='months',y='response', text='month_response_cum')

        cumm2 = df.groupby(['day']).agg(numpy.mean)
        cumm2['day_response_cum'] = pd.to_timedelta(cumm2['response'])
        cumm2['days'] = cumm2.index

        # cumm2.iplot(kind='bar', x='days',y='response', text='day_response_cum')

        cumm2 = df.groupby(['hour']).agg(numpy.mean)
        cumm2['hour_response_cum'] = pd.to_timedelta(cumm2['response'])
        cumm2['hours'] = cumm2.index

        # cumm2.iplot(kind='bar', x='hours',y='response', text='hour_response_cum')
        return responsemean


# consistent interaction main function
def consistentinteraction(userid):
    ext = tweetextract(userid)

    tweets = ext[0]

    api = ext[1]
    postsinterval = tweeting(tweets)
    responseinterval = responseinteraction(tweets, userid, api)
    # print("user interaction is identified with how frequently he posts and replies")
    # print(userid+ " posts tweets in average intervals of ")
    # print(postsinterval)
    # print(" and replies to other tweets in average time duration ")
    # print(responseinterval)
    posts_interval = int(round(postsinterval.total_seconds() / 60))
    if responseinterval.total_seconds() == responseinterval.total_seconds():
        response_interval = int(round(responseinterval.total_seconds() / 60))
    else:
        response_interval = 0
    return np.array([posts_interval, response_interval])


#
# spamming idenitfication main function

def spammingbehaviour(userid):
    ext = tweetextract(userid)

    tweets = ext[0]

    api = ext[1]
    feat = features(tweets, userid, api)
    scaler = StandardScaler()

    data = arff.loadarff(path_for_twitter_groundtruth)
    classifier, scaler = spamclassifer(data)
    feat = scaler.transform(feat)
    out = classifier.predict(feat)
    return out


def get_count_for_cyber_bullying_tweets(tweets):
    loaded_model = pickle.load(open(path_for_cyberbullying_model, 'rb'))
    count_vector = pickle.load(open(path_for_count_vector, 'rb'))
    New_Test_Sample_Vector = count_vector.transform(tweets)
    predicted_array = loaded_model.predict(New_Test_Sample_Vector)
    count_ones = 0
    for i in predicted_array:
        if i == 1:
            count_ones += 1
    return count_ones


def get_count_for_hate_speech_tweets(tweets):
    with open(path_for_hatespeech_model, 'rb') as file:
        Pickled_LR_Model = pickle.load(file)
    predicted_array = Pickled_LR_Model.predict(tweets)
    count_ones = 0
    for i in predicted_array:
        if i == 1:
            count_ones += 1
    return count_ones


def username_bio(handle):
    tweets = api.user_timeline(screen_name=handle, count=1)
    tweet_list = [tweet for tweet in tweets]
    json_dict = tweet_list[0].__getattribute__("_json")
    screen_name = '@' + handle
    name = json_dict['user']['name']
    bio = json_dict['user']['description']
    return screen_name, name, bio


def generate_report():
    file = open('feature_values', 'rb')
    feature_values = pickle.load(file)
    doc = SimpleDocTemplate(feature_values['pdf_name'], pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=54, bottomMargin=54)
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    ptext = '<font size=20><b><u>TwitEtiquette Report</u></b></font>'
    Story.append(Spacer(1, 48))
    Story.append(Paragraph(ptext, styles["Center"]))
    Story.append(Spacer(1, 24))

    img = "header_image.jpg"

    IST = pytz.timezone('Asia/Kolkata')
    datetime_ist = datetime.now(IST)
    datetime_string = datetime_ist.strftime('%d/%m/%Y %H:%M:%S')

    im = Image(img, 7 * inch, 2 * inch)
    Story.append(im)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    ptext = '<font size=12>Date and Time of Report Generation: %s</font>' % datetime_string
    Story.append(Spacer(1, 48))
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 24))

    ptext = '<font size=12>Twitter Handle: %s</font>' % (feature_values['screen_name'],)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    ptext = '<font size=12>Twitter Username: %s</font>' % (feature_values['name'],)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    ptext = '<font size=12>Bio: %s</font>' % (feature_values['bio'],)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 24))

    ptext = '<font size=11>Netizen Score (out of 100)</font>'
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))

    ptext = '<font size=14><b>%s</b></font>' % (feature_values['score'],)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 24))

    if feature_values['score'] < 75:
        img = "bad.jpeg"
    else:
        img = "good.jpeg"

    im = Image(img, 5 * inch, 3 * inch)
    Story.append(im)
    Story.append(Spacer(1, 12))

    ptext = '<font size=12>Please find the details below</font>'
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = "Based on user's tweets, classified as spammer or non spammer"
    if feature_values['is_spammer'] == 'Potential Spammer':
        tip = "User tweets in large numbers leading to be classified as a potential spammer," \
              "Keep the tweeting frequency less than current tweeting frequency."
    else:
        tip = "User tweets regularly but not too often, " \
              "Tweet frequency is good and not classified under spamming. Keep it up!."

    ptext = '<font size=14><b>Spamming: %s</b></font>' % (feature_values['is_spammer'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Average Time Interval between 2 consecutive tweets'
    if feature_values['post_Interval'] < 1000:
        tip = "Difference between consecutive tweets is very less, " \
              "it can lead to overwhelming posts."
    else:
        tip = "User posts tweet at regular intervals, " \
              "Average Time interval between consecutive tweets is good. Keep it up."

    ptext = '<font size=14><b>Average Time Interval: %s mins</b></font>' % (feature_values['post_Interval'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Average time taken by the user to respond to tweets'
    if feature_values['response_interval'] > 2880:
        tip = "Average Response time is very high, try to reply to tweets quickly."
    else:
        tip = "Average Response time is very good, keep it up!"

    ptext = '<font size=14><b>Response Time Interval: %s mins</b></font>' % (feature_values['response_interval'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Average number of hashtags used in tweets'
    if feature_values['avg_number_of_hashtags'] > 4:
        tip = "Too many hashtags are used, Use only topic relevant hashtags"
    else:
        tip = "Usage of hashtags are done very well. Keep it up!"

    ptext = '<font size=14><b># of Hashtags: %s</b></font>' % (feature_values['avg_number_of_hashtags'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Percentage of use of capital letters used in tweets'
    if feature_values['percentage_of_capital_letter'] == 100:
        tip = "Too many capital letters are used, please reduce usage of capital letters"
    elif 90 < feature_values['percentage_of_capital_letter'] < 100:
        tip = "More than usual usage of capital letters, try to reduce to a lighter tone in tweets"
    elif 70 < feature_values['percentage_of_capital_letter'] < 90:
        tip = "Usage of capital letters is higher than an average user, please use lighter tone in tweets"

    ptext = '<font size=14><b>Capital Letters: %s &percnt;</b></font>' % (
        feature_values['percentage_of_capital_letter'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Percentage of tweets greater than 120 characters'
    tip = "Tweets with less than 120 characters are recommended"

    ptext = '<font size=14><b>Percentage of tweets greater than 120 characters: %s &percnt;</b></font>' % (
        feature_values['percentage_of_tweets_greater_than_120_chars'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Average length of tweets posted by user'
    if 120 - feature_values['avg_length_of_tweets'] > 0:
        tip = 'Average length of tweets is higher than usual.'
    if 120 - feature_values['avg_length_of_tweets'] < 0:
        tip = 'Average length of tweets is within limits, keep it up.'

    ptext = '<font size=14><b>Average Tweet Length: %s</b></font>' % (
        feature_values['avg_length_of_tweets'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Use of directed profane language in posted tweets by user'
    tip = 'Reduce directed and targeted insults to improve the score'
    ptext = '<font size=14><b>CyberBullying posted tweets: %s</b></font>' % (
        feature_values['count_of_cyberbullying_for_posted_tweets'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Use of directed profane language in tweets liked by user'
    tip = 'Avoid liking and interacting with Cyberbullying tweets to improve the score'
    ptext = '<font size=14><b>CyberBullying liked tweets: %s</b></font>' % (
        feature_values['count_of_cyberbullying_for_liked_tweets'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Use of hate speech in tweets posted by user'
    tip = 'Reduce involving abusive words and negative content in tweets'
    ptext = '<font size=14><b>Hate speech posted tweets: %s</b></font>' % (
        feature_values['count_of_hatespeech_for_posted_tweets'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    desc = 'Use of hate speech in tweets liked by user'
    tip = 'Avoid involvement with tweets containing negative and hateful content'
    ptext = '<font size=14><b>Hate Speech liked tweets: %s</b></font>' % (
        feature_values['count_of_hatespeech_for_liked_tweets'],)
    dtext = '<font size=11>Description: %s</font>' % (desc,)
    ttext = '<font size=11>Tip: %s</font>' % (tip,)
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(dtext, styles["Justify"]))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(ttext, styles["Justify"]))
    Story.append(Spacer(1, 12))

    img = "plot_1.png"

    im = Image(img, 6 * inch, 4 * inch, hAlign='CENTER')
    Story.append(im)
    Story.append(Spacer(1, 12))

    img = "plot_2.png"

    im = Image(img, 6 * inch, 4 * inch, hAlign='CENTER')
    Story.append(im)
    Story.append(Spacer(1, 36))

    ptext = '<font size=13>Thank you for using the tool!</font>'
    Story.append(Paragraph(ptext, styles["Justify"]))
    Story.append(Spacer(1, 12))
    ptext = '<font size=11>Sincerely,</font>'
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 6))
    ptext = '<font size=11>Team TwitEtiquette</font>'
    Story.append(Paragraph(ptext, styles["Normal"]))
    Story.append(Spacer(1, 6))
    img = "logo_circular.png"

    im = Image(img, 1 * inch, 1 * inch, hAlign='LEFT')
    Story.append(im)
    Story.append(Spacer(1, 12))

    doc.build(Story)


def analyze_handle(userid):
    # Function: analyze_handle(userid)
    # Returns dictionary
    # keys are:
    # is_spammer : 1 for spammer, 0 for non-spammer
    # post_Interval : average time interval between 2 posts
    # response_interval : average time taken to reply
    # avg_number_of_hashtags : average number of hashtags used in every post
    # percentage_of_capital_letter : percentage of capital letters used
    # percentage_of_tweets_greater_than_120_chars : percentage of tweets which have more than 120 characters
    # avg_length_of_tweets : average length of the tweets
    # count_of_cyberbullying_for_posted_tweets : number of tweets classified as cyberbullying
    # count_of_cyberbullying_for_liked_tweets : number of liked tweets classified as cyberbullying
    # count_of_hatespeech_for_posted_tweets : number of tweets classified as hate speech
    # count_of_hatespeech_for_liked_tweets : number of liked tweets classified as hate speech

    list_features = {}
    score = 100

    screen_name, name, bio = username_bio(userid)
    list_features['screen_name'] = screen_name
    list_features['name'] = name
    list_features['bio'] = bio

    # spam behaviour
    output = spammingbehaviour(userid)
    if output[0] == 0:
        # it is a spammer
        list_features['is_spammer'] = 'Potential Spammer'
    else:
        # print("non-spammer")
        list_features['is_spammer'] = 'Not a Spammer'

    # consistent interaction
    interactions = consistentinteraction(userid)
    list_features['post_Interval'] = interactions[0]
    list_features['response_interval'] = interactions[1]
    tweets_text = get_recent_tweets(userid, 500)
    liked_tweets_text = get_recent_liked_tweets(userid, 100)
    list_features['avg_number_of_hashtags'] = round(count_average_hashtags(tweets_text), 2)
    list_features['percentage_of_capital_letter'] = round(get_percentage_of_capital_letters(tweets_text), 2)
    list_features['percentage_of_tweets_greater_than_120_chars'] = round(
        get_percentage_of_tweets_greater_120(tweets_text), 2)
    list_features['avg_length_of_tweets'] = round(get_average_length_of_tweets(tweets_text), 2)
    list_features['count_of_cyberbullying_for_posted_tweets'] = get_count_for_cyber_bullying_tweets(tweets_text)
    list_features['count_of_cyberbullying_for_liked_tweets'] = get_count_for_cyber_bullying_tweets(liked_tweets_text)
    list_features['count_of_hatespeech_for_posted_tweets'] = get_count_for_hate_speech_tweets(tweets_text)
    list_features['count_of_hatespeech_for_liked_tweets'] = get_count_for_hate_speech_tweets(liked_tweets_text)

    if list_features['is_spammer'] == 'Potential Spammer':
        score -= 40
    if list_features['post_Interval'] < 1000:
        score += (0.001 * list_features['post_Interval'])
    if list_features['post_Interval'] > 2440:
        score -= (0.01 * list_features['post_Interval'])
    if list_features['response_interval'] > 2880:
        score -= (0.01 * list_features['response_interval'])
    if list_features['avg_number_of_hashtags'] > 4:
        score -= (1.5 * list_features['avg_number_of_hashtags'])
    if list_features['percentage_of_capital_letter'] == 100:
        score -= 10
    if 90 < list_features['percentage_of_capital_letter'] < 100:
        score -= 8
    if 70 < list_features['percentage_of_capital_letter'] < 90:
        score -= 7
    score -= (0.2 * list_features['percentage_of_tweets_greater_than_120_chars'])
    if 120 - list_features['avg_length_of_tweets'] > 0:
        score += (0.01 * list_features['avg_length_of_tweets'])
    if 120 - list_features['avg_length_of_tweets'] < 0:
        score -= (0.01 * list_features['avg_length_of_tweets'])
    score -= (0.75 * list_features['count_of_cyberbullying_for_posted_tweets'])
    score -= (0.5 * list_features['count_of_cyberbullying_for_liked_tweets'])
    score -= (0.75 * list_features['count_of_hatespeech_for_posted_tweets'])
    score -= (0.5 * list_features['count_of_hatespeech_for_liked_tweets'])
    score = round(score, 1)
    if score > 99.9:
        list_features['score'] = 99.9
    else:
        list_features['score'] = score
    list_features['pdf_name'] = screen_name + ".pdf"

    # score threshold
    if list_features['score'] < 70:
        list_features['photo'] = "https://drive.google.com/uc?export=view&id=1F7tu1jc0hIjT9CttIV9IwfvFr5ldaFAq"
    else:
        list_features['photo'] = "https://drive.google.com/uc?export=view&id=1_7CmPOZTItAe--lJFNfJnLf4ffsWVtZk"

    file = open('feature_values', 'wb')
    pickle.dump(list_features, file)
    file.close()

    return list_features
