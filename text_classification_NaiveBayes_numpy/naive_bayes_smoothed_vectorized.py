import os
import argparse
import numpy as np
import nltk
from nltk.corpus import stopwords
import copy
import string
import json
import time
import csv
import re
import collections
from nltk.stem import WordNetLemmatizer
import inspect
import sys
from sklearn.feature_extraction.text import CountVectorizer
lemmatizer = WordNetLemmatizer()


N = None


class NaiveBayesSmoothed():

    def __init__(self):
        pass



    def load_train_data(self, trainfilepath):
        path = os.path.dirname(os.path.realpath(__file__)) + '/../../' + trainfilepath
        self.train_raw = np.asarray(np.load(path, allow_pickle=True), dtype= None).T
        self.train_raw = self.train_raw[:N]



    def split_data(self, data, pct_train=0.8):
        n_train = int(round(pct_train * data.shape[0]))

        self.train_split = self.train_raw[:n_train]

        if pct_train <1:
            self.test_split = self.train_raw[n_train:]
            print('new len: ', str(n_train))

        print('old len: ', str(data.shape[0]))



    def _common_preprocessing_dependent_on_manual_most_freq(self, sent):
        sent_lower = sent.lower()
        token = re.sub("[^\w]", " ", sent_lower).split()
        token = [lemmatizer.lemmatize(x).encode('ascii',errors='replace').decode() for x in token]

        return token



    def _common_preprocessing_dependent_on_manual_most_freq_WITH_normalization(self, sent):
        sent_lower = sent.lower()
        sent_normalized = sent_lower

        ### NORMALIZATION ###

        # 2) .. .. ..... -> DOTSPACE
        sent_normalized = re.sub(r"( ){,}\.( ){,}(\.){1,}[ \.]{,}", " DOTS ", sent_normalized)

        # 2b) 2+ spaces -> 1 space
        sent_normalized = re.sub(r"( ){2,}", " SPACES ", sent_normalized)

        # 3) exclamation marks !
        sent_normalized = re.sub(r"( ){,}\!( ){,}(\!){1,}[ \!]{,}"," EXCLAMMARK ", sent_normalized)

        # 4) question marks !
        sent_normalized = re.sub(r"( ){,}\?( ){,}(\?){1,}[ \?]{,}", " QUESTMARK ", sent_normalized)

        # 5) repeated letters
        sent_normalized = re.sub(r"(\w)\1{2,}", r"\1", sent_normalized)
        sent_normalized = re.sub(r"(\w\w)\1{2,}", r"\1\1", sent_normalized)
        sent_normalized = re.sub(r"(ha ){2,}", "haha ", sent_normalized)

        # 6) haha/lol/hehe
        sent_normalized = re.sub(r"( ){,}(haha|hehe|lol|hihi|ahah)( ){,}", " LAUGHTERALL ", sent_normalized)

        # 7) time-like patterns (15h30 | 15:30pm | 3:30 | 3:30pm)
        sent_normalized = re.sub(r"( ){,}\d{1,2}[\:h]\d{2}( ){,}(?:AM|PM|am|pm){,1}( ){,}", " TIMEEE ", sent_normalized)

        token = re.sub("[^\w]", " ", sent_normalized).split()
        token = [lemmatizer.lemmatize(x).encode('ascii',errors='replace').decode() for x in token]

        return token



    def _find_and_remove_most_used_words(self, array_token, n_most_used_words):

        #evaluate if we are in the training preprocess, if yes then find the most used words, otherwise (in test preprocess) do nothing
        if inspect.currentframe().f_back.f_code.co_name == "preprocess_train":
            self.most_used_words = {x[0] for x in collections.Counter(np.concatenate(array_token)).most_common(n_most_used_words)}

        # remove most used words
        for i in range(array_token.shape[0]):
            array_token[i] = [w for w in array_token[i] if w not in self.most_used_words]

        return array_token



    def preprocess_train(self, data, common_preprocessing, n_most_used_words=None):

        if data.ndim > 1:
            data_all_sentences = data[:, 0]
        else:
            data_all_sentences = data

        array_token = np.empty_like(data_all_sentences, dtype=object)

        # Common preprocessing
        for i, sent in enumerate(data_all_sentences):
            array_token[i] = getattr(self, common_preprocessing)(sent)

        # find and remove most used words
        if n_most_used_words is not None:
            array_token = self._find_and_remove_most_used_words(array_token,n_most_used_words)

        #OUTPUT ----> DATA TRAIN PREPROCESSED
        self.train_preprocessed = np.stack((array_token, data[:,1]), axis=1)



    def train(self, data):

        self.classes = list(np.unique(data[:,1]))
        self.vocabulary = list(set(np.concatenate(data[:,0])))
        print("length of vocabulary: {}".format(len(self.vocabulary)))

        #priors
        unique, counts = np.unique(data[:,1], return_counts=True)
        self.priors = np.array([dict(zip(unique, counts / data.shape[0]))[m] for m in self.classes])

        #frequencies by class
        bow = []
        for m, cls in enumerate(self.classes):
            vectorizer = CountVectorizer(vocabulary=self.vocabulary)
            text_joined = [' '.join(np.concatenate(data[data[:,1]==cls,0]))]
            bow.append(list(vectorizer.fit_transform(text_joined).toarray()[0,:]))

        self.bow_all_cls = np.array(bow).T



    def load_test_data(self, testfilepath):
        path = os.path.dirname(os.path.realpath(__file__)) + '/../../' + testfilepath
        self.test_raw = np.asarray(np.load(path, allow_pickle=True), dtype= None).T
        self.test_raw = self.test_raw[:N]



    def preprocess_test(self, data, common_preprocessing, n_most_used_words=None):

        if data.ndim>1:
            data_all_sentences = data[:,0]
        else:
            data_all_sentences = data

        array_token = np.empty_like(data_all_sentences, dtype=object)

        # Common preprocessing
        for i, sent in enumerate(data_all_sentences):
            array_token[i] = getattr(self, common_preprocessing)(sent)

        #find and remove most used words
        if n_most_used_words is not None:
            array_token = self._find_and_remove_most_used_words(array_token, n_most_used_words)

        #OUTPUT --->TEST DATA PREPROCESSED
        self.test_data_preprocessed = array_token




    def predict(self, test_data_preprocessed, alpha = 0.5):

        vectorizer = CountVectorizer(vocabulary=self.vocabulary)
        sent_joined = np.array(list(map(lambda row: ' '.join(row), test_data_preprocessed)))
        sent_bow = vectorizer.fit_transform(sent_joined).toarray()

        sum_log_prob = np.dot( sent_bow , np.log( (self.bow_all_cls+alpha)/np.sum(self.bow_all_cls, axis=0)))\
                       - (np.sum(sent_bow, axis=1)* (np.log(np.sum(self.bow_all_cls) + alpha * len(self.vocabulary))))[:, np.newaxis]

        class_map = sum_log_prob + np.log(self.priors)
        self.predictions = [self.classes[x] for x in list(np.argmax(class_map, axis=1).astype(int))]

        return self.predictions



    def output_to_csv(self, filename, predictions):

        with open(os.path.dirname(os.path.realpath(__file__))+"/../../results/" + filename, 'w') as csvFile:
            dataout = []
            dataout.append(['Id', 'Category'])
            for i, pred in enumerate(predictions):
                dataout.append([str(i)]+[pred])

            writer = csv.writer(csvFile)
            writer.writerows(dataout)



    def evaluate(self, true_label, pred_label):
        if len(true_label) != len(pred_label):
            raise ValueError('length of true labels {} do not match length of pred labels {}'.format(len(true_label), len(pred_label)))

        n_eq = 0
        for true, pred in zip(true_label, pred_label):
            if true == pred:
                n_eq += 1

        self.accuracy = n_eq/len(true_label)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train')
    parser.add_argument('--data_test')
    parser.add_argument('--split')
    args = parser.parse_args()


    ############ TUNING OF ALPHA AND NMOSTUSED WORDS ############


    # ARGUMENTS: --split 0.8 --data_train data/raw/data_train.pkl --data_test data/raw/data_test.pkl

    ####### RESULTS: 0.5 --> 0.5556  ; 1 --> 0.5485
    ####### SELECTED: 0.5


    #winner:
    # alpha: 0.17
    # nmostused: 60

    for alpha in reversed([0.17]):
        for nmostused in [60]:
        #for alpha in [0.5]:
            time_ = time.time()
            print('------- RUN START --------')

            model = NaiveBayesSmoothed()
            model.load_train_data(args.data_train)
            if args.split is not None:
                model.split_data(model.train_raw, float(args.split))
                model.preprocess_train(model.train_split, common_preprocessing="_common_preprocessing_dependent_on_manual_most_freq_WITH_normalization", n_most_used_words=nmostused)
            else:
                model.preprocess_train(model.train_raw, common_preprocessing="_common_preprocessing_dependent_on_manual_most_freq_WITH_normalization", n_most_used_words=nmostused)
            #print('time preprocess: {}'.format(time.time()-time_))
            time_ = time.time()


            model.train(model.train_preprocessed)
            #print('time train: {}'.format(time.time()-time_))
            time_ = time.time()


            #model.load_test_data(args.data_test)
            model.preprocess_test(model.test_split, common_preprocessing="_common_preprocessing_dependent_on_manual_most_freq_WITH_normalization", n_most_used_words=nmostused)
            #print('time test preprocess: {}'.format(time.time()-time_))
            time_ = time.time()


            model.predict(model.test_data_preprocessed, alpha)
            #print('time test predict: {}'.format(time.time()-time_))


            model.evaluate(list(model.test_split[:,1]), model.predictions)
            print('test accuracy for alpha of {} and nmostused of {} is {}'.format(alpha, nmostused , model.accuracy))

            print('------- RUN END --------')


    ############ RUNNING ON FULL DATA  ############

    #####  WARNING !!!! #####
    # ====> not updated since vectorized
    # ====> PAS UNE PRIORITE DE GOSSER CA LIVE CEST JUSTE POUR OUTPUTTER LE SUBMISSION.CSV ON PEUT CLANCHER NOS MODELES AVEC CROSS-VALIDATION AVANT





    # ARGUMENTS: --data_train data/raw/data_train.pkl --data_test data/raw/data_test.pkl

    # model = NaiveBayesSmoothed()
    # model.load_train_data(args.data_train)
    #
    # if args.split is not None:
    #     model.split_data(model.train_raw, float(args.split))
    #     model.preprocess_train(model.train_split)
    # else:
    #     model.preprocess_train(model.train_raw)
    #
    #
    # model.train(model.train_preprocessed, 0.5)
    #
    #
    # if args.split is not None:
    #     model.preprocess_test(model.test_split)
    #
    # else:
    #     model.load_test_data(args.data_test)
    #     model.preprocess_test(model.test_raw)
    #
    # model.predict(model.test_preprocessed)
    # model.output_to_csv('submission.csv', model.predictions)

    # print(1)
