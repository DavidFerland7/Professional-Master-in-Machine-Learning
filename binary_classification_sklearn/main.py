# NOTE:
# best train score (cross-validated) = mean_train_score  (when rank_test_score == 1)
# best valid score (cross-validated) = mean_test_score  (when rank_test_score == 1)
# best test metrics (hold-out) = {metrics}_test


import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
from joblib import dump, load
import csv
from datetime import datetime


######################################
######### GLOBAL PARAMETERS ##########

N = None

root = "/Users/david/Documents/5_UDEM_machine_learning/1_Machine_learning/project/repo/Project_IFT6390/task2/"  # to change for other users
project_data_path = "data/"
file_name = 'KaggleV2-May-2016.csv'
train_file = 'train'  # 'original' or 'train'
model_selected = 'NB'  # RF | NB | LR
metrics = ['accuracy']
scoring_tuning = metrics
scoring_test = metrics
cv_results_all = []
n_jobs = 6
# exec = ['train_load_data', 'train_format', 'train_fit', 'train_tuning', 'train_save_model', 'test_load_model', 'test_load_data', 'test_predict', 'test_evaluate']   ### <------ ALL CHOICES
# exec = ['train_load_data', 'train_save_holdout']    # <------- LOAD ORIGINAL DATA AND SAVE A TRAIN(80%) AND HOLDOUT(20%)
exec = ['train_load_data', 'train_format', 'train_tuning', 'train_save_model', 'test_load_model', 'test_load_data', 'test_predict', 'test_evaluate']  # <------ tuning


######################################
######## FUNCTIONS PARAMETERS ########

_vars = {
    'features': {
        'int': ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'],
        'str': ['Gender'],
        'date': ['ScheduledDay', 'AppointmentDay'],
        'other': ['Neighbourhood']
    },
    'label': ['No-show']
}

param_load_data_pandas_from_csv = {
    'function': {
        'shuffle': False
    },
    'data': {
        'common': {
            'usecols': ['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show'],
            'nrows': N,
        },
        'type_specific': {
            'original': {'filepath_or_buffer': root + project_data_path + 'original' + '/' + file_name},
            'train': {'filepath_or_buffer': root + project_data_path + 'train' + '/' + file_name},
            'test': {'filepath_or_buffer': root + project_data_path + 'test' + '/' + file_name},
            'test_unlabel': {
                'filepath_or_buffer': root + project_data_path + 'test' + '/' + file_name,
                'usecols': ['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'],
            }
        }
    }
}

param_split_data = {
    'pct_train': 0.8
}

param_save_as_csv = {
    'pathout_train': root + project_data_path + "train/" + file_name,
    'pathout_test': root + project_data_path + "test/" + file_name,
    'header': True
}


########################################
######### FUNCTIONS DEFINITIONS ########

def load_data_pandas_from_csv(param_data, data_type='train', shuffle=False):
    df = None
    file_name, ext = os.path.splitext(param_data['type_specific'][data_type]['filepath_or_buffer'])
    if ext == '.csv':
        df = pd.read_csv(**{**param_data['common'], **param_data['type_specific'][data_type]})
    elif ext == '.pkl':
        array = np.asarray(np.load(path_data_train, allow_pickle=True), dtype=None).T
        df = pd.DataFrame(array, columns=names)[:nrows]

    if shuffle == True:
        df = df.sample(frac=1, replace=False, random_state=1).reset_index(drop=True)

    return df


def save_as_csv(data, data_type='train', index=False, header=False, sep=',', pathout_train=None, pathout_test=None):
    pathout = locals()["pathout_" + data_type]
    data.to_csv(pathout, index=index, header=header, sep=sep)


def split_data(X, y, pct_train=1):
    if pct_train == 1:
        return pd.concat((X, y), axis=1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=pct_train, random_state=1, stratify=y)
        #X_train, X_test, y_train, y_test = X_train.reset_index(), X_test.reset_index(), y_train.reset_index(), y_test.reset_index()
        print('new len: ', str(X_train.shape[0]))
        print('old len: ', str(X.shape[0]))
        return pd.concat((X_train, y_train), axis=1), pd.concat((X_test, y_test), axis=1)


def formatting(data, _vars):
    for type, vars in _vars['features'].items():
        if type == 'date':
            for var in vars:
                # datetime format can be easily manipulated for all its components (day, month, year, hour, minute, dayofweek,..)
                data[var] = pd.to_datetime(data[var], infer_datetime_format=True)
        elif type == 'str':
            pass
        elif type == 'int':
            pass
    # label encode target values ( e.g. Yes -> 1, No -> 0 )
    for var in _vars['label']:
        data[var] = np.ravel(label_binarize(data[var], classes=['No', "Yes"]))
    return data


########################################
########### PIPES DEFINITIONS ##########

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_feature_names(self):
        return self.columns

    def transform(self, df, y=None):

        #### One-hot encode str variables #####
        for var in _vars['features']['str'] + _vars['features']['other']:
            # values in train and in test as well
            df = df.assign(**{var: df[var].apply(lambda x: x.replace(' ', '_'))})
            df = pd.get_dummies(df, prefix=var, columns=[var])

            # values in train but not in test
            for col in getattr(self, "dummy_values_" + var):
                if col not in df.columns:
                    df[col] = 0

            # values in test but not in train
            for col in [c for c in df.columns.tolist() if c.startswith(var + "_")]:
                if col not in getattr(self, "dummy_values_" + var):
                    df.drop(columns=col, inplace=True)

        # features from : ScheduledDay -> period of the day person has scheduled
        var = 'ScheduledDay'
        df[var + '_weekday'] = df[var].apply(lambda x: x.dayofweek)
        df[var + '_hour_0-6'] = 0
        df[var + '_hour_7-11'] = 0
        df[var + '_hour_12-12'] = 0
        df[var + '_hour_13-17'] = 0
        df[var + '_hour_18-23'] = 0
        df.loc[(df[var].apply(lambda x: 0 <= x.hour) & df[var].apply(lambda x: x.hour <= 6)), var + '_hour_0-6'] = 1
        df.loc[(df[var].apply(lambda x: 7 <= x.hour) & df[var].apply(lambda x: x.hour <= 11)), var + '_hour_7-11'] = 1
        df.loc[(df[var].apply(lambda x: 12 <= x.hour) & df[var].apply(lambda x: x.hour <= 12)), var + '_hour_12-12'] = 1
        df.loc[(df[var].apply(lambda x: 13 <= x.hour) & df[var].apply(lambda x: x.hour <= 17)), var + '_hour_13-17'] = 1
        df.loc[(df[var].apply(lambda x: 18 <= x.hour) & df[var].apply(lambda x: x.hour <= 23)), var + '_hour_18-23'] = 1

        # features from : AppointmentDay -> day of actual appointement
        var = 'AppointmentDay'
        df[var + '_weekday'] = df[var].apply(lambda x: x.dayofweek)
        df[var + '_month'] = df[var].apply(lambda x: x.month)

        # number of day between appointement and schedule day (2 obs had appointment day before schedule day (invalid value) so diff is capped at 0)
        df['number_days_before_appointment'] = np.maximum((df['AppointmentDay'].dt.date - df['ScheduledDay'].dt.date).dt.days, 0)

        # DROP ORIGINAL DATETIME VARIABLES
        df.drop(columns=['ScheduledDay', 'AppointmentDay'], inplace=True)

        # df.to_csv('valid_values_df.csv')
        #print("data shape after feature extraction: {}".format(df.shape))

        # save columns header so we can retrieve them later with get_feature_names()
        self.columns = df.columns.tolist()

        return df

    def fit(self, df, y=None):
        """Returns self (unless something different happens in train and test)"""

        # one-hot encode str variables
        for var in _vars['features']['str'] + _vars['features']['other']:
            setattr(self, "dummy_values_" + var, [var + "_" + value.replace(" ", "_") for value in df[var].unique().tolist()] + [var + "_ValueInTestNotInTrain"])

        return self


if __name__ == '__main__':

    ###################### LOAD DATA ########################

    if 'train_load_data' in exec:
        # load data
        data_train_raw = load_data_pandas_from_csv(param_load_data_pandas_from_csv['data'], data_type=train_file, **param_load_data_pandas_from_csv['function'])
        print("shape of data_train_raw :{}".format(data_train_raw.shape))
        print(data_train_raw.head())

    ###################### SPLIT DATA ########################
    # Note: should be done only once

    if 'train_save_holdout' in exec:
        # define features/label variables
        features = [x for xs in list(_vars['features'].values()) for x in xs]
        label = _vars['label']

        data_train, data_test = split_data(data_train_raw[features], data_train_raw[label], **param_split_data)

        # save train/test data (to be done once)
        save_as_csv(data_train, data_type='train', **param_save_as_csv)  # save data_train (80%)
        save_as_csv(data_test, data_type='test', **param_save_as_csv)  # save data_test (20%)

    ###################### DATA FORMATTING ########################
    # Note: This part is common to all combinations of models

    if 'train_format' in exec:
        if 'data_train' not in globals():
            data_train = data_train_raw
        data_train_formatted = formatting(data_train, _vars)
        print("data_train_formatted:\n{}".format(data_train_formatted.head()))

    ###################### TRAINING ########################

    if 'train_fit' in exec:
        # get feature/label names in lists
        features = list(data_train_formatted.columns).copy()
        label = [features.pop(features.index(lab)) for lab in _vars['label']]

        if model_selected == 'LR':
            # model definition
            pipe = Pipeline([
                ('feat_extractor', FeatureExtractor()),
                ('clf', LogisticRegression(multi_class='multinomial', n_jobs=n_jobs, random_state=40, verbose=1, class_weight='balanced', solver='newton-cg')),
            ])

        elif model_selected == 'NB':
            pass

        elif model_selected == 'RF':
            pass
        else:
            raise ValueError("no model selected")

        model = pipe
        model.fit(data_train_formatted[features], data_train_formatted[label])

        # save model
        if 'train_save_model' in exec:
            dump(model, os.path.dirname(os.path.abspath(__file__)) + "/{}.joblib".format(model_selected))

    ####################### TUNING #########################

    if 'train_tuning' in exec:
        # get feature/label names in lists
        features = list(data_train_formatted.columns).copy()
        label = [features.pop(features.index(lab)) for lab in _vars['label']]

        # model definition
        if model_selected == 'LR':
            pipe = Pipeline([
                ('feat_extractor', FeatureExtractor()),
                ('clf', LogisticRegression(n_jobs=n_jobs, random_state=40, solver='newton-cg')),
            ])

            param_grid = {
                'clf__C': [0.0005, 0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25],
                'clf__class_weight': ['balanced', None],
            }
        elif model_selected == 'NB':
            pipe = Pipeline([
                ('feat_extractor', FeatureExtractor()),
                ('clf', ComplementNB()),
            ])

            param_grid = {
                #'clf__alpha':[0.1, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2.5, 5],  => 2.5
                'clf__alpha': [0.1, 0.5, 1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 5],
            }
        elif model_selected == 'RF':
            pass
            pipe = Pipeline([
                ('feat_extractor', FeatureExtractor()),
                ('clf', RandomForestClassifier(n_jobs=n_jobs, random_state=2, criterion="gini", bootstrap=True, oob_score=False)),
            ])

            param_grid = {
                # 1)
                # 'clf__class_weight': ['balanced', None],
                # 'clf__n_estimators': [5,10,25,50,75,100,125,150,200,250,300],

                # 2)
                # 'clf__class_weight': [None],
                # 'clf__n_estimators': [200],
                # 'clf__max_depth': [None, 15, 20, 25, 30, 35],


                # 3)
                # 'clf__class_weight': [None],
                # 'clf__n_estimators': [200],
                # 'clf__max_depth': [30],
                # 'clf__min_samples_split': [2,3,4,5,6,7,8]

                # 4)
                # 'clf__class_weight': [None],
                # 'clf__n_estimators': [200],
                # 'clf__max_depth': [30],
                # 'clf__min_samples_split': [2],
                # 'clf__max_features': [0.01,0.05,0.8,0.1,0.12,0.15]

                # 5)
                'clf__class_weight': [None],
                'clf__n_estimators': [200],
                'clf__max_depth': [30],
                'clf__min_samples_split': [2],
                'clf__max_features': [0.1],


            }

        else:
            raise ValueError("no model selected")

        # get best model under different metrics
        for scoring in scoring_tuning:
            model = GridSearchCV(pipe, param_grid=param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=40), n_jobs=n_jobs, scoring=scoring_tuning, return_train_score=True, refit=scoring)

            model.fit(data_train_formatted[features], np.ravel(data_train_formatted[label]))

            # print and save training results
            print("best_params_ of model {} with scoring {} = {}".format(model_selected, scoring, model.best_params_))
            model.cv_results_['scoring'] = [scoring for x in range(len(model.cv_results_['params']))]
            model.cv_results_['model'] = [model_selected for x in range(len(model.cv_results_['params']))]

            # save model
            if 'train_save_model' in exec:
                dump(model, os.path.dirname(os.path.abspath(__file__)) + "/{}_{}.joblib".format(model_selected, scoring))

    ###################### LOAD TEST DATA #########################
    if 'test_load_data' in exec:
        data_test = load_data_pandas_from_csv(param_load_data_pandas_from_csv['data'], data_type='test', **param_load_data_pandas_from_csv['function'])
        print("shape of data_test :{}".format(data_test.shape))
        print(data_test.head())

        # apply formatting
        data_test_formatted = formatting(data_test, _vars)

        # get feature/label names in lists
        features_test = list(data_test_formatted.columns).copy()
        label_test = [features_test.pop(features_test.index(lab)) for lab in _vars['label'] if lab in features_test]

    if 'test_predict' in exec:

        for scoring in scoring_test:

            ###################### LOAD MODEL #########################
            if 'test_load_model' in exec:
                model = load(os.path.dirname(os.path.abspath(__file__)) + "/{}_{}.joblib".format(model_selected, scoring))

            # predict
            pred = model.predict(data_test_formatted[features_test])

            ###################### EVALUATE #########################

            if 'test_evaluate' in exec:
                print(classification_report(data_test_formatted[label_test], pred))

                test_metrics_on_best_model = precision_recall_fscore_support(data_test_formatted[label_test], pred, average='binary')
                test_accuracy = accuracy_score(data_test_formatted[label_test], pred)
                model.cv_results_['precision_test'] = [test_metrics_on_best_model[0] for x in range(len(model.cv_results_['params']))]
                model.cv_results_['recall_test'] = [test_metrics_on_best_model[1] for x in range(len(model.cv_results_['params']))]
                model.cv_results_['fbeta_score_test'] = [test_metrics_on_best_model[2] for x in range(len(model.cv_results_['params']))]
                model.cv_results_['accuracy_test'] = [test_accuracy for x in range(len(model.cv_results_['params']))]

                cv_results_ = pd.DataFrame(model.cv_results_).to_dict('records')  # just to add index
                cv_results_all.extend(cv_results_)

    ####### OUTPUT STATS ######
    filepath = os.path.dirname(os.path.abspath(__file__)) + '/results/tuning/cv_results_{}_{}.csv'.format(model_selected, datetime.now().strftime('%Y_%m_%d__%H_%M_%S'))
    keys = []
    for dict_stat in cv_results_all:
        keys.extend(list(dict_stat.keys()))
    seen = set()
    seen_add = seen.add
    header_all = [x for x in keys if not (x in seen or seen_add(x))]
    header_base = ['model', 'scoring', 'params'] + [c for c in header_all if c.startswith("param_")] + ['accuracy_test', 'precision_test', 'recall_test', 'fbeta_score_test'] + np.ravel(np.array([['rank_test_' + x, 'mean_test_' + x, 'mean_train_' + x, 'std_test_' + x, 'std_train_' + x] for x in metrics])).tolist()

    header = header_base + [head for head in header_all if head not in header_base]

    for dict_stat in cv_results_all:
        for k in header:
            if k not in dict_stat.keys():
                dict_stat[k] = ''

    with open(filepath, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=list(header))
        dict_writer.writeheader()
        dict_writer.writerows(cv_results_all)
