# тут прописать импорт необходимых библиотек
from sklearn.feature_selection import chi2, f_regression

import os
from dataclasses import dataclass, field
from typing import List, Dict
import json

from tqdm import tqdm

import pandas as pd
import numpy as np

#ml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold, cross_val_score

from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss

import shap


class ShareNans():
    def __init__(self, thr_nulls=0.1):
        self.thr_nulls = thr_nulls
        self.selected_columns = []
        self.advanced_info = {}

    def fit(self, data):
        self.advanced_info = {col: data['data'][col].isna().sum() / data['data'].shape[0] for col in data['data'].columns}
        self.selected_columns = [col for col in self.advanced_info.keys() if self.advanced_info[col] <= self.thr_nulls]

    def fit_transform(self, data):
        self.fit(data)
        return data[self.selected_columns]


# выполнять только для определенного набора признаков, нельзя для категориальных признаков
class ShareUniq():
    def __init__(self, thr_nunique=0.9):
        self.thr_nunique = thr_nunique
        self.advanced_info = []
        self.advanced_info = {}

    def fit(self, data):
        self.advanced_info = {col: data['data'][col].nunique() / data['data'].shape[0] for col in data['data'].columns}
        self.advanced_info = [col for col in self.advanced_info.keys() if self.advanced_info[col] >= self.thr_nunique]

    def fit_transform(self, data):
        self.fit(data)
        return data[self.advanced_info]


class SelectByCorr():
    def __init__(self, thr_corr_pair=0.25, thr_corr_with_target=0.5):
        self.thr_corr_pair = thr_corr_pair
        self.thr_corr_with_target = thr_corr_with_target
        self.columns = []
        self.advanced_info = {}

    def fit(self, data, **kwargs):
        corr_selection_result = data['data'].corr(**kwargs)\
            .unstack()\
            .reset_index()\
            .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr_abs'})\
            .assign(corr_abs = lambda row: row['corr_abs'].map(np.fabs))\
            .assign(is_target = lambda row: (row['feature_1'] == data['target_name']).astype(int))\
            .query("corr_abs!=1")\
            .query("(corr_abs > @thr_corr_with_target and is_target==1) or (corr_abs < @thr_corr_pair and is_target==0)")
        
        self.advanced_info = corr_selection_result
        self.columns = set(corr_selection_result['feature_1']) & set(corr_selection_result['feature_2']) - set([data['target_name']])

    def fit_transform(self, data):
        self.fit(data)
        return data[self.columns]


class SelectByMethod():
    __doc__ = """Doc string"""

    def __init__(self, thr_alpha=0.1, fillna=True, only_numeric_features=True, model='classification',
                 scaler_for_chi_squared=None):
        self.thr_alpha = thr_alpha
        self.fillna = fillna
        self.only_numeric_features = only_numeric_features
        self.model = thr_alpha
        self.scaler_for_chi_squared = scaler_for_chi_squared
        self.selected_columns_ = []
        self.advanced_info_ = {}

    def fit(self, data, fill_value=9999):
        X, y = data['data'][data['feature_names']], data['data'][data['target_name']]

        if self.fillna:
            X = X.fillna(fill_value)
        else:
            X = X.dropna()
            y = y[X.index]

        if self.only_numeric_features:
            X = X.select_dtypes(np.number)

        self.X_columns_ = X.columns

        if self.scaler_for_chi_squared:
            X = self.scaler_for_chi_squared.fit_transform(X)

        if self.model == 'classification':
            scores = chi2(X, y)
        else:
            scores = f_regression(X, y)
        self.advanced_info_ = {col: score for col, score in zip(self.X_columns_, scores[1])}
        self.selected_columns_ = [col for col, score in zip(self.X_columns_, scores[1]) if score <= thr_alpha]


@dataclass
class Dataset:
    data: str
    features: List
    target: str
    id: int = 9999

    def get_Xy(self):
        data_ = pd.DataFrame(json.loads(self.data))
        return {
            'X': data_[self.features],
            'y': data_[self.target]
        }


@dataclass
class MlModel:
    model_type: str = 'RandomForestClassifier'
    params: Dict = field(
        default_factory=lambda: {'n_jobs': os.cpu_count() // 2, 'random_state': 88, 'class_weight': 'balanced'})
    id: int = 777

    def make_model(self, model_type, params):
        if model_type == 'RandomForestClassifier':
            return RandomForestClassifier().set_params(**params)
        if model_type == 'GradientBoostingClassifier':
            return GradientBoostingClassifier().set_params(**params)

    def get_model(self):
        return self.make_model(self.model_type, self.params)


@dataclass
class OptParams:
    model_type: str = 'RandomForestClassifier'
    params: Dict = field(init=False)

    def __post_init__(self):
        if self.model_type == 'RandomForestClassifier':
            self.params = {
                'n_estimators': {'start': 100, 'end': 500, 'step': 10},
                'max_depth': {'start': 1, 'end': 15, 'step': 3},
                'min_samples_split': {'start': 5, 'end': 25, 'step': 5}
            }

    def get_opt_space(self):
        return {
            k: hp.choice(label=k, options=np.linspace(v['start'], v['end'], v['step'], dtype=int)) for k, v in
            self.params.items()
        }


@dataclass
class Opt:
    data: Dataset
    params: OptParams
    pipeline: MlModel
    metric: field(default_factory=precision_score)
    trials: Trials
    early_stop: Dict = field(default_factory=lambda: {'iteration_stop_count': 100, 'percent_increase': 10})

    def objective(self, params, pipeline=None, X_train=None, y_train=None, metric=None):
        """
        Кросс-валидация с текущими гиперпараметрами

        :pipeline: модель
        :X_train: матрица признаков
        :y_train: вектор меток объектов
        :metric: callable, example lambda est, x, y: accuracy_score(y, est.predict(x))
        :return: средняя точность на кросс-валидации
        """

        pipeline = pipeline.set_params(**params)

        # задаём параметры кросс-валидации (3-фолдовая с перемешиванием)
        skf = ShuffleSplit(n_splits=3, random_state=1)

        # проводим кросс-валидацию
        score = cross_val_score(estimator=pipeline, X=X_train, y=y_train,
                                scoring=lambda est, x, y: metric(y, est.predict(x)), cv=skf, n_jobs=os.cpu_count() // 2)

        # возвращаем результаты, которые записываются в Trials()
        return {'loss': -score.mean(), 'params': params, 'status': STATUS_OK}

    def start_opt(self):
        best = fmin(
            # функция для оптимизации
            fn=partial(self.objective, pipeline=self.pipeline.get_model(),
                       X_train=self.data.get_Xy()['X'], y_train=self.data.get_Xy()['y'],
                       metric=self.metric),
            # пространство поиска гиперпараметров
            space=self.params.get_opt_space(),
            # алгоритм поиска
            algo=tpe.suggest,
            # число итераций
            # (можно ещё указать и время поиска)
            max_evals=250,
            # куда сохранять историю поиска
            trials=self.trials,
            # random state
            rstate=np.random.RandomState(1),
            # early stop
            early_stop_fn=no_progress_loss(**self.early_stop),
            # progressbar
            show_progressbar=True
        )
