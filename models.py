# тут прописать импорт необходимых библиотек
import pandas as pd
from sklearn.feature_selection import chi2, f_regression
import numpy as np

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


class ModelFit(object):
    pass
