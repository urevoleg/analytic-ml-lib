# тут прописать импорт необходимых библиотек
import pandas as pd

class ShareNans():
    def __init__(self, thr_nulls=0.1):
        self.thr_nulls = thr_nulls
        self.columns = []
        self.advanced_info = {}

    def fit(self, data):
        self.advanced_info = {col: data[col].isna().sum() / data.shape[0] for col in data.columns}
        self.columns = [col for col in self.advanced_info.keys() if self.advanced_info[col] <= self.thr_nulls]

    def fit_transform(self, data):
        self.fit(data)
        return data[self.columns]


# выполнять только для определенного набора признаков, нельзя для категориальных признаков
class ShareUniq():
    def __init__(self, thr_nunique=0.9):
        self.thr_nunique = thr_nunique
        self.columns = []
        self.advanced_info = {}

    def fit(self, data):
        self.advanced_info = {col: data[col].nunique() / data.shape[0] for col in data.columns}
        self.columns = [col for col in self.advanced_info.keys() if self.advanced_info[col] >= self.thr_nunique]

    def fit_transform(self, data):
        self.fit(data)
        return data[self.columns]


class SelectByCorr():
    def __init__(self, thr_corr=0.5):
        self.thr_corr = thr_corr
        self.columns = []
        self.advanced_info = {}

    def fit(self, data):
        self.advanced_info = {col: data[col].nunique() / data.shape[0] for col in data.columns}
        self.columns = [col for col in self.advanced_info.keys() if self.advanced_info[col] >= self.thr_corr]

    def fit_transform(self, data):
        self.fit(data)
        return data[self.columns]