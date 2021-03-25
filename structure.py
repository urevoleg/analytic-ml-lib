# ### Библиотеки
import re
import datetime as dt
import json

import pandas as pd
import numpy as np


class DatasetStructure:
    def __init__(self, id_dataset=999, descr="Template dataset", unit_index='index', target_name='target', time_line='ts',
                 author='User', dtypes_mapping={}, names_mapping={}, pandas_dataframe=None):
        self.id_dataset = id_dataset
        self.descr = descr
        self.unit_index = unit_index
        self.target_name = target_name
        self.time_line = time_line
        self.author = author
        self.dt_create = self.get_datetime_()
        self.dtypes_mapping = {
            'float': 'NUMBER',
            'int': 'NUMBER',
            'object': 'VARCHAR2',
            'datetime': 'DATE'
        }
        if dtypes_mapping != {}:
            self.dtypes_mapping = dtypes_mapping

        self.names_mapping = names_mapping
        self.dataset = pandas_dataframe
        self.columns_dtypes = {}
        self.data = []

    def get_datetime_(self, dt_fmt='%d.%m.%Y %H:%M:%S'):
        return dt.datetime.now().strftime(dt_fmt)

    def get_mapping_(self, k, null, d):
        return d.get(k, null)

    def make_data_(self, dataset):
        """dataframe to list of dicts"""
        return dataset.to_dict('records')

    def make_columns_dtypes_(self, dataset):
        res = {}
        for k, v in self.dataset.dtypes.to_dict().items():
            res[k] = [self.get_mapping_(k, 'Нет описания', self.names_mapping)] + \
                     [self.get_mapping_(k_map, 'None', self.dtypes_mapping) for k_map in self.dtypes_mapping.keys() if
                      re.match(re.compile(k_map), str(v))]
        return res

    def create_structure(self, is_return=False):
        self.data = self.make_data_(self.dataset)
        self.columns_dtypes = self.make_columns_dtypes_(self.dataset)

        if is_return:
            return {
                'id': self.id_dataset,
                'descr': self.descr,
                'columns_dtypes': self.columns_dtypes,
                'unit_index': self.unit_index,
                'target_name': self.target_name,
                'time_line': self.time_line,
                'dt_create': self.get_datetime_(),
                'author': self.author,
                'data': self.data
            }


if __name__ == '__main__':
    with open('titanic.json', 'r') as f:
        df = pd.DataFrame(json.loads(f.read()))

    print(df.head())

    template = dict(id_dataset=99, descr="""Common descr of dataset""", unit_index=['PassengerId'], target_name='Survived',
                    time_line=None,
                    author='User', dtypes_mapping={}, names_mapping={}, pandas_dataframe=df)

    ds = DatasetStructure(**template)
    ds.create_structure(is_return=False)

    print(ds.columns_dtypes.keys())