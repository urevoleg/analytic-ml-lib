from models import Dataset, MlModel, Opt, OptParams
import pandas as pd
import numpy as np

import json
from pprint import pprint

import tqdm

from sklearn.metrics import precision_score
from hyperopt import Trials


if __name__ == "__main__":
    file_dataset = "df_σ02_350_08Х18Н10Т.json"
    target = "is_defect"

    with open(file_dataset, 'r') as f:
        df = pd.DataFrame(json.loads(f.read()))

    for thr in tqdm([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1], desc="Thr"):
        df = df.assign(is_defect=lambda row: (row[target] - thr * row[target + '_norm']) < 0).drop([target, target + '_norm'], axis=1)
        share = df[target].mean()
        d = Dataset(data=json.dumps(df.select_dtypes(np.number).to_dict('records')),
                    features=df.select_dtypes(np.number).drop(target, axis=1).columns, target=target)
        m = MlModel(model_type='RandomForestClassifier')
        search_space = OptParams(model_type=type(m.get_model()).__name__)
        opt = Opt(data=d,
                  params=search_space,
                  pipeline=m,
                  metric=precision_score,
                  trials=Trials()
                  )

        print(thr, share)
        pprint(opt.trials.best_trial)
        print("\n")