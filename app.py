from models import Dataset, MlModel, Opt, OptParams
import pandas as pd
import numpy as np

import json
from pprint import pprint

from utils import to_tlg

import tqdm

from functools import partial

from sklearn.metrics import precision_score
from hyperopt import Trials

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("Start...")
    file_dataset = "df_σ02_350_08Х18Н10Т.json"
    target_mech = "σ0,2_350"
    norm_mech = "σ0,2_350_norm"
    target = "is_defect"

    with open(file_dataset, 'r') as f:
        df = pd.DataFrame(json.loads(f.read()))

    print("Dataset: read is done!")

    output = {}

    for thr in tqdm.tqdm([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1], desc="Thr"):
        df_train = df.assign(is_defect=lambda row: (row[target_mech] - thr * row[norm_mech] < 0).astype(int)).drop([target_mech, norm_mech], axis=1)
        share = df_train[target].mean()
        d = Dataset(data=json.dumps(df_train.select_dtypes(np.number).to_dict('records')),
                    features=df_train.select_dtypes(np.number).drop(target, axis=1).columns, target=target)
        m = MlModel(model_type='RandomForestClassifier')
        search_space = OptParams(model_type=type(m.get_model()).__name__)
        opt = Opt(data=d,
                  params=search_space,
                  pipeline=m,
                  metric=partial(precision_score, zero_division=0),
                  trials=Trials()
                  )

        opt.start_opt()

        output['thr'] += [thr]
        output['share'] += [share]
        output['best_trial'] += [opt.trials.best_trial['result']]

        #to_tlg(f"thr: {thr:.2f}\tshare: {share:.2f}\tprecision: {opt.trials.best_trial['result']['loss']:.3f}\n")

    with open('output_results.json', 'w') as f:
        f.write(json.dumps(output))
