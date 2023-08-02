#%%
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from atma15.eda import visualize_importance
from atma15.features.features import MemberRatio, TargetEncoding
from atma15.model import lgb_params, run_lgb
import pandas as pd

#%%

exp_name = Path(os.path.basename(__file__)).stem
input_dir = Path("../input")
output_dir = Path("../output")

exp_dir = f"{output_dir}/{exp_name}"

Path(exp_dir).mkdir(exist_ok=True, parents=True)

# %%
reader = Reader(rating_scale=(1, 10))

train_df = pd.read_csv(input_dir / 'train.csv')

train_data = Dataset.load_from_df(train_df[['user_id', 'anime_id', 'score']], reader)
train_data = train_data.build_full_trainset()

algo = KNNBaseline()
algo.fit(train_data)


# Run 5-fold cross-validation and print results.
# _ = cross_validate(algo, train_data, measures=['RMSE',], cv=5, verbose=True)
#%%

# predictions = algo.test(testset)

# %%

# Load the test dataset
test_df = pd.read_csv(input_dir / 'test.csv')
test_df['score'] = 0

# Convert the test dataset to the surprise format
test_set = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()

# Predict ratings for the testset
predictions = algo.test(test_set)

submission = pd.read_csv(input_dir /'sample_submission.csv')
#%%

oof_rmse = "1.1906"
sub_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path = f"{exp_dir}/submission_{sub_time}_{oof_rmse}.csv"

#%%
# Extract the predicted ratings and add them to the test dataframe
submission['score'] = [pred.est for pred in predictions]

submission.to_csv(path, index=False)


# %%
