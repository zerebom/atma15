# %%

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
from atma15.features.features import MemberRatio, TargetEncoding, ExplicitFeature
from atma15.model import lgb_params, run_lgb

pl.Config.set_fmt_str_lengths(100)

# %%
exp_name = Path(os.path.basename(__file__)).stem
input_dir = "../input"
output_dir = "../output"

exp_dir = f"{output_dir}/{exp_name}"
Path(exp_dir).mkdir(exist_ok=True, parents=True)

# %%
raw_train = pl.read_csv(f"{input_dir}/train.csv")
raw_test = pl.read_csv(f"{input_dir}/test.csv")
anime = pl.read_csv(f"{input_dir}/anime_preprocessed.csv")
sub = pl.read_csv(f"{input_dir}/sample_submission.csv")
# %%

str_cols = [
    "genres",
    "japanese_name",
    "aired",
    "producers",
    "licensors",
    "studios",
    "duration",
    "rating",
]
# cat_cols = ["type", "source"]
id_cols = ["anime_id", "user_id"]


# %%
train = raw_train.join(anime, how="left", on="anime_id")
test = raw_test.join(anime, how="left", on="anime_id")

target_col = "score"


# %%
train = train.with_row_count()

# %%


agg_dict = {
    # ("user_id", "score"): ["mean", "count"],
    # ("anime_id", "score"): ["mean", "count"],
    ("type", "score"): ["mean", "sum", "count", "std", "min", "max"],
    ("source", "score"): ["mean", "sum", "count", "std", "min", "max"],
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

features = [MemberRatio(), TargetEncoding("score", agg_dict, folds)]

# %%
for f in features:
    train = f.fit(train)
    test = f.transform(test)

# %%
train


# %%


X = train
y = X[["row_nr", "score"]].sort("row_nr")
y_bins = train["score"].to_numpy()


# %%
new_train = train


# %%
drop_cols = str_cols + id_cols

# %%
test_X = test.drop(drop_cols)
X = new_train.drop(drop_cols)


# %%

list1 = test_X.columns
list2 = X.columns

print("trainにないアイテム", [item for item in list1 if item not in list2])
print("testにないアイテム", [item for item in list2 if item not in list1])


# %%


# %%
X = X.drop("score").sort("row_nr")


oof_preds_list = []
preds_list = []
model_list = []

for fold_n, (train_idx, val_idx) in enumerate(folds.split(X, y_bins)):
    print(f"Fold {fold_n + 1} started.")

    X_train, X_val = X.filter(pl.col("row_nr").is_in(train_idx)), X.filter(
        pl.col("row_nr").is_in(val_idx)
    )

    X_train = X_train.drop(["row_nr"]).to_pandas()
    X_val = X_val.drop(["row_nr"]).to_pandas()

    y_train, y_val = (
        y.filter(pl.col("row_nr").is_in(train_idx))["score"].to_numpy(),
        y.filter(pl.col("row_nr").is_in(val_idx))["score"].to_numpy(),
    )
    model = run_lgb(X_train, y_train, X_val, y_val, lgb_params)

    oof_preds = model.predict(X_val)
    preds = model.predict(test_X)

    oof_pred_df = pl.DataFrame({"idx": val_idx, "preds": oof_preds})

    oof_preds_list.append(oof_pred_df)
    preds_list.append(preds)
    model_list.append(model)

# %%
#%%


visualize_importance(model_list, X_train)

# %%

oof_scores = pl.concat(oof_preds_list).sort("idx")["preds"].to_numpy()
y_true = train.sort("row_nr")["score"].to_numpy()

oof_rmse = str(round(np.sqrt(mean_squared_error(y_true, oof_scores)), 5)).replace(
    ".", "_"
)
oof_rmse
# %%
round(np.sqrt(mean_squared_error(y_true, oof_scores)), 5)

# %%

sub_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
path = f"{exp_dir}/submission_{sub_time}_{oof_rmse}.csv"
print(path)

pl.DataFrame(
    pl.Series(
        name="score",
        values=np.mean(preds_list, axis=0),
    )
).write_csv(path)

# %%
