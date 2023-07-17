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
from atma15.features.features import MemberRatio, TargetEncoding
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
plt.hist(train["score"], bins=10)

# %%
plt.hist(pl.concat(preds_list)["preds"], bins=10)


# %%


tmp_df = valid.groupby("user_id").agg(pl.col("te_anime_id").mean())


# %%


class TargetEncoder:
    def __init__(self, columns_names):
        self.columns_names = columns_names
        self.learned_values = {}

    def fit(self, X, y=None):
        X_ = X.copy()
        self.learned_values = {}
        for column in self.columns_names:
            learned_values = X_.groupby(column)["target"].mean()
            self.learned_values[column] = learned_values

    def transform(self, X):
        """
        transform data using target encoder
        """
        X_ = X.copy()
        for column in self.columns_names:
            X_[column] = X_[column].map(self.learned_values.get(column, np.nan))
        return X_


# %%


# %%


# %%
def attach_simple_featre(df):
    df = df.with_columns(
        [
            pl.col("score").mean().over("user_id").alias("user_score_mean"),
            pl.col("score").min().over("user_id").alias("user_score_min"),
            pl.col("score").max().over("user_id").alias("user_score_max"),
            pl.col("score").std().over("user_id").alias("user_score_std"),
            pl.col("score").count().over("user_id").alias("user_score_cnt"),
            pl.col("score").mean().over("anime_id").alias("anime_score_mean"),
            pl.col("score").min().over("anime_id").alias("anime_score_min"),
            pl.col("score").max().over("anime_id").alias("anime_score_max"),
            pl.col("score").std().over("anime_id").alias("anime_score_std"),
            pl.col("score").count().over("anime_id").alias("anime_score_cnt"),
        ]
    )
    return df


# train = attach_simple_featre(train)
# test = test.join(train, on="anime_id", how="inner")
# %%
agg_name = "user"


def agg(df, agg_name):
    df = df.groupby(agg_name + "_id").agg(
        [
            pl.col("score").mean().alias(f"{agg_name}_score_mean"),
            pl.col("score").count().alias(f"{agg_name}_score_count"),
            pl.col("score").std().alias(f"{agg_name}_score_std"),
            pl.col("score").max().alias(f"{agg_name}_score_max"),
            pl.col("score").min().alias(f"{agg_name}_score_min"),
        ]
    )
    return df


user_agg_df = agg(raw_train, "user")
anime_agg_df = agg(raw_train, "anime")
anime_agg_df
# %%
test

# %%
raw_train = raw_train.join(user_agg_df, on="user_id", how="left").join(
    anime_agg_df, on="anime_id", how="left"
)
test = test.join(user_agg_df, on="user_id", how="left").join(
    anime_agg_df, on="anime_id", how="left"
)
# %%

train, valid = train_test_split(raw_train, test_size=0.2, random_state=42)

# %%
drop_cols = ["user_id", "anime_id", "score"]


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# DataFrameとしてデータをロード（ここではanime_id, user_id, ratingという名前の列を仮定）


# ratingを10で割って整数にし、層化を可能にします（この方法は問題に応じて適宜調整してください）
y_bins = np.floor(y / 10)

# cross validationのループ
for fold_n, (train_idx, val_idx) in enumerate(folds.split(X, y, y_bins)):
    print(f"Fold {fold_n + 1} started at {time.ctime()}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = run_lgb(X_train, y_train, X_val, y_val)

    y_pred_val = model.predict(X_val)
    print(
        f"Fold {fold_n + 1}. RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_val))}\n"
    )
# %%
preds = model.predict(test.drop(["user_id", "anime_id"]).to_pandas())

sub["score"]
# %%
pl.DataFrame(pl.Series(values=preds, name="score")).write_csv(
    f"{exp_dir}/submission.csv"
)

# %%
from typing import Union

# %%

num_cols = [
    "members",
    "watching",
    "completed",
    "on_hold",
    "dropped",
    "plan_to_watch",
    "producers",
    "studios",
    "rating",
    "source",
]


cat_cols = ["type", "source"]

multi_label_cols = [
    "genres",
]

#  'episodes',
#  'japanese_name',
#  'aired',
#  'licensors',
#  'duration',
