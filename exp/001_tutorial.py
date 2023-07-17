# %%

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pl.Config.set_fmt_str_lengths(100)

# %%
exp_name = "001_tutorial"
input_dir = "../input"
output_dir = "../output"

exp_dir = f"{output_dir}/{exp_name}"

Path(exp_dir).mkdir(exist_ok=True, parents=True)

# %%
raw_train = pl.read_csv(f"{input_dir}/train.csv")
test = pl.read_csv(f"{input_dir}/test.csv")
anime = pl.read_csv(f"{input_dir}/anime.csv")
# %%
sub = pl.read_csv(f"{input_dir}/sample_submission.csv")
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


lgb_params: dict[str, str | float | int] = {
    "boosting_type": "gbdt",  # Gradient Boosting Decision Tree
    "objective": "regression",  # 回帰タスク
    "metric": "rmse",  # RMSE (Root Mean Square Error)
    "learning_rate": 0.1,  # 学習率
    "n_estimators": 10000,  # ツリーの数
    "max_depth": -1,  # ツリーの深さ制限なし
    "num_leaves": 31,  # ツリーの葉の最大数
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,  # 乱数シード
    "n_jobs": -1,  # 使用するCPUコア数（-1は全てのコアを使用）
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(
    train.drop(drop_cols).to_pandas(),
    train["score"].to_pandas(),
    eval_set=[(valid.drop(drop_cols).to_pandas(), valid["score"].to_pandas())],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=30),
    ],
)
# %%
preds = model.predict(test.drop(["user_id", "anime_id"]).to_pandas())

sub["score"]
# %%
pl.DataFrame(pl.Series(values=preds, name="score")).write_csv(f"{exp_dir}/submission.csv")

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
