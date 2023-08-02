# %%
# w2vFeatureを追加
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
from implicit.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold

from atma15.eda import visualize_importance
from atma15.features.features import W2V, CFEmb, Implict, MemberRatio, TargetEncoding
from atma15.model import lgb_params, run_lgb
from atma15.utils import seed_everything

seed_everything(42)
pl.Config.set_fmt_str_lengths(100)

exp_name = Path(os.path.basename(__file__)).stem
input_dir = "../input"
output_dir = "../output"

exp_dir = f"{output_dir}/{exp_name}"
Path(exp_dir).mkdir(exist_ok=True, parents=True)

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


raw_train = pl.read_csv(f"{input_dir}/train.csv")
raw_test = pl.read_csv(f"{input_dir}/test.csv")
anime = pl.read_csv(f"{input_dir}/anime_preprocessed_v2.csv")
sub = pl.read_csv(f"{input_dir}/sample_submission.csv")


anime = anime.with_columns(
    [
        pl.col("original_work_name").cast(pl.Categorical).cast(pl.Int32),
    ]
)


def join_anime(df, anime_df):
    return df.join(anime_df, how="left", on="anime_id")


if not "row_nr" in raw_train.columns:
    raw_train = raw_train.with_row_count()

if not "row_nr" in raw_test.columns:
    raw_test = raw_test.with_row_count()

train_y = raw_train[["row_nr", "score"]]
raw_X_test = join_anime(raw_test, anime)
raw_train = join_anime(raw_train, anime)


# %%

st_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gr_folds = GroupKFold(n_splits=5)


def split_generator(
    X: pl.DataFrame, y: pl.DataFrame, folds: StratifiedKFold | GroupKFold
) -> tuple[pl.DataFrame, pl.DataFrame, list, list, list, list]:
    for fold_n, (train_idx, val_idx) in enumerate(folds.split(X, y["score"].to_list())):
        print(len(train_idx), len(val_idx))

        X_train = X.filter(pl.col("row_nr").is_in(train_idx))
        X_val = X.filter(pl.col("row_nr").is_in(val_idx))
        print(len(X_train), len(X_val))

        y_train = y.filter(pl.col("row_nr").is_in(train_idx))["score"].to_list()
        y_val = y.filter(pl.col("row_nr").is_in(val_idx))["score"].to_list()

        yield X_train, X_val, y_train, y_val, train_idx, val_idx


agg_dict = {
    ("user_id", "score"): ["mean", "sum", "count", "std", "min", "max"],
    ("anime_id", "score"): ["mean", "sum", "count", "std", "min", "max"],
    # ("original_work_name", "score"): ["mean", "sum", "count", "std"],
    ("user_id", "members"): ["mean", "sum", "count", "std"],
    ("user_id", "watching"): ["mean", "sum", "count", "std"],
    ("user_id", "completed"): ["mean", "sum", "count", "std"],
    ("user_id", "on_hold"): ["mean", "sum", "count", "std"],
    ("user_id", "dropped"): ["mean", "sum", "count", "std"],
    ("anime_id", "members"): ["mean", "sum", "count", "std"],
    ("anime_id", "watching"): ["mean", "sum", "count", "std"],
    ("anime_id", "completed"): ["mean", "sum", "count", "std"],
    ("anime_id", "on_hold"): ["mean", "sum", "count", "std"],
    ("anime_id", "dropped"): ["mean", "sum", "count", "std"],
}


als = AlternatingLeastSquares(
    factors=64, regularization=0.05, alpha=2.0, calculate_training_loss=True
)
bpr = BayesianPersonalizedRanking()

features = [
    Implict(raw_train, raw_test, als, "als"),
    Implict(raw_train, raw_test, bpr, "bpr"),
    TargetEncoding(agg_dict, st_folds),
    # CFEmb(),
    W2V(),
    MemberRatio(),
]

drop_cols = str_cols + id_cols


def drop_untrainable_cols(df, drop_cols):
    df = df.drop(drop_cols)
    if "score" in df.columns:
        df = df.drop("score")

    if "row_nr" in df.columns:
        df = df.drop("row_nr")
    return df

#%%
for X_train, X_val, y_train, y_val, train_idx, val_idx in split_generator(
    raw_train, train_y, st_folds
):

    X_test = raw_X_test
    print(X_train.shape, X_val.shape, raw_X_test.shape)
    for f in features:
        X_train = f.fit(X_train)
        X_val = f.transform(X_val)
        X_test = f.transform(X_test)

    print(X_train.shape, X_val.shape, X_test.shape)

    X_num_train = drop_untrainable_cols(X_train, drop_cols)
    X_num_val = drop_untrainable_cols(X_val, drop_cols)
    X_num_test = drop_untrainable_cols(X_test, drop_cols)

    print(X_num_train.shape)

    model = run_lgb(X_num_train, y_train, X_num_val, y_val, lgb_params)
    oof_preds = model.predict(X_num_val)
    preds = model.predict(X_num_test)

    oof_pred_df = pl.DataFrame({"idx": val_idx, "preds": oof_preds})
    oof_preds_list.append(oof_pred_df)
    preds_list.append(preds)
    model_list.append(model)


# break
# %%
[len(p) for p in oof_preds_list]

#%%
visualize_importance(model_list, X_num_train, n_top=100)

# %%


def calc_oof_rmse(
    raw_train: pl.DataFrame, oof_preds_list: list[np.array]
) -> tuple[float, str, pl.DataFrame]:
    all_oof_df = pl.concat(oof_preds_list).sort("idx")
    indicies = all_oof_df["idx"].to_list()
    oof_scores = all_oof_df["preds"].to_numpy()
    y_true = (
        raw_train.filter(pl.col("row_nr").is_in(indicies))
        .sort("row_nr")["score"]
        .to_numpy()
    )

    oof_rmse = round(np.sqrt(mean_squared_error(y_true, oof_scores)), 5)
    oof_rmse_str = str(oof_rmse).replace(".", "_")
    print("oof_rmse_str:", oof_rmse_str)
    return oof_rmse, oof_rmse_str, all_oof_df

def save_pred(
    preds_list: list[np.ndarray], exp_dir: Path, oof_rmse: str
) -> pl.DataFrame:
    sub_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{exp_dir}/submission_{sub_time}_{oof_rmse}.csv"
    print(path)

    sub_df = pl.DataFrame(
        pl.Series(
            name="score",
            values=np.mean(preds_list, axis=0),
        )
    ).write_csv(path)
    return sub_df, path


# oof_rmse, oof_rmse_str, all_oof_df = calc_oof_rmse(raw_train, oof_preds_list)
sub_df, save_path = save_pred(preds_list, exp_dir, "unknown")
oof_path = Path(exp_dir) / (Path(save_path).stem + "_oof.csv")
# all_oof_df.with_columns(raw_train["score"]).write_csv(oof_path)


# %%
train_user_ids = raw_train["user_id"].unique().to_list()

raw_test = raw_test.with_columns(
    pl.when(pl.col("user_id").is_in(train_user_ids))
    .then(1)
    .otherwise(0)
    .alias("is_train_user")
)

# %%
plt.hist(
    raw_train.groupby("user_id").agg(pl.col("anime_id").count().alias("cnt"))["cnt"],
    bins=300,
)

# %%

another_pred_path = "/Users/wantedly565/ghq/github.com/zerebom/atma15/output/004_surprise/submission_2023_07_17_00_51_19_1.1906.csv"
another_pred_s = pl.read_csv(another_pred_path)["score"]

# %%
anime

# %%

sub_df = raw_test.with_columns(
    [
        pl.Series(np.mean(preds_list, axis=0)).alias("is_train_user_score"),
        another_pred_s.alias("is_not_train_user_score"),
    ]
).with_columns(
    pl.when(pl.col("is_train_user") == 1)
    .then(pl.col("is_train_user_score"))
    .otherwise(pl.col("is_not_train_user_score"))
    .alias("score")
)
# %%
pl.DataFrame(sub_df["score"]).write_csv(save_path)

# %%

plt.hist(sub_df["score"], bins=100, alpha=0.5)
# plt.hist(pl.concat(oof_preds_list).sort("idx")["preds"].to_numpy(), bins=100, alpha=0.5)

# %%
save_path

# %%
for pred in preds_list:
    plt.hist(pred, bins=100)
    plt.show()
