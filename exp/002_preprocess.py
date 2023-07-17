# %%

import os
import re
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split,KFold

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
anime = pl.read_csv(f"{input_dir}/anime.csv")
sub = pl.read_csv(f"{input_dir}/sample_submission.csv")
# %%

# %%
cat_cols = ["genres", "producers", "studios", "licensors"]
for cat_col in cat_cols:
    cat_list = set(np.concatenate(anime[cat_col].str.split(", ").to_numpy()))
    anime = anime.with_columns(
        [pl.col(cat_col).str.split(", ").list.lengths().alias(f"has_{cat_col}_cnt")]
    )
    for cat in cat_list:
        anime = anime.with_columns(
            pl.col(cat_col).str.contains(cat).alias(f"has_{cat}")
        )
# %%

rating_dict = {
    "G - All Ages": 0,
    "PG - Children": 1,
    "R+ - Mild Nudity": 2,
    "PG-13 - Teens 13 or older": 3,
    "Rx - Hentai": 4,
    "R - 17+ (violence & profanity)": 5,
    "Unknown": -1,
}

anime = anime.with_columns(pl.col("rating").map_dict(rating_dict).alias("rating_cd"))
# %%

anime = anime.with_columns(
    pl.col("episodes").str.replace("Unknown", "-1").cast(pl.Int32).alias("episodes")
)


# %%

def convert_to_minutes(time_str):
    # 単位の時間（時）と分を抽出
    hours = re.findall(r"(\d+) hr.", time_str)
    minutes = re.findall(r"(\d+) min.", time_str)

    total_minutes = 0

    # 時間が存在する場合、分に変換
    if hours:
        total_minutes += int(hours[0]) * 60

    # 分が存在する場合、加算
    if minutes:
        total_minutes += int(minutes[0])

    return total_minutes


anime = anime.with_columns(
    pl.col("duration").apply(convert_to_minutes).alias("duration_min")
)
# %%


def get_years(aired_str):
    # "to" で分割し、日付範囲の両端を取得
    dates = aired_str.split(" to ")

    # 開始年を取得
    start_year = datetime.strptime(dates[0], "%b %d, %Y").year

    # 終了年が存在する場合は取得、存在しない場合は開始年を終了年とする
    end_year = (
        datetime.strptime(dates[1], "%b %d, %Y").year if len(dates) > 1 else start_year
    )

    return start_year, end_year


anime = anime.with_columns(
    pl.col("aired").apply(get_years).list.get(0).alias("start_year"),
    pl.col("aired").apply(get_years).list.get(0).alias("end_year"),
)


# %%


def get_days(aired_str):
    # "to" で分割し、日付範囲の両端を取得
    if "Unknown" in aired_str or "?" in aired_str:
        return -1

    try:
        dates = aired_str.split(" to ")

        # 開始日を取得
        start_date = datetime.strptime(dates[0], "%b %d, %Y")

        # 終了日が存在する場合は取得、存在しない場合は開始日を終了日とする
        end_date = (
            datetime.strptime(dates[1], "%b %d, %Y") if len(dates) > 1 else start_date
        )

        # 放映日数を算出（1日を加算することで、開始日と終了日を両方含む）
        aired_days = (end_date - start_date).days + 1

        return aired_days
    except:
        return -1


anime = anime.with_columns(pl.col("aired").apply(get_days).alias("aired_days"))
anime = anime.with_columns(
    [
        pl.col("type").cast(pl.Categorical).cast(pl.Int32),
        pl.col("source").cast(pl.Categorical).cast(pl.Int32),
    ]
)
#%%



# %%
train = raw_train.join(anime, how="left", on="anime_id")
test = raw_test.join(anime, how="left", on="anime_id")


"""
for fold in range(5):
    train,val = ...
    for (key_col, target_col), agg_func_list in agg_dict.items():
        for agg_func in agg_func_list:

"""

key_col = "user"
id_col = key_col + "_id"
target_col = "score"


agg_dict = {
    ("user_id", "score"): ["mean", "sum", "count", "std", "min", "max"],
    ("anime_id", "score"): ["mean", "sum", "count", "std", "min", "max"],
    # ("user_id", "score"): ["count"],
    # ("anime_id", "score"): ["count"],
}

agg_expr = {
    "mean": pl.col(target_col).mean(),
    "sum": pl.col(target_col).sum(),
    "count": pl.col(target_col).count(),
    "std": pl.col(target_col).std(),
    "min": pl.col(target_col).min(),
    "max": pl.col(target_col).max(),
}


# train.with_row_count()
X = train.with_row_count()
y = X[["row_nr", "score"]]
y_bins = train["score"].to_numpy()
#%%




# %%


# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = KFold(n_splits=5, shuffle=True, random_state=42)

valid_list = []

for fold_n, (train_idx, val_idx) in enumerate(folds.split(X, y_bins)):
    print(f"Fold {fold_n + 1} started.")

    X_train, X_val = X.filter(pl.col("row_nr").is_in(train_idx)), X.filter(
        pl.col("row_nr").is_in(val_idx)
    )

    for (key_col, target_col), agg_func_list in agg_dict.items():
        for agg_func_str in agg_func_list:
            te_col = f"te_{key_col}_{agg_func_str}_{target_col}"
            print(te_col)

            d = (
                X_train.groupby([id_col])
                .agg(agg_expr[agg_func_str])
                .to_pandas()
                .set_index(id_col)
                .to_dict()[target_col]
            )

            # train = train.with_columns(pl.col(id_col).map_dict(d).alias(te_col))
            X_val = X_val.with_columns(pl.col(id_col).map_dict(d).alias(te_col))
            print(len(X_val.columns))
    valid_list.append(X_val)

# %%
# new_train = pl.concat(valid_list)
new_train = pl.concat(valid_list).drop("score")
new_train2 = train.drop("score").with_row_count()
cols = ["user_id", "anime_id", "row_nr", "has_Drama", "start_year"]
# %%


# %%

for key_col in ["anime_id", "user_id"]:
    agg_cols = [col for col in new_train.columns if col.startswith(f"te_{key_col}")]
    agg_df = new_train.groupby(key_col).agg(pl.col(agg_cols).mean())
    test = test.join(agg_df, how="left", on=key_col)
# %%



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
drop_cols = str_cols + id_cols

# %%
test_X = test.drop(drop_cols)
X = new_train.drop(drop_cols)


# %%

list1 = test_X.columns
list2 = X.columns

print([item for item in list1 if item not in list2])
print([item for item in list2 if item not in list1])


# %%

lgb_params: dict[str, str | float | int] = {
    "boosting_type": "gbdt",  # Gradient Boosting Decision Tree
    "objective": "regression",  # 回帰タスク
    "metric": "rmse",  # RMSE (Root Mean Square Error)
    "learning_rate": 0.1,  # 学習率
    "n_estimators": 10000,  # ツリーの数
    "max_depth": -1,  # ツリーの深さ制限なし
    "num_leaves": 63,  # ツリーの葉の最大数
    "subsample": 0.8,
    "subsample_fraq": 3,
    "colsample_bytree": 0.8,
    "min_child_samples": 6,
    "min_split_gain": 0.3,
    "random_state": 42,  # 乱数シード
    "n_jobs": -1,  # 使用するCPUコア数（-1は全てのコアを使用）
}


def run_lgb(X_train, y_train, X_val, y_val, params):
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


# %%


# %%
X = X.sort("row_nr")


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


def visualize_importance(models, feat_train_df):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.2)))
    sns.boxenplot(
        data=feature_importance_df,
        y="column",
        x="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    fig.tight_layout()
    ax.grid()
    return fig, ax


visualize_importance(model_list, X_train)

# %%

oof_scores = pl.concat(oof_preds_list).sort("idx")["preds"].to_numpy()
y_true = train["score"].to_numpy()
print(np.sqrt(mean_squared_error(y_true, oof_scores)))


# %%
pl.DataFrame(
    pl.Series(
        name="score",
        values=np.mean(preds_list, axis=0),
    )
).write_csv(f"{exp_dir}/submission.csv")

# %%
plt.hist(train["score"], bins=10)

# %%
plt.hist(pl.concat(preds_list)["preds"], bins=10)


# %%
# pl.DataFrame(pl.Series(values=preds, name="score")).write_csv(
#     f"{exp_dir}/submission.csv"
# )
# %%


# %%

np.concatenate(preds_list)


# %%


tmp_df = valid.groupby("user_id").agg(pl.col("te_anime_id").mean())


# %%
train["score"]


# %%
train


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
