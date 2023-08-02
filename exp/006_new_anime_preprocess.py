# %%
# unseen, seenのユーザに対してそれぞれ推論する

from tqdm.notebook import tqdm
from itertools import combinations
from IPython.display import display
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import Levenshtein
from tqdm import tqdm
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
import Levenshtein
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold

from atma15.eda import visualize_importance
from atma15.model import lgb_params, run_lgb

pl.Config.set_fmt_str_lengths(100)

# %%
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
anime = pl.read_csv(f"{input_dir}/anime.csv")
sub = pl.read_csv(f"{input_dir}/sample_submission.csv")

#%%


multilabel_cols = ["producers", "studios"]
multilabel_dfs = []
n_components = 10


for c in multilabel_cols:
    list_srs = anime[c].str.split(by=",").to_list()
    # MultiLabelBinarizerを使うと簡単に変換できるのでオススメです
    mlb = MultiLabelBinarizer()
    ohe_srs = mlb.fit_transform(list_srs)
    # ユニーク数が多いので、SVDで次元圧縮する
    svd = TruncatedSVD(n_components=n_components)
    svd_arr = svd.fit_transform(ohe_srs)
    col_df = pl.DataFrame(
        svd_arr,
        schema=[f"svd_{c}_{ix}" for ix in range(n_components)]
    )
    multilabel_dfs.append(col_df)

anime = pl.concat([anime] + multilabel_dfs,how="horizontal")

#%%

cat_cols = ["genres", "licensors"]
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
#%%

def get_original_work_name(df:pl.DataFrame, threshold=0.2) -> pl.DataFrame:
    df = df.to_pandas()

    _feature = df["japanese_name"].tolist()
    _n = df.shape[0]

    _disjoint_set = DisjointSet(list(range(_n)))
    for i, j in tqdm(combinations(range(_n), 2)):
        if _feature[i] is np.nan or _feature[j] is np.nan:
            lv_dist, jw_dist = 0.5, 0.5
        else:
            lv_dist = 1 - Levenshtein.ratio(_feature[i], _feature[j])
            jw_dist = 1 - Levenshtein.jaro_winkler(_feature[i], _feature[j])
        _d = (lv_dist + jw_dist) / 2

        if _d < threshold:
            _disjoint_set.merge(i, j)

    _labels = [None] * _n
    for subset in _disjoint_set.subsets():
        label = _feature[list(subset)[0]]
        for element in subset:
            _labels[element] = label
    df["original_work_name"] = _labels
    df = pl.DataFrame(df)

    return df

anime = get_original_work_name(anime)


#%%

anime.write_csv("../input/anime_preprocessed_v2.csv")


# %%
