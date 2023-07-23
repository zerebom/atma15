# %%
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


# %%
# %%
oof_path = "/home/wantedly565/repo/atma15/output/005_split_cv/submission_2023_07_22_14_28_17_1.19167_oof.csv"
oof_path = "/home/wantedly565/repo/atma15/output/005_split_cv/submission_2023_07_22_14_28_17_1.19167_oof.csv"

sns.kdeplot(
    data=oof_df.to_pandas().sample(5000),
    x="score",
    y="preds",
    cmap="viridis",
    fill=True,
)
sns.kdeplot(
    data=oof_df.to_pandas().sample(5000),
    x="score",
    y="preds",
    cmap="viridis",
    fill=True,
)
# %%

oof_df = oof_df.with_columns((pl.col("preds") - pl.col("score")).alias("diff"))

# %%
plt.hist(oof_df["diff"], bins=100)


exp_name = Path(os.path.basename(__file__)).stem
input_dir = "../input"
output_dir = "../output"
raw_train = pl.read_csv(f"{input_dir}/train.csv")
exp_dir = f"{output_dir}/{exp_name}"
Path(exp_dir).mkdir(exist_ok=True, parents=True)

# %%
oof_df = oof_df.with_columns(raw_train)
# %%
user_agg_df = oof_df.groupby("user_id").agg(
    [
        pl.col("diff").std().alias("diff_std"),
        pl.col("diff").mean().alias("diff_mean"),
        pl.col("anime_id").count().alias("cnt"),
    ]
)

sns.kdeplot(
    data=user_agg_df.to_pandas(),
    x="cnt",
    y="diff_mean",
    cmap="viridis",
    fill=True,
)

# %%
pl.Config.set_tbl_rows(100)

plt.hist(user_agg_df.sort("cnt", descending=True).tail(300)["diff_mean"])


# %%

diff_df = (
    oof_df.with_columns(((pl.col("score") - pl.col("preds")) ** 2).alias("mse"))
    .groupby("user_id")
    .agg(
        pl.col("mse").sum().alias("sum"),
        pl.col("mse").count().alias("count"),
        pl.col("mse").mean().alias("mean"),
        pl.col("score").std().alias("score_std"),
    )
    .sort("sum", descending=True)
)
diff_df

#%%



# %%
big_diff_users = diff_df.head(3)["user_id"].unique().to_list()

#%%
big_diff_users

#%%

#%%

anime = pl.read_csv(f"{input_dir}/anime.csv")

bd_df = raw_train.filter(pl.col("user_id").is_in(big_diff_users))
bd_df = bd_df.join(oof_df[["user_id","anime_id","preds","diff"]],how="left",on=["user_id","anime_id"])
bd_df = bd_df.join(anime,how="left",on=["anime_id"])

#%%

#%%
bd_df




# %%
