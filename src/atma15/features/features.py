import random

import numpy as np
import pandas as pd
import polars as pl
from gensim.models import word2vec
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import BaseCrossValidator
from tqdm import tqdm


class Feature:
    def fit(self, df):
        pass

    def transform(self, df):
        pass


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class Implict(Feature):
    def __init__(self, train_df, test_df, model, prefix: str) -> None:
        super().__init__()
        src_df = pl.concat(
            (train_df[["user_id", "anime_id"]], test_df[["user_id", "anime_id"]])
        )

        self.uid2idx = {
            k: idx for idx, k in enumerate(src_df["user_id"].unique().sort())
        }
        self.aid2idx = {
            k: idx for idx, k in enumerate(src_df["anime_id"].unique().sort())
        }
        self.idx2uid = {v: k for k, v in self.uid2idx.items()}
        self.idx2aid = {v: k for k, v in self.aid2idx.items()}

        src_df = src_df.with_columns(
            [
                pl.col("user_id").map_dict(self.uid2idx).alias("user_idx"),
                pl.col("anime_id").map_dict(self.aid2idx).alias("anime_idx"),
            ]
        )
        self.model = model
        self.prefix = prefix
        self.csr_data = csr_matrix(
            (
                np.ones(len(src_df)),
                (src_df["user_idx"].to_numpy(),src_df["anime_idx"].to_numpy()),
            )
        )
        self.src_df = src_df

    def fit(self, df):
        self.model.fit(self.csr_data)
        self.src_df = self.calc_sim(self.src_df)

        self.anime_emb_df = pl.DataFrame(self.model.item_factors).select(
            [
                pl.all().prefix(f"{self.prefix}_a_"),
                pl.Series(self.aid2idx.keys()).alias("anime_id"),
            ]
        )

        self.user_emb_df = pl.DataFrame(self.model.user_factors).select(
            [pl.all().prefix(f"{self.prefix}_u_"), pl.Series(self.uid2idx.keys()).alias("user_id")]
        )
        key_cols = ["user_id", "anime_id"]
        df = df.join(self.anime_emb_df, on="anime_id", how="left")
        df = df.join(self.user_emb_df, on="user_id", how="left")
        df = df.join(
            self.src_df[key_cols + [f"{self.prefix}_sim"]], on=key_cols, how="left"
        )
        return df

    def transform(self, df):
        key_cols = ["user_id", "anime_id"]
        df = df.join(self.anime_emb_df, on="anime_id", how="left")
        df = df.join(self.user_emb_df, on="user_id", how="left")
        df = df.join(
            self.src_df[key_cols + [f"{self.prefix}_sim"]], on=key_cols, how="left"
        )
        return df

    def calc_sim(self, df):
        item_arr = self.model.item_factors
        user_arr = self.model.user_factors
        sim_list = []
        for aid, uid in zip(df["anime_idx"].to_numpy(), df["user_idx"].to_numpy()):
            sim_list.append(cos_sim(item_arr[aid], user_arr[uid]))

        df = df.with_columns(pl.Series(sim_list).alias(f"{self.prefix}_sim"))
        return df


class CFEmb(Feature):
    def __init__(self):
        pass

    def fit(self, df):
        # アニメの出現数をカウント
        sim_anime_df = pl.read_csv("/home/wantedly565/repo/atma15/input/sim_anime.csv")

        sim_arr = (
            sim_anime_df.sort(["anime_id", "anime_id_right"])
            .pivot(
                values="weight",
                index="anime_id",
                columns="anime_id_right",
                aggregate_function="mean",
            )
            .drop("anime_id")
            .fill_null(0)
            .fill_nan(0)
            .to_numpy()
        )

        svd = TruncatedSVD(n_components=10)
        svd_arr = svd.fit_transform(sim_arr)

        anime_name = np.reshape(
            sim_anime_df.sort("anime_id")["anime_id"].unique().to_numpy(), (-1, 1)
        )

        self.cf_emb_df = pl.concat(
            [
                pl.DataFrame(anime_name.astype(str), schema=["anime_id"]),
                pl.DataFrame(
                    svd_arr,
                    schema=["cf_svd_" + str(i) for i in range(svd_arr.shape[1])],
                ),
            ],
            how="horizontal",
        )

        df = df.join(self.cf_emb_df, how="left", on="anime_id")
        return df

    def transform(self, df):
        df = df.join(self.cf_emb_df, how="left", on="anime_id")
        return df


class CF(Feature):
    def __init__(self) -> None:
        super().__init__()
        sim_anime_df = pl.read_csv(
            "/home/wantedly565/repo/atma15/input/sim_anime_df.csv"
        )
        self.top_sim_df = (
            sim_anime_df.with_columns(
                pl.col("weight").rank(descending=True).over("anime_id").alias("rank")
            )
            .sort(["anime_id", "rank"])
            .filter(pl.col("rank") < 50)
        )

    def fit(self, df):
        self.sum_df = (
            df[["user_id", "anime_id", "score"]]
            .join(
                self.top_sim_df,
                how="left",
                left_on=["anime_id"],
                right_on=["anime_id_right"],
            )
            .groupby(["user_id", "anime_id"])
            .agg(
                [
                    pl.col("weight").count().alias("weight_cnt"),
                    (pl.col("weight")).mean().alias("weight_mean"),
                    (pl.col("weight")).sum().alias("weight_sum"),
                    (pl.col("weight")).min().alias("weight_min"),
                    (pl.col("weight")).max().alias("weight_max"),
                ]
            )
        )
        df = df.join(self.sum_df, how="left", on=["user_id", "anime_id"])
        return df

    def transform(self, df):
        df = df.join(self.sum_df, how="left", on=["user_id", "anime_id"])
        return df


class W2V(Feature):
    def __init__(
        self, n_components: int = 20, window: int = 5, embedding_size: int = 32
    ) -> None:
        super().__init__()
        self.window = window
        self.embedding_size = embedding_size
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.svg_fitted = False

    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        self.user_factors, self.item_factors = self.calc_w2v_features_dict(df)

        self.svd_user_factors = self.compress_dim(self.user_factors)
        self.svd_item_factors = self.compress_dim(self.item_factors)

        df = self.calc_sim(df, self.user_factors, self.item_factors)
        df = self.attach_emb_by_cols(df, "user_id", self.svd_user_factors)
        df = self.attach_emb_by_cols(df, "anime_id", self.svd_item_factors)

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.calc_sim(df, self.user_factors, self.item_factors)
        df = self.attach_emb_by_cols(df, "user_id", self.svd_user_factors)
        df = self.attach_emb_by_cols(df, "anime_id", self.svd_item_factors)
        return df

    def calc_sim(
        self, df: pl.DataFrame, user_factors: dict, item_factors: dict
    ) -> pl.DataFrame:
        factores = df.with_columns(
            [
                pl.col("user_id").map_dict(user_factors).alias("user_factors"),
                pl.col("anime_id").map_dict(item_factors).alias("item_factors"),
            ]
        )[["user_factors", "item_factors"]].to_numpy()

        l = []
        for f in tqdm(factores):
            l.append(self.cosine_similarity(f[0], f[1]))

        df = df.with_columns(pl.Series(l).alias("w2v_sim"))
        return df

    def attach_emb_by_cols(
        self, df: pl.DataFrame, key_col: str, emb_dic: dict
    ) -> pl.DataFrame:
        df = (
            df.with_columns([pl.col(key_col).map_dict(emb_dic).alias("emb_dic")])
            .with_columns(
                [
                    pl.col("emb_dic").list.get(i).alias(f"emb_{key_col}_{i}")
                    for i in range(self.n_components)
                ]
            )
            .drop("emb_dic")
        )
        return df

    def compress_dim(self, factors):
        if self.svg_fitted:
            svd_arr = self.svd.transform(np.array(list(factors.values())))
        else:
            svd_arr = self.svd.fit_transform(np.array(list(factors.values())))
            self.svg_fitted = True

        svg_factors = {k: v for k, v in zip(factors.keys(), svd_arr)}
        return svg_factors

    def calc_w2v_features_dict(
        self, train_df: pl.DataFrame, SEED=42
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        df = train_df.to_pandas()

        anime_ids = df["anime_id"].unique().tolist()
        user_anime_list_dict = {
            user_id: anime_ids.tolist()
            for user_id, anime_ids in df.groupby("user_id")["anime_id"]
        }

        # スコアを考慮する場合
        # 今回は1～10のレーティングなので、スコアが5のアニメは5回、スコアが10のアニメは10回、タイトルをリストに追加する
        title_sentence_list = []
        for user_id, user_df in df.groupby("user_id"):
            user_title_sentence_list = []
            for anime_id, anime_score in user_df[["anime_id", "score"]].values:
                for i in range(anime_score):
                    user_title_sentence_list.append(anime_id)
            title_sentence_list.append(user_title_sentence_list)

        # ユーザごとにshuffleしたリストを作成
        shuffled_sentence_list = [
            random.sample(sentence, len(sentence)) for sentence in title_sentence_list
        ]  ## <= 変更点

        # 元のリストとshuffleしたリストを合わせる
        train_sentence_list = title_sentence_list + shuffled_sentence_list

        # word2vecのパラメータ
        w2v_params = {
            "vector_size": self.embedding_size,  ## <= 変更点
            "window": self.window,
            "seed": SEED,
            "min_count": 1,
            "workers": 32,
        }

        # word2vecのモデル学習
        print("train")
        model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

        print("trained")
        # ユーザーごとの特徴ベクトルと対応するユーザーID
        user_factors = {
            user_id: np.mean(
                [model.wv[anime_id] for anime_id in user_anime_list], axis=0
            )
            for user_id, user_anime_list in user_anime_list_dict.items()
        }

        # アイテムごとの特徴ベクトルと対応するアイテムID
        item_factors = {aid: model.wv[aid] for aid in anime_ids}

        return user_factors, item_factors

    @staticmethod
    def cosine_similarity(a, b):
        try:
            cos = np.dot(a, b) / (norm(a) * norm(b))
            return cos
        except:
            return np.nan


class SeverScale(Feature):
    def __init__(self):
        pass

    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.with_columns(
                pl.col("te_anime_id_mean_score").mean().alias("user_anime_mean_mean")
            )
            .with_columns(
                (
                    (
                        # ユーザの評価値の平均 - ユーザが見てるアニメの平均の平均(どれくらい人気なアイテムをみるか)
                        # これが高いと、甘口、低いと辛口
                        pl.col("te_user_id_mean_score")
                        - pl.col("user_anime_mean_mean")
                    ).alias("severity_scale")
                )
            )
            .with_columns(
                # アニメの平均評価にユーザの甘口・辛口度合いを足し引き
                (pl.col("te_anime_id_mean_score") + pl.col("severity_scale")).alias(
                    "caribrated_anime_mean_score"
                )
            )
        )
        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df)


# class ExplicitFeature:
#     def __init__(self, algo=KNNBaseline()):
#         self.algo = algo
#         self.feat_name = algo.__class__.__name__

#     def fit(self, df: pl.DataFrame) -> pl.DataFrame:
#         input_df = df.to_pandas()

#         reader = Reader(rating_scale=(1, 10))
#         data = Dataset.load_from_df(input_df[["user_id", "anime_id", "score"]], reader)
#         train_data = data.build_full_trainset()

#         self.algo.fit(train_data)

#         print("train Predict")
#         train_predictions = self.algo.test(train_data.build_testset())
#         est_train_ratings = [pred.est for pred in train_predictions]

#         df = df.with_columns(pl.Series(est_train_ratings).alias(self.feat_name))
#         return df

#     def transform(self, df: pl.DataFrame) -> pl.DataFrame:
#         input_df = df.to_pandas()
#         input_df["score"] = 0
#         reader = Reader(rating_scale=(1, 10))
#         test_data = Dataset.load_from_df(
#             input_df[["user_id", "anime_id", "score"]], reader
#         )
#         # testset = test_data.construct_testset(input_df[['user_id', 'anime_id', 'score']].values.tolist())  # transform to testset format

#         print("test Predict")
#         predictions = self.algo.test(test_data.build_full_trainset().build_testset())
#         est_ratings = [pred.est for pred in predictions]
#         df = df.with_columns(pl.Series(est_ratings).alias(self.feat_name))
#         return df


class MemberRatio(Feature):
    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            [
                (pl.col("watching") / pl.col("members")).alias("watching_rate"),
                (pl.col("completed") / pl.col("members")).alias("completed_rate"),
                (pl.col("on_hold") / pl.col("members")).alias("on_hold_rate"),
                (pl.col("dropped") / pl.col("members")).alias("dropped_rate"),
            ]
        )
        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df)


class TargetEncoding(Feature):
    def __init__(
        self,
        agg_dict: dict[tuple[str, str], list[str]],
        fold_instance: BaseCrossValidator,
        min_count: int = 5,
    ):
        self.agg_dict = agg_dict
        self.fold_instance = fold_instance
        self.min_count = min_count

    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        self.te_dict = {}
        user_ids = (
            df.groupby("user_id")
            .agg(pl.col("anime_id").count().alias("cnt"))
            .filter(pl.col("cnt") >= self.min_count)["user_id"]
            .unique()
            .to_list()
        )

        input_df = df.filter(pl.col("user_id").is_in(user_ids))

        for (key_col, target_col), agg_func_list in self.agg_dict.items():
            agg_expr = self.create_agg_expr(target_col)
            for agg_func_str in agg_func_list:
                te_col = f"te_{key_col}_{agg_func_str}_{target_col}"

                d = (
                    input_df.groupby([key_col])
                    .agg(agg_expr[agg_func_str])
                    .to_pandas()
                    .set_index(key_col)
                    .to_dict()[target_col]
                )

                df = df.with_columns(pl.col(key_col).map_dict(d).alias(te_col))
                self.te_dict[te_col] = d

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for te_col, d in self.te_dict.items():
            key_col = te_col.split("_")[1]

            if key_col == "user" or key_col == "anime":
                key_col += "_id"

            if key_col == "original":
                key_col = "original_work_name"

            df = df.with_columns(pl.col(key_col).map_dict(d).alias(te_col))
        return df

    def create_agg_expr(self, target_col):
        agg_expr = {
            "mean": pl.col(target_col).mean(),
            "sum": pl.col(target_col).sum(),
            "count": pl.col(target_col).count(),
            "std": pl.col(target_col).std(),
            "min": pl.col(target_col).min(),
            "max": pl.col(target_col).max(),
        }
        return agg_expr
