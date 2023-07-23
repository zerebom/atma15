
pair_df = pl.concat(
    [raw_train[["user_id", "anime_id"]], raw_test[["user_id", "anime_id"]]]
)

anime_counts = pair_df["anime_id"].value_counts()

# 同一ユーザーの視聴アニメペアを作成し、各アニメの出現数を結合
pair2pair_df = (
    pair_df.join(pair_df, on="user_id")  # 同一ユーザーの視聴アニメペア作成
    .join(
        anime_counts, left_on="anime_id", right_on="anime_id", how="left"
    )  # 左側のアニメの出現数を結合
    .join(
        anime_counts, left_on="anime_id_right", right_on="anime_id", how="left"
    )  # 右側のアニメの出現数を結合
)

# 各アニメペアの類似度を計算
sim_anime_df = (
    pair2pair_df.with_columns(
        # 二つのアニメの出現数の平方を算出し、それを基に重み付け
        (pl.col("counts") * pl.col("counts_right"))
        .sqrt()
        .alias("item_count_coef")
        .cast(pl.Float32)
    )
    .with_columns((1 / pl.col("item_count_coef")).alias("weight"))  # 逆数で重み付け
    .groupby(["anime_id", "anime_id_right"])  # アニメペアごとにグループ化
    .agg(pl.sum("weight"))  # 重み付けの和を求める
)
