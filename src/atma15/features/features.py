import polars as pl
from sklearn.model_selection import BaseCrossValidator


class Feature:
    def fit(self, df):
        pass

    def transform(self, df):
        pass


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
        target_col: str,
        agg_dict: dict[tuple[str, str], list[str]],
        fold_instance: BaseCrossValidator,
    ):
        self.agg_dict = agg_dict
        self.agg_expr = {
            "mean": pl.col(target_col).mean(),
            "sum": pl.col(target_col).sum(),
            "count": pl.col(target_col).count(),
            "std": pl.col(target_col).std(),
            "min": pl.col(target_col).min(),
            "max": pl.col(target_col).max(),
        }
        self.fold_instance = fold_instance

    def fit(self, df: pl.DataFrame) -> pl.DataFrame:
        X = df
        y_bins = df["score"].to_numpy()

        valid_list = []
        for fold_n, (train_idx, val_idx) in enumerate(
            self.fold_instance.split(X, y_bins)
        ):
            print(f"Fold {fold_n + 1} started.")

            X_train, X_val = X.filter(pl.col("row_nr").is_in(train_idx)), X.filter(
                pl.col("row_nr").is_in(val_idx)
            )

            for (key_col, target_col), agg_func_list in self.agg_dict.items():
                for agg_func_str in agg_func_list:
                    te_col = f"te_{key_col}_{agg_func_str}_{target_col}"
                    print(te_col)

                    d = (
                        X_train.groupby([key_col])
                        .agg(self.agg_expr[agg_func_str])
                        .to_pandas()
                        .set_index(key_col)
                        .to_dict()[target_col]
                    )

                    X_val = X_val.with_columns(
                        pl.col(key_col).map_dict(d).alias(te_col)
                    )
                    print(len(X_val.columns))
            valid_list.append(X_val)
        out_df = pl.concat(valid_list)
        self.out_df = out_df
        return out_df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for key_col, _ in self.agg_dict.keys():
            agg_cols = [
                col for col in self.out_df.columns if col.startswith(f"te_{key_col}")
            ]
            agg_df = self.out_df.groupby(key_col).agg(pl.col(agg_cols).mean())
            df = df.join(agg_df, how="left", on=key_col)

        return df
