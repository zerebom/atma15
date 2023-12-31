import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_importance(models, feat_train_df,n_top=50):
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
        .index[:n_top]
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
