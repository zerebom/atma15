import lightgbm as lgb


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
    "verbosity": -1,  # 学習の状況を表示しない
}
