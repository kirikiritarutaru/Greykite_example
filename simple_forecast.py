import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig, MetadataParam)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import \
    summarize_grid_search_results

warnings.filterwarnings("ignore")

sf_dir = Path('simple_forecast_plots')
sf_dir.mkdir(parents=True, exist_ok=True)


def simple_forecast():
    # サンプルデータの用意
    dl = DataLoader()
    df = dl.load_peyton_manning()

    # データの情報を指定
    # freq設定; 時間: H, 日: D, 週: W
    metadata = MetadataParam(time_col="ts", value_col="y", freq="D")

    # 予測器
    forecaster = Forecaster()

    # 予測結果の保存先
    result = forecaster.run_forecast_config(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=365,  # 365 ステップ
            coverage=0.95,         # 予測
            metadata_param=metadata
        )
    )

    # 時系列データのプロット
    ts = result.timeseries
    fig = ts.plot()
    fig.write_html(str(sf_dir / 'forecast_plot.html'))

    # グリッドサーチ
    grid_search = result.grid_search
    cv_results = summarize_grid_search_results(
        grid_search=grid_search, decimals=2, cv_report_metrics=None,
        column_order=[
            "rank", "mean_test", "split_test", "mean_train", "split_train",
            "mean_fit_time", "mean_score_time", "params"
        ]
    )
    cv_results["params"] = cv_results["params"].astype(str)
    cv_results.set_index("params", drop=True, inplace=True)
    cv_results.transpose()
    cv_results.to_csv(str(sf_dir / 'cv_results.csv'))

    # 各要素(トレンド、周期性、イベント効果)に分解して可視化
    frc = result.forecast
    fig = frc.plot_components()
    fig.write_html(str(sf_dir / 'components.html'))

    # backtest
    backtest = result.backtest
    fig = backtest.plot()
    fig.write_html(str(sf_dir / 'backtest_plot.html'))

    backtest_eval = defaultdict(list)
    for metric, value in backtest.train_evaluation.items():
        backtest_eval[metric].append(value)
        backtest_eval[metric].append(backtest.test_evaluation[metric])
    metrics = pd.DataFrame(backtest_eval, index=["train", "test"]).T
    metrics.to_csv(str(sf_dir / 'metrics.csv'))

    summary = result.model[-1].summary()
    print(summary)

    model = result.model
    print('model:')
    print(model)


if __name__ == '__main__':
    simple_forecast()
