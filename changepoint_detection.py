import warnings
from pathlib import Path

import pandas as pd
from greykite.algo.changepoint.adalasso.changepoint_detector import \
    ChangepointDetector
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

warnings.filterwarnings("ignore")

cd_dir = Path('changepoint_detection_plots')
cd_dir.mkdir(parents=True, exist_ok=True)


def detect_trend_change_points():
    # データセットのロード
    dl = DataLoaderTS()
    ts = dl.load_peyton_manning_ts()
    df = ts.df

    # 時系列データをプロットして確認
    fig = ts.plot()
    fig.write_html(str(cd_dir / 'check_ts.html'))

    # モデル作成
    model = ChangepointDetector()
    res = model.find_trend_changepoints(df=df, time_col="ts", value_col="y")

    # 1. mean aggregation (小さな変動・季節性の影響を排除)
    # 2. 時系列全体に変化点の候補を配置
    # 3. 適応的LASSOを使って、重要でない変化点の係数をゼロに
    # [N.B.] 年間の季節性の影響は長すぎてmean aggregationで排除できない→trendにfitさせることを推奨
    # [N.B.] 予測タスクでは、期間のおしりの方はデータが少なく
    #  trendにfitさせられないことが多い、期間の最後から一定時間は変化点を配置しないことを推奨
    # 4. post-filter を使って時間的に近すぎる変化点を除外

    # 推定した変化点を表示
    print(pd.DataFrame({"trend_changepoints": res["trend_changepoints"]}))
    fig = model.plot(plot=False)
    fig.write_html(str(cd_dir / 'disp_changepoitns.html'))

    # モデルのプロットするコンポーネントを設定
    fig = model.plot(
        # whether to plot the observations
        observation=True,
        # whether to plot the unaggregated values
        observation_original=True,
        # whether to plot the trend estimation
        trend_estimate=True,
        # whether to plot detected trend changepoints
        trend_change=True,
        # whether to plot estimated yearly seasonality
        yearly_seasonality_estimate=True,
        # whether to plot the adaptive lasso estimated trend
        adaptive_lasso_estimate=True,
        # detected seasonality change points, discussed in next section
        seasonality_change=False,
        # plot seasonality by component (daily, weekly, etc.),
        # discussed in next section
        seasonality_change_by_component=True,
        # plot estimated trend+seasonality, discussed in next section
        seasonality_estimate=False,
        # set to True to display the plot (need to import plotly
        # interactive tool) or False to return the figure object
        plot=False
    )
    fig.write_html(str(cd_dir / 'set_components_CD.html'))

    # specify dataset information
    metadata = dict(
        # name of the time column ("datepartition" in example above)
        time_col="ts",
        # name of the value column ("macrosessions" in example above)
        value_col="y",
        freq="D"        # "H" for hourly, "D" for daily, "W" for weekly, etc.
        # Any format accepted by ``pd.date_range``
    )
    # specify changepoint parameters in model_components
    model_components = dict(
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "yearly_seasonality_order": 15,
                "regularization_strength": 0.5,
                "resample_freq": "7D",
                "potential_changepoint_n": 25,
                "no_changepoint_proportion_from_end": 0.2
            },
            "seasonality_changepoints_dict": {
                "potential_changepoint_distance": "60D",
                "regularization_strength": 0.5,
                "no_changepoint_proportion_from_end": 0.2
            }
        },
        custom={
            "fit_algorithm_dict": {"fit_algorithm": "ridge"}
        }
    )

    # Generates model config
    config = ForecastConfig.from_dict(
        dict(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=365,  # forecast 1 year
            coverage=0.95,  # 95% prediction intervals
            metadata_param=metadata,
            model_components_param=model_components))

    # Then run with changepoint parameters
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(df=df, config=config)
    fig = result.model[-1].plot_trend_changepoint_detection(dict(plot=False))
    fig.write_html(str(cd_dir / 'plot_trend_changepoint_detection.html'))

    backtest = result.backtest
    fig = backtest.plot()
    fig.write_html(str(cd_dir / 'backtest_CD.html'))

    forecast = result.forecast
    fig = forecast.plot()
    fig.write_html(str(cd_dir / 'forecast_CD.html'))

    fig = backtest.plot_components()
    fig.write_html(str(cd_dir / 'component_CD.html'))


if __name__ == '__main__':
    detect_trend_change_points()
