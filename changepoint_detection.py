import warnings

import pandas as pd
import plotly
from greykite.algo.changepoint.adalasso.changepoint_detector import \
    ChangepointDetector
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum

warnings.filterwarnings("ignore")


def detect_trend_change_points():
    # データセットのロード
    dl = DataLoaderTS()
    ts = dl.load_peyton_manning_ts()
    df = ts.df

    # 時系列データをプロットして確認
    fig = ts.plot()
    fig.write_html('check_ts.html')

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
    fig.write_html('disp_changepoitns.html')

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
    fig.write_html('set_components(changepoints_detection).html')


if __name__ == '__main__':
    detect_trend_change_points()
