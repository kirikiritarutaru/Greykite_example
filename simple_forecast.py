import warnings
from collections import defaultdict

import pandas as pd
import plotly
from greykite.common.data_loader import DataLoader
from greykite.framework.templates.autogen.forecast_config import (
    ForecastConfig, MetadataParam)
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import \
    summarize_grid_search_results

warnings.filterwarnings("ignore")


def simple_forecast():
    # Loads dataset into pandas DataFrame
    dl = DataLoader()
    df = dl.load_peyton_manning()

    # specify dataset information
    metadata = MetadataParam(
        time_col="ts",  # name of the time column ("date" in example above)
        # name of the value column ("sessions" in example above)
        value_col="y",
        freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
                  # Any format accepted by `pandas.date_range`
    )

    # Creates forecasts and stores the result
    forecaster = Forecaster()
    # result is also stored as `forecaster.forecast_result`.
    result = forecaster.run_forecast_config(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=365,  # forecasts 365 steps ahead
            coverage=0.95,         # 95% prediction intervals
            metadata_param=metadata
        )
    )
    ts = result.timeseries
    fig = ts.plot()
    fig.write_html('forecast_plot.html')

    grid_search = result.grid_search
    cv_results = summarize_grid_search_results(
        grid_search=grid_search,
        decimals=2,
        # The below saves space in the printed output.
        # Remove to show all available metrics and columns.
        cv_report_metrics=None,
        column_order=[
            "rank", "mean_test", "split_test", "mean_train", "split_train",
            "mean_fit_time", "mean_score_time", "params"
        ]
    )
    # Transposes to save space in the printed output
    cv_results["params"] = cv_results["params"].astype(str)
    cv_results.set_index("params", drop=True, inplace=True)
    cv_results.transpose()


if __name__ == '__main__':
    simple_forecast()
