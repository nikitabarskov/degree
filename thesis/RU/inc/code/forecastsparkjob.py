import functools
from collections import namedtuple

import pandas as pd

from inventoryforecast.common import *
from inventoryforecast.common import logginghelper
from inventoryforecast.configuration import Configuration
from inventoryforecast.jobs import *

ForecastSparkJobMetrics = namedtuple(
    typename='ForecastSparkJobMetrics',
    field_names=['runtime', 'forecast_impressions_sum'])

LOG = logginghelper.get_logger(__name__)


def run(data_provider, ad_unit_ids_from_sample, conf: Configuration,
        forecast_model, report_date: pd.Timestamp):
    def make_prediction(ad_unit_ts, metric):
        LOG.info(f'Produce forecast for {metric}')
        ad_unit_ts = ad_unit_ts.rename(index=str,
                                       columns={DATE: DS, metric: Y})
        forecast_ts = forecast_model.predict(ad_unit_ts)
        forecast_ts = forecast_ts.rename(index=str,
                                         columns={DS: DATE, VALUE: metric})
        return forecast_ts

    start = pd.Timestamp.now(tz=conf.dc_timezone)
    statistic = data_provider.read_page_report(
        start_date=conf.reports_source_start_date,
        end_date=report_date)
    LOG.info(f'Amount of adUnitIdsFromSample {len(ad_unit_ids_from_sample)}')
    statistic = statistic[statistic.adUnitId.isin(ad_unit_ids_from_sample)]
    statistic = _fill_missing_values_in_time_series(
        statistic,
        conf.reports_source_start_date,
        report_date,
        conf.metrics)
    ad_unit_ids_amount = statistic.adUnitId.nunique()
    LOG.info(f'Produce time series forecast for {ad_unit_ids_amount} '
             f'adUnitIds')
    current_ad_unit_id_num = 1

    forecast_df = pd.DataFrame(columns=([DATE, AD_UNIT_ID] + conf.metrics))
    for ad_unit_id, ts in statistic.groupby(AD_UNIT_ID, as_index=False):
        LOG.info(f'Produce time series forecast for {ad_unit_id} '
                 f'adUnitId ({current_ad_unit_id_num}/{ad_unit_ids_amount})')
        forecast = functools.reduce(
            lambda x, y: x.merge(y, on=[DATE]),
            [make_prediction(ts, metric) for metric in conf.metrics])
        forecast[AD_UNIT_ID] = ad_unit_id
        forecast_df = forecast_df.append(forecast)
        current_ad_unit_id_num += 1
    forecast_df.date = pd.to_datetime(forecast_df.date).dt.date
    forecast_df.adUnitId = forecast_df \
        .adUnitId \
        .astype('long')
    for metric in conf.metrics:
        forecast_df[metric] = forecast_df[metric].astype('long')
    imps = forecast_df.impressions.sum()
    data_provider.write_forecast_report(forecast_df, report_date)
    return ForecastSparkJobMetrics(
        runtime=(pd.Timestamp.now(tz=conf.dc_timezone) - start).total_seconds(),
        forecast_impressions_sum=imps), forecast_df


def _fill_missing_values_in_time_series(time_series, start_date, end_date,
                                        metrics):
    columns = [DATE, AD_UNIT_ID] + metrics
    tss = pd.DataFrame(columns=columns)
    dates = pd.date_range(start=start_date,
                          end=end_date)
    for dimension_id, ts in time_series.groupby(by=[AD_UNIT_ID],
                                                as_index=False):
        if len(ts) > len(dates):
            raise Exception(
                f'Time series is broken! '
                f'It contains {len(ts)} days, but should {len(dates)}')
        if len(ts) < len(dates):
            LOG.info(
                f'For adUnitId={dimension_id} contains {len(ts)} days of '
                f'statistic, refill it with default value {DEFAULT_VALUE}')
            ts = ts \
                .drop(columns=[AD_UNIT_ID]) \
                .set_index(keys=[DATE])
            ts = ts \
                .reindex(index=pd.DatetimeIndex(data=dates, name=DATE),
                         fill_value=DEFAULT_VALUE)
            ts[AD_UNIT_ID] = dimension_id
        ts.reset_index(inplace=True)
        tss = tss.append(ts[columns])
    for metric in metrics:
        tss[metric] = tss[metric].astype('long')
    return tss
