import functools
import math
import random
from collections import defaultdict
from collections import namedtuple

import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import TimestampType

from inventoryforecast.common import *
from inventoryforecast.common import logginghelper
from inventoryforecast.common.datetime import (DateTimeFormatter,
                                               DateTimeDeltaFormatter, )
from inventoryforecast.configuration import Configuration

LOG = logginghelper.get_logger(__name__)

DATE_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE
DELTA_FORMMATER = DateTimeDeltaFormatter.DAYS

FutureSampleSparkJobMetrics = namedtuple('FutureSampleSparkJobMetrics',
                                         ['runtime', 'uniques_count'])

NEW_USER_ID = 'newUserId'


def run(spark, data_provider, conf: Configuration, report_date):
    start = pd.Timestamp.now(tz=conf.dc_timezone)
    future_sample_parts = math.ceil(conf.forecast_period / conf.sample_period)
    uniques_count_period = [period * conf.sample_period
                            for period in range(1, future_sample_parts + 1)]
    LOG.info(f'Future Sample will be produced for {future_sample_parts}')
    first_part_df = data_provider.read_sample(
        report_date - pd.Timedelta(days=conf.sample_period),
        conf.sample_period)
    second_part_df = data_provider.read_sample(report_date, conf.sample_period)
    first_period_dates_users = first_part_df \
        .groupBy(sf.col(DATE)) \
        .agg(sf.collect_set(sf.col(USER_ID))) \
        .rdd \
        .collectAsMap()
    second_period_dates_users = second_part_df \
        .groupBy(sf.col(DATE)) \
        .agg(sf.collect_set(sf.col(USER_ID))) \
        .rdd \
        .collectAsMap()
    first_half_users = functools.reduce((lambda x, y: x.union(y)),
                                        [set(users) for users in
                                         first_period_dates_users.values()])
    second_half_users = functools.reduce((lambda x, y: x.union(y)),
                                         [set(users) for users in
                                          second_period_dates_users.values()])
    retention_users = first_half_users.intersection(second_half_users)
    intersection = len(retention_users)
    not_retained_count = len(first_half_users) - intersection
    incoming_count = len(second_half_users) - intersection
    expected_count_of_uniques = (incoming_count + not_retained_count) / 2
    LOG.info(f'Not retained users count: {not_retained_count}')
    LOG.info(f'Incoming users count: {incoming_count}')
    LOG.info(f'Intersection: {intersection}')
    LOG.info(f'Expected count of uniques: {expected_count_of_uniques}')
    days_by_user = defaultdict(int)
    for users in list(first_period_dates_users.values()) + list(
            second_period_dates_users.values()):
        for user in users:
            days_by_user[user] += 1
    days_by_user = {user: days_count
                    for user, days_count in days_by_user.items()
                    if user in second_half_users}
    retentions = _init_retentions(conf.sample_period, days_by_user,
                                  expected_count_of_uniques)
    LOG.info(f'Retention distribution: {retentions}')
    LOG.info(f'Number of future sample parts is {future_sample_parts}')
    offsets = [pd.Timedelta(days=offset * conf.sample_period)
               for offset in range(1, future_sample_parts + 1)]
    new_users_by_offset = defaultdict(dict)
    random.seed(12345)
    for user, days_count in days_by_user.items():
        for i in range(len(offsets)):
            current_offset = offsets[i]
            if random.random() < retentions[days_count - 1]:
                if i == 0:
                    new_users_by_offset[current_offset][user] = user
                else:
                    prev_offset = offsets[i - 1]
                    new_users_by_offset[current_offset][user] = \
                        new_users_by_offset[prev_offset][user]
            else:
                new_users_by_offset[current_offset][user] = user + str(i)

    future_sample_reports = [
        _extrapolate_sample_events(
            spark,
            data_provider,
            second_part_df,
            offset,
            new_users_by_offset[offset],
            report_date)
        for offset in offsets]
    future_sample_report = functools.reduce(lambda x, y: x.union(y),
                                            [fsr
                                             for fsr in future_sample_reports])
    data_provider.write_future_sample_report(future_sample_report, report_date)

    LOG.info('Successfully produced future sample and future sample report '
             f'for report date '
             f'{datetime.format_date(report_date, DATE_FORMATTER)}')

    future_sample_report = future_sample_report.toPandas()

    future_sample_report.date = pd.to_datetime(future_sample_report.date)
    future_sample_report.adUnitId = future_sample_report \
        .adUnitId \
        .astype('long')
    future_sample_report.impressions = future_sample_report \
        .impressions \
        .astype('long')
    future_sample_report.uniques = future_sample_report \
        .uniques \
        .astype('long')

    uniques_count = dict()
    for period in uniques_count_period:
        date_filter = future_sample_report.date.isin(
            pd.date_range(start=report_date,
                          periods=period + 1,
                          closed='right'))
        uniques_count[str(period)] = future_sample_report[date_filter] \
            .userId \
            .nunique()

    LOG.info('Successfully count amount of uniques per forecast period '
             f'for report date '
             f'{datetime.format_date(report_date, DATE_FORMATTER)}')

    runtime = pd.Timestamp.now(tz=conf.dc_timezone) - start
    metrics = FutureSampleSparkJobMetrics(runtime=runtime.total_seconds(),
                                          uniques_count=uniques_count)
    return metrics, future_sample_report


def _init_retentions(sample_period, days_by_user, aim):
    distribution = []
    observed_period = sample_period * 2

    # Each group of users have some number (N) of visits for previous period.
    # We have (sample_period) rolls of dice 1d(observed_period).
    # If value of the thrown dice is higher than (N) for (sample_period) tries
    # user will not be retained
    # +7%, because a user is not a dice (statistics from sampling)

    for i in range(observed_period):
        distribution.insert(i, pow((observed_period - 1 - i) / observed_period,
                                   sample_period) + 0.07)
    user_distribution = defaultdict(int)
    for user, days_count in days_by_user.items():
        user_distribution[days_count] += 1
    mistake = aim - sum([distribution[days_count - 1] * users_count
                         for days_count, users_count in
                         user_distribution.items()])
    for i in range(observed_period):
        distribution[i] += mistake * distribution[i] / (aim - mistake)
        distribution[i] = 1 - distribution[i]
    return distribution


def _extrapolate_sample_events(spark, data_provider, sample, offset, new_users,
                               report_date):
    LOG.info(f'Produce Future Sample for '
             f'{datetime.format_timedelta(offset, DELTA_FORMMATER)} '
             f'offset')
    future_sample_df = sample \
        .withColumn(TIMESTAMP,
                    (sf.col(TIMESTAMP) + offset.total_seconds())
                    .cast(IntegerType())) \
        .withColumn(DATE,
                    sf.date_format(sf.col(TIMESTAMP).cast(TimestampType()),
                                   "yyyy-MM-dd")) \
        .withColumn(TIMESTAMP,
                    sf.col(TIMESTAMP).cast(StringType()))
    new_users_df = spark.createDataFrame([(k, v)
                                          for k, v in new_users.items()],
                                         [USER_ID, NEW_USER_ID])
    future_sample_df = future_sample_df \
        .join(new_users_df, [USER_ID]) \
        .drop(USER_ID) \
        .withColumnRenamed(NEW_USER_ID, USER_ID)

    data_provider.write_future_sample(future_sample_df, report_date)
    LOG.info(f'Produce Future Sample Report for '
             f'{datetime.format_timedelta(offset, DELTA_FORMMATER)} '
             f'offset')
    future_sample_report_df = future_sample_df \
        .groupBy(DATE, AD_UNIT_ID, USER_ID) \
        .agg(sf.count(INTERACTION).alias(IMPRESSIONS)) \
        .withColumn(UNIQUES, sf.lit(1))

    return future_sample_report_df
