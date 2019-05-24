from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sps
from numpy import random
from scipy.optimize._lsq.trf_linear import trf_linear

from inventoryforecast.common import *
from inventoryforecast.common import logginghelper
from inventoryforecast.common.reporttype import ReportType
from inventoryforecast.configuration import Configuration

OptimizerSparkJobMetrics = namedtuple(
    'OptimizerSparkJobMetrics',
    ['runtime', 'matrix_size', 'iterations', 'cost'])

LOG = logginghelper.get_logger(__name__)


def run(spark, data_provider, report_type, conf: Configuration, report_date,
        **kwargs) -> OptimizerSparkJobMetrics:
    start = pd.Timestamp.now()
    if report_type is ReportType.FUTURE_SAMPLE_OPTIMIZATION:
        matrix_report = kwargs.get('future_sample_report')
        LOG.info(f'\n{matrix_report.head()}')
        right_side_report = kwargs.get('forecast_report')
        LOG.info(f'\n{right_side_report.head()}')
        max_iterations = conf.future_sample_optimization_max_iterations
    elif report_type is ReportType.AUTOCORRECTION_OPTIMIZATION:
        matrix_report = data_provider \
            .read_sample_report_as_pandas(date=report_date)
        right_side_report = data_provider \
            .read_page_report(date=report_date)
        max_iterations = conf.autocorrection_sample_optimization_max_iterations
    else:
        raise Exception('Unexpected optimization type!')
    LOG.info(f'Matrix report\n{matrix_report.head()}')
    LOG.info(f'Right Side report\n{right_side_report.head()}')
    initial_values = data_provider.get_initial_values(
        matrix_report,
        conf,
        report_type,
        report_date)
    matrix, right_side, x_lsq, lb, ub, user_uniques = _build_equation_system(
        matrix_report,
        right_side_report,
        conf.metrics,
        conf.left_border,
        conf.right_border,
        initial_values)
    LOG.info(f'Matrix size is {matrix.shape[0]} x {matrix.shape[1]}')
    LOG.info(f'Right side size is '
             f'{right_side.shape[0]} x {right_side.shape[1]}')
    optimization_result = trf_linear(A=matrix,
                                     b=np.reshape(right_side.toarray(),
                                                  matrix.shape[0]),
                                     x_lsq=x_lsq,
                                     lb=lb,
                                     ub=ub,
                                     tol=conf.optimization_tolerance,
                                     lsq_solver='lsmr',
                                     lsmr_tol='auto',
                                     max_iter=max_iterations,
                                     verbose=2)
    LOG.info('Successfully solve the matrix')
    results = pd.DataFrame(
        list(zip(optimization_result.x, user_uniques)),
        columns=[STRATA_SIZE, USER_ID])
    results.strataSize = (results.strataSize + random.random()).astype('long')
    results_sdf = spark.createDataFrame(results)
    data_provider.write_optimization_report(results_sdf, report_date,
                                            report_type)
    return OptimizerSparkJobMetrics(
        runtime=(pd.Timestamp.now() - start).total_seconds(),
        matrix_size=matrix.shape,
        cost=optimization_result.cost,
        iterations=optimization_result.nit)


def _build_equation_system(matrix_report, right_side_report, metrics,
                           left_border, right_border, initial_values):
    user_uniques = matrix_report.userId.unique()
    matrix_report['code'] = matrix_report \
                                .adUnitId \
                                .astype('str') \
                            + matrix_report \
                                .date \
                                .astype('str')
    right_side_report['code'] = right_side_report \
                                    .adUnitId \
                                    .astype('str') \
                                + right_side_report \
                                    .date \
                                    .astype('str')
    right_side_report = right_side_report[
        right_side_report
            .code
            .isin(matrix_report.code)]
    matrix_row_indices = matrix_report.code.unique()
    right_side_row_indices = right_side_report.code.unique()
    LOG.info(f'Number of rows in matrix {len(matrix_row_indices)} '
             f'Number of cols in matrix {len(user_uniques)} '
             f'Number of rows in right side {len(right_side_row_indices)} ')
    matrix_rows = pd.Categorical(matrix_report.code,
                                 categories=matrix_row_indices)
    matrix_cols = pd.Categorical(matrix_report.userId, categories=user_uniques)
    matrix_shape = (len(matrix_row_indices), len(user_uniques))
    right_side_rows = pd.Categorical(right_side_report.code,
                                     categories=matrix_row_indices)
    right_side_cols = [0] * len(right_side_rows)
    right_side_shape = (len(right_side_row_indices), 1)

    matrices = []
    right_sides = []
    for metric in metrics:
        LOG.info(f'Create matrix for {metric}')
        matrices.append(
            sps.csr_matrix((matrix_report[metric],
                            (matrix_rows.codes, matrix_cols.codes)),
                           matrix_shape))
        right_sides.append(
            sps.csr_matrix((right_side_report[metric],
                            (right_side_rows.codes, right_side_cols)),
                           right_side_shape))
    return sps.vstack(matrices, format='csr'), \
           sps.vstack(right_sides, format='csr'), \
           np.resize(initial_values, len(user_uniques)), \
           np.resize(np.asarray([left_border], dtype='float32'),
                     len(user_uniques)), \
           np.resize(np.asarray([right_border], dtype='float32'),
                     len(user_uniques)), \
           user_uniques
