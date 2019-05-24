import os
from collections import namedtuple

import pandas as pd
import pysftp

from inventoryforecast.common import DATE
from inventoryforecast.common import DateTimeFormatter
from inventoryforecast.common import ReportType
from inventoryforecast.common import TIMESTAMP
from inventoryforecast.common import USER_ID
from inventoryforecast.common import datetime
from inventoryforecast.common import logginghelper
from inventoryforecast.configuration import Configuration

FtpUploadJobMetrics = namedtuple('FtpUploadJobMetrics', ['runtime'])

LOG = logginghelper.get_logger(__name__)


# TODO Rewrite to f-string
def run(data_provider, hdfs_client, report_date, report_type, conf: Configuration) -> FtpUploadJobMetrics:
    start = pd.Timestamp.now(tz=conf.dc_timezone)
    if report_type == ReportType.AUTOCORRECTION_OPTIMIZATION:
        df = data_provider.read_sample(report_date, 1)
        optimization_report = data_provider.read_optimization_report(report_date, report_type, print_info=True)
        write = data_provider.write_autocorrection_sample_with_strata_size
        path = conf.csv_autocorrection_sample_with_strata_size_path
        new_file_name = '%s-%s-%s.tsv.gz' % (conf.adserver, 'autocorrection_sample',
                                             datetime.format_date(report_date,
                                                                  DateTimeFormatter.ISO_LOCAL_DATE))
    elif report_type == ReportType.FUTURE_SAMPLE_OPTIMIZATION:
        LOG.info('Start FTPUploadJob for Future Sample!')
        df = data_provider.read_future_sample(report_date, conf.forecast_period)
        optimization_report = data_provider.read_optimization_report(report_date, report_type, print_info=True)
        write = data_provider.write_future_sample_with_strata_size
        path = conf.csv_future_sample_with_strata_size_path
        new_file_name = '%s-%s-%s.tsv.gz' % (conf.adserver, 'future_sample',
                                             datetime.format_date(report_date + pd.Timedelta(days=1),
                                                                  DateTimeFormatter.ISO_LOCAL_DATE))
    else:
        raise Exception('Unsupported report type!')
    df_with_strata_size = df \
        .join(optimization_report, [USER_ID]) \
        .orderBy(DATE, TIMESTAMP)
    write(df_with_strata_size, report_date)
    file_names = [file
                  for file in hdfs_client.list('%s/%s'
                                               % (path, datetime.format_date(report_date, DateTimeFormatter.ISO_LOCAL_DATE)))
                  if '.csv.gz' in file]
    file_name = file_names[0]
    hdfs_client.copy_to_local_fs('%s/%s/%s' % (path, datetime.format_date(report_date, DateTimeFormatter.ISO_LOCAL_DATE),
                                               file_name), '/')
    os.rename('/%s' % file_name, '/%s' % new_file_name)
    for host in conf.ftp_address:
        LOG.info('Upload %s to %s:%s/%s' % (file_name, host, conf.ftp_directory, new_file_name))
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        with pysftp.Connection(host, username=conf.ftp_user, private_key=conf.ftp_private_key, cnopts=cnopts) as sftp:
            with sftp.cd(conf.ftp_directory):
                sftp.put('/%s' % new_file_name)
    return FtpUploadJobMetrics(runtime=(pd.Timestamp.now(conf.dc_timezone) - start).total_seconds())
