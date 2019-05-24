from setuptools import find_packages
from setuptools import setup

from inventoryforecast import __version__

setup(
    name='inventoryforecast',
    version=__version__,
    description='Inventale Inventory Forecast Spark Application to run on DCOS',
    author='Inventale',
    packages=find_packages(),
    install_requires=[
        'pystan',
        'numpy',
        'pandas',
        'scipy',
        'pyspark',
        'fbprophet',
        'statsmodels', 'pytest', 'hdfs', 'pytz'
    ],
    include_package_data=True
)
