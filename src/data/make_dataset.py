# -*- coding: utf-8 -*-
import logging
import os, sys

from src.data.split_train_test import Split


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.getcwd()
    sys.path.append(project_dir)
    Split()

    main()

data_missing_columns = raw_data.columns[raw_data.isna().any()]
