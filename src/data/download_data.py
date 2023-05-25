import tarfile
import logging
import urllib.request
from pathlib import Path


def main(data_filepath):
    tarball_path = data_filepath.joinpath('data/external/housing.tgz')
    raw_path = data_filepath.joinpath('data/raw/')

    if tarball_path.exists():
        logging.warning( FileExistsError("Data already downloaded!") )
    elif raw_path.exists():
        logging.warning( FileExistsError("Data already extracted!") )

    if not tarball_path.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path=raw_path)

    logger = logging.getLogger(__file__)
    logger.info('downloading and extracting the raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path('download_data.py').resolve().parents[0]

    main(project_dir)