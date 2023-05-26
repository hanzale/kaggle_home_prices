import sys, os
import zipfile
import logging
import urllib.request

def main(project_dir):
    zipfile_path = os.path.join(project_dir, 'data/external/home-data-for-ml-course.zip')
    raw_path = os.path.join(project_dir, 'data/raw/')

    if os.path.exists(zipfile_path):
        logging.warning( FileExistsError("Data already downloaded!") )
        exit()
    elif os.path.exists(raw_path):
        logging.warning( FileExistsError("Data already extracted!") )
        exit()

    with zipfile.ZipFile(zipfile_path) as f:
        f.extractall(path=raw_path)

    logger = logging.getLogger(__file__)
    logger.info('downloading and extracting the raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #sys.path.append(os.getcwd())
    project_dir = os.getcwd()
    
    main(project_dir)