"""
Create "corpus_dir" and "models_dir" directories after project installation as configured in the config.ini file.
"""
import configparser
from pathlib import Path


def dir_setup():
    config = configparser.ConfigParser()
    config.read('config.ini')
    defaults = config['DEFAULT']

    corpus_dir = Path(defaults['corpus_dir'])
    models_dir = Path(defaults['models_dir'])

    if not corpus_dir.is_dir():
        corpus_dir.mkdir(parents=True)
    if not models_dir.is_dir():
        models_dir.mkdir(parents=True)


if __name__ == "__main__":
    dir_setup()
