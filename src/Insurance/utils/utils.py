import os
import sys
import  yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from logger import logger
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """Read the Yaml file and returns the dict

    Args:
        path -> str path like string

    Returns:
        ConfigBox: ConfigBox type  

    """
    try:
        with open(path,"r") as yaml_file:
            logger.info("Reading the yaml file from the path {0}".format(path))
            content=yaml.safe_load(yaml_file)
            logger.info("Yaml file {0} read successfully".format(yaml_file))
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        logger.exception(e)
        raise e

@ensure_annotations
def create_directories(path_list: list, verbose=True):
    """
    Args: 
        path_list : List of directories to be created

    Returns:
        None
    """

    try:
        for dir in path_list:
            logger.info("Creating the directory {0}".format(dir))
            os.makedirs(dir,exist_ok=True)
            logger.info("Directory {0} created Successfully.".format(dir))
    except Exception as e:
        logger.exception(e)
        raise e