import os
import sys
import logging

log_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_directory="logs"
file_format=os.path.join(log_directory,"app.log")
os.makedirs(log_directory,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=log_str,
    handlers=[
        logging.FileHandler(file_format),
        logging.StreamHandler(sys.stdout)
    ]
)

logger=logging.getLogger("CarInsurancelogger")