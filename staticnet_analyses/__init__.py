import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(filename)-20s%(lineno)4d:\t %(message)s',
                              datefmt='%d-%m-%Y:%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
