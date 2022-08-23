
import logging
from logging.config import fileConfig
import os
from pathlib import Path

import utils.readfile as rf
import utils.writefile as wf

MAINDIR = Path(__file__).parent

class myLogger: 
    def __init__(self):
        filepath = MAINDIR / "config" / "config.json"
        fileConfig(filepath, disable_existing_loggers=False)
        self.logger= logging.getLogger()