import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import json
import logging

class Analysis:
    def __init__(self):

        self.data = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.model_list = []

        cfg = Analysis.read_in_config()
    
        self.test_size = cfg["test_size"]
        self.kernel_options = cfg["kernel_options"]
        self.C_options = cfg["C_options"]
        
        self.log_file_name = cfg["log_file_name"]
        self.log_format = cfg["log_format"]

        self.logger = self.create_logger()

    @staticmethod
    def read_in_config():
        with open("./config.json","r") as file:
            output_ = file.read()
        return json.loads(output_)

    def create_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(self.log_format)

        #1. file handler logger
        file_handler = logging.FileHandler(self.log_file_name)
        file_handler.setFormatter(formatter)

        #2. stream logger
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def import_data(self):
        self.data = load_digits()

    def split_data_train_test(self):
        (self.X_train, self.X_test, 
        self.y_train, self.y_test) = train_test_split(self.data.data,self.data.target,test_size=self.test_size)

    def score_models(self):

        for k in self.kernel_options:
            for c in self.C_options:
                my_svc = SVC(kernel = k, C=c)
                my_svc.fit(self.X_train,self.y_train)

                score = my_svc.score(self.X_test,self.y_test)

                info = {"Model": my_svc,
                                "Score": round(score,3)
                                }

                self.model_list.append(info)
                self.logger.info(info)

        self.logger.info("The model scoring is finished")

