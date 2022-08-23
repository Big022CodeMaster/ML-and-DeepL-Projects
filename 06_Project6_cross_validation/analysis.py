from functools import partial
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
import json
import logging
import statistics
import os
from pathlib import Path

MAIN_DIR = Path(__file__).parent

class Analysis:
    def __init__(self):
        os.chdir(Path(__file__).parent)

        self.data = pd.DataFrame()
        self.model_list = []

        file_path_config = MAIN_DIR / "config.json"
        file_mp = MAIN_DIR / "model_param_dict.json"

        self.cfg = Analysis.read_in_dict(file_path_config)
        self.mp = Analysis.read_in_dict(file_mp)

        self.logger = self.create_logger()

    @staticmethod
    def read_in_dict(filename):
        with open(filename,"r") as file:
            output_ = file.read()
        return json.loads(output_)

    def create_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(self.cfg["log_format"])

        #1. file handler logger
        file_handler = logging.FileHandler(self.cfg["log_file_name"])
        file_handler.setFormatter(formatter)

        #2. stream logger
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def import_data(self):
        self.X = load_digits().data
        self.y = load_digits().target

    def cross_val_score_models(self):

        model_list= []

        for m in self.mp:
            for p in self.mp[m]:
                for v in self.mp[m][p]:
                    model = eval(m + "(" + str(p) + "=" + str(v) + ")")
                    score = cross_val_score(model,self.X,self.y, cv=self.cfg["n_folds"])
                    mean_score = round(statistics.mean(score),3)
                    info = {"model": model,"score": mean_score}
                    model_list.append(info)
                    self.model_list.append(info)
                    self.logger.info(info)

        self.logger.info("The model scoring is finished")

    def best_models(self):
        n = self.cfg["n_top_models"]
        self.model_list.sort(key=lambda d: d["score"], reverse = True) # d ist hier das Element, nach dem sortiert wird
        self.topmodels= self.model_list[0:n]

        self.output = f"The best {n} models are (in order): " + Analysis.dict_to_str(self.topmodels)

        self.logger.info(self.output)

    @staticmethod
    def dict_to_str(dict_):
        return " ".join([str(i) + " - " + str(item["model"]) + str(item["score"]) for (i,item) in enumerate(dict_,1)])

    def save_decision(self):
        file_path = MAIN_DIR / "decision.txt"

        with open(file_path,"w") as file:
            file.write(self.output)


