import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import json
import pickle
import sys
import logging
import logging.handlers


MAIN_DIR_PATH = Path(__file__).parent

os.chdir(MAIN_DIR_PATH)

class Titanic:
    def __init__(self):
        self.logger = logging.getLogger()
       
        self.configure_logger()

        self.datasets = dict()
        self.le_dict = dict()
        self.models = dict()

        logging.info("the class is initialized")

    def configure_logger(self):

        logger_dict = Titanic.read_dict('logger_config.json')

        level = "logging." + logger_dict["level"]
        self.logger.setLevel(eval(level))

        x = logger_dict["formatter"]
        formatter = logging.Formatter(logger_dict["formatter"])

        #1. file handler logger
       
        file_handler = logging.handlers.TimedRotatingFileHandler(logger_dict["file_name"], 
                                                                when = logger_dict["when"], 
                                                                interval = logger_dict["interval"], 
                                                                backupCount = logger_dict["backupCount"], 
                                                                encoding = logger_dict["encoding"])
        file_handler.setFormatter(formatter)

        #2. stream handler
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        logging.info("the logger is configured")


    @staticmethod
    def read_dict(name):
        filepath = MAIN_DIR_PATH / name

        with open(filepath, mode = "r") as file:
            output = json.load(file)

        return output

    @staticmethod
    def debug_function(text):
        print(f"called: {text}")

    def import_dataset(self,file_name = "titanic.csv"):
        self.datasets["X_total"] = pd.read_csv(MAIN_DIR_PATH / file_name)[["Sex", "Age","Pclass","Fare","Survived"]]
        logging.info('')

    def prepare_dataset(self):
        self.datasets["X_total"].dropna(how="any", axis=0, inplace=True)
        logging.info("dataset prepared")


    def split_X_y(self):
        self.datasets["X"] = self.datasets["X_total"].drop("Survived", axis = "columns")
        self.datasets["y"] = self.datasets["X_total"][["Survived"]]
        logging.info("datset split into X and y")

    def encode_categories(self):
        le_sex = LabelEncoder()
        self.datasets["X"]["Sex"] = le_sex.fit_transform(self.datasets["X"]["Sex"])
        self.le_dict["Sex"] = dict(zip(le_sex.classes_,le_sex.transform(le_sex.classes_)))
        logging.info("categories are encoded")

    def correct_datatypes_X_y(self):
        self.datasets["X"]["Sex"] = self.datasets["X"]["Sex"].astype("category")
        self.datasets["X"]["Fare"] = self.datasets["X"]["Fare"].astype("float16")
        self.datasets["X"]["Age"] = self.datasets["X"]["Age"].astype("float16")
        self.datasets["y"]["Survived"] = self.datasets["y"]["Survived"].astype("category")
        logging.info('datatypes are corrected')
        
    def split_train_test(self,test_size = .2):
        (self.datasets["X_train"], self.datasets["X_test"], 
        self.datasets["y_train"], self.datasets["y_test"]) = \
        train_test_split(self.datasets["X"],self.datasets["y"],test_size=test_size)
        logging.info('datatset is split into training and testing sets')

    def fit_model(self,max_iter=5000):
        self.models["predict_survival"] = LogisticRegression(max_iter=max_iter).fit(self.datasets["X_train"],self.datasets["y_train"].to_numpy().ravel())
        logging.info('the model is fitted')

    def save_model_score(self, file_name = "model_score.txt"):
        score_ = self.models["predict_survival"].score(self.datasets["X_test"],self.datasets["y_test"]).round(3)
        text_output = "Model Score: " + str(score_)
        logging.info('the model score is calculated and saved')       
        
        file_path = MAIN_DIR_PATH / file_name
        with open(file_path, mode ="w") as file:
            file.write(text_output)
        logging.info("the file is accessed for writing")

    def save_models(self, file_name = "predict_survival.pkl"):

        for model in self.models:

            file_name = model + ".pkl"
            file_path = MAIN_DIR_PATH / file_name

            with open(file_path, mode ="bw") as file:
                pickle.dump(self.models[model], file)
        
        logging.info("all models are saved")
