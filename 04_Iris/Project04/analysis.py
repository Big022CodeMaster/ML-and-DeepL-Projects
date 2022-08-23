from tkinter.tix import MAIN
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from pathlib import Path
import json
import pickle
import logging
from logging.config import fileConfig
from mailclient import Mailclient

MAINFOLDER = Path(__file__).parent
os.chdir(MAINFOLDER)

class Analysis:
    def __init__(self):
        self.cfg = Analysis.read_dict('config.json')

        fileConfig(self.cfg["logging_cfg_filename"], disable_existing_loggers=False)
        self.logger = logging.getLogger()

        self.df = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.model_list = []
        self.enc = []

    def import_data(self):
        filepath = MAINFOLDER / self.cfg["data_set_name"]
        self.df = pd.read_csv(filepath)

        self.logger.info("data has been imported")

    @staticmethod
    def read_dict(filepath):
        filepath = MAINFOLDER / filepath
        with open(filepath,"r") as file:
            dict_ = json.loads(file.read())

        return dict_

    @staticmethod
    def save_object(object,filepath):
        filepath = MAINFOLDER / filepath
        with open(filepath, mode = "wb") as file:
            pickle.dump(object,file)

    def prepare_dataset(self):
        
        for c in self.df:
            if self.df[c].dtype=='object':
                le = LabelEncoder()
                self.df[c] = le.fit_transform(self.df[c])
                dict_ = {i: le.transform([i]) for i in le.classes_}
                self.enc.append(dict_)

        self.logger.info("data has been prepared")

    def split_data_X_y(self):
        self.X = self.df.drop(self.cfg["X_sel"], axis = "columns")
        self.y = self.df[self.cfg["y_sel"]]
        self.logger.info("data has been split into X and y")

    def my_train_test_split(self):
        (self.X_train, self.X_test, 
        self.y_train, self.y_test) = train_test_split(self.X,self.y,test_size= self.cfg["test_size"])
        
        self.logger.info("data has been split into X_train, x_train, y_test, y_train")

    def fit_models(self):
        for n in self.cfg["est"]:
            model = RandomForestClassifier(n_estimators=n)
            model.fit(self.X_train,self.y_train)
            score_= model.score(self.X_test,self.y_test)
            self.model_list.append({"model": model,
                                "score": round(score_,4)})

        self.logger.info("the models have been fitted")
        
    
    def save_models(self):
        filepath = MAINFOLDER / self.cfg["model_filename"]
        Analysis.save_object(self.model_list,filepath)

        self.logger.info("the models have been saved")
        # mail = Mailclient()
        # mail.send("The analysis has been finished", "This body", attachment_name = self.cfg["model_filename"])




    

