from pathlib import Path
import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from logging.config import fileConfig
import sys

FILEPATH = Path(__file__).parent
FILEPATH_DATASET_FOLDER = FILEPATH / "data"
FILEPATH_CONFIG_FOLDER = FILEPATH / "config"
FILEPATH_FIGURE_FOLDER = FILEPATH / "figure"

os.chdir(FILEPATH)

class HR:
    def __init__(self):
        config_dict = HR.read_dict('config.json')

        filepath = FILEPATH / config_dict["rel_ini_file_path"]

        fileConfig(filepath, disable_existing_loggers=False)
        self.logger= logging.getLogger()

        self.model = LogisticRegression(max_iter = 2000)

        self.cat_dict = HR.read_dict('category.json')
        self.dec_dict = HR.read_dict('decision.json')
        self.dataset_names = ['X_total','X','y','X_train','y_train','X_test','y_test','y_pred']

        self.test_size = config_dict["test_size"]
        self.num_of_records = config_dict["num_of_records"]

        self.base_cat_names = list()

        for dataset_name in self.dataset_names:
            setattr(self, dataset_name, pd.DataFrame())
        
        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"An HR instance is created")
       

    @staticmethod
    def read_dict(name):
        filepath = FILEPATH_CONFIG_FOLDER / name

        with open(filepath,mode= "r") as file:
            dict_ = json.load(file)
        
        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"A dict instance is created")

        return dict_

    def store_dataset(self, dataset_name, dataset):
        setattr(self, dataset_name, dataset)

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"A dataset is stored")

    def generate_data(self, seed = False):
        if seed == True:
            random.seed(4)

        for cat_ in self.cat_dict:
            for subcat_ in self.cat_dict[cat_]:
               self.X_total[subcat_]= random.choices(list(self.cat_dict[cat_][subcat_]['values']),k=self.num_of_records) 

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The data generation process is completed")

    def score_data_in_X(self):

        for cat_ in self.cat_dict:
            if cat_ == "data":
                for subcat_ in self.cat_dict[cat_]:
                        self.X[subcat_] = self.X_total[subcat_].map(self.cat_dict[cat_][subcat_]['values'])
            else:
                score = np.zeros(self.X_total.shape[0])
                
                for subcat_ in self.cat_dict[cat_]:
                    score += self.cat_dict[cat_][subcat_]['weight'] * self.X_total[subcat_].map(self.cat_dict[cat_][subcat_]['values'])
                    
                self.X[cat_] = score
            
        self.X['total score'] = self.X.sum(axis = 1)

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The data scoring process is completed")
 
    def calculate_decision(self):
        y_list = list() 

        for i, row in self.X.iterrows():
            for cat_ in self.dec_dict:
                if self.dec_dict[cat_]["low"] <= row["total score"] <= self.dec_dict[cat_]["high"]:
                    y_list.append(cat_)
                    break

        self.y["decision"] = y_list

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The decision calculations are completed")

    def split_datasets(self):
        self.X_train, self.X_test, self.y_train['y_train'], self.y_test['y_test'] = train_test_split(self.X, self.y, test_size= self.test_size)

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The dataset has been split")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train['y_train'])

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The model has been trained")

    def predict_test_values(self):
        self.y_pred = pd.DataFrame({'y_pred': self.model.predict(self.X_test)})

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"Predicted y values have been calculated")

    def save_single_dataset(self, folder: Path, name: str, ext: str):
        path = folder / '.'.join([name,ext])
        getattr(self,name).to_csv(path, index = False)

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")

    def save_datasets(self):
        if not os.path.exists(FILEPATH_DATASET_FOLDER):
            os.makedirs(FILEPATH_DATASET_FOLDER)

        for name in self.dataset_names:
            self.save_single_dataset(FILEPATH_DATASET_FOLDER,name,'csv')

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"All datasets have been saved")

    def print_confusion_matrix(self):
        if not os.path.exists(FILEPATH_FIGURE_FOLDER):
            os.makedirs( FILEPATH_FIGURE_FOLDER)
        FILEPATH_FIGURE = FILEPATH_FIGURE_FOLDER / 'cm.jpg'

        categories_ticks = [el.strip().replace(" ", "\n") for el in self.dec_dict]
        categories = list(self.dec_dict) 

        cm = confusion_matrix(self.y_test['y_test'],self.y_pred['y_pred'],labels=categories)
        cm_df = pd.DataFrame(cm, index = [i for i in categories],
                  columns = [i for i in categories])
        plt.figure(figsize = (10,10))
        sns.heatmap(cm_df, annot = True, xticklabels = categories_ticks, yticklabels = categories_ticks)

        plt.ylabel("True")
        plt.savefig(FILEPATH_FIGURE,dpi = 300)
        plt.show()

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The confusion matrix has been saved and printed")

    def print_coef_(self):
        print(f"{self.model.coef_=}")
        print(f"{self.model.coef_.shape=}")

        logging.debug(f"{sys._getframe().f_code.co_name}" + "is called")
        logging.info(f"The coefficients have been printed")