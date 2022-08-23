import os
from pathlib import Path
import json
import pandas as pd
from sklearn import linear_model
import pickle

FILEPATH = Path(__file__).parent
FILEPATH_CONFIG = FILEPATH / "config" / "config.json"
FILEPATH_DATASET_FOLDER = FILEPATH / "data"

os.chdir(FILEPATH)

class NetPCI:

    def __init__(self):
        config_dict = NetPCI.import_json()
        
        self.country_code = config_dict['country_code']
        self.input_file_name = FILEPATH / config_dict['input_file_name']
        self.output_file_name = FILEPATH / config_dict['output_file_name']
        self.to_be_predicted = pd.DataFrame({"year": 
                                            range(config_dict['estimated_start_year'],
                                                  config_dict['estimated_stop_year']+1)}
                                            )
        self.start_training = config_dict['start_training']
        self.end_training = config_dict['end_training']

    @staticmethod    
    def import_json():
        with open(FILEPATH_CONFIG, mode = "r") as file:
            result = json.load(file)
        return result

    def import_data(self):
        df = pd.read_csv(self.input_file_name, skiprows = lambda x: x in range(4))
        df = df[df['Country Code']==self.country_code]
        df = df.loc[:,str(self.start_training):str(self.end_training)].T
        df.columns = ['netPCI']
        df['year']= [int(i) for i in df.index.values]
        df.reset_index(drop = True, inplace = True)
        self.df = df

    def split_data(self):
        self.input_df = self.df.drop("netPCI",axis = "columns")
        self.netPCI_df = self.df.drop("year",axis = "columns")
        
    def regression_fit(self):
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.input_df,self.netPCI_df)

    def predict_values(self):
        self.to_be_predicted['netPCI'] = self.reg.predict(self.to_be_predicted).round(2)

    def print_coefficients(self):
        print("Extract Cofficients for  y=mx+b")
        print(f"m = {self.reg.coef_[0][0]:.2f}")    
        print(f"b = {self.reg.intercept_[0]:.2f}")

    def print_scoring(self):
        print(f"Scoring: {self.reg.score(self.X_test,self.to_be_predicted['netPCI'])}")

    def save_results(self):
        self.to_be_predicted.to_csv(self.output_file_name)


    def save_model_as_pickle(self):
        if not os.path.exists(FILEPATH_DATASET_FOLDER):
            os.makedirs(FILEPATH_DATASET_FOLDER)
        with open(FILEPATH_DATASET_FOLDER/"regr_model1.pkl", model = "wb") as file:
            pickle.dump(self.reg,file)

    def save_model_as_joblib(self):
        pass

    #TODO: save the model for the future
    #1. pickle
    #2.joblib

