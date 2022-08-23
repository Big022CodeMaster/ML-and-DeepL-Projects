from analysis import Analysis
import os
from pathlib import Path


os.chdir(Path(__file__).parent)


def main():
    my_analysis = Analysis()

    # 1. import data
    my_analysis.import_data()

    # 2. split data into training and test data
    my_analysis.split_data_train_test()

    # 3. score models and store output in self.model_list
    my_analysis.score_models()


if __name__=="__main__":
    main()