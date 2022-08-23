
from titanic import Titanic
import pickle


def main():
    titanic = Titanic()

    # 1 import dataset
    titanic.import_dataset()

    # 2 prepare dataset
    titanic.prepare_dataset()

    # 3 split dataset into X and y
    titanic.split_X_y()

    # 4 encode categories
    titanic.encode_categories()

    # 5 adjust datatypes for X and y
    titanic.correct_datatypes_X_y()

    # 6 split
    titanic.split_train_test()

    # fit model
    titanic.fit_model()

    #score the model
    titanic.save_model_score()

    # save the model
    titanic.save_models()

    # with open("./predict_survival.pkl", mode ="br") as file:
    #             model = pickle.load(file)
    
    print(titanic.datasets["y_train"].to_numpy().ravel().shape)

    # print(model)
            


if __name__=="__main__":
    main()