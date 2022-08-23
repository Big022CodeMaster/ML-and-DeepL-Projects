from analysis import Analysis

def main():
    my_analysis = Analysis()

    # 1. import data
    my_analysis.import_data()

    # 2. prepare and encode data if necessary
    my_analysis.prepare_dataset()

    # 2. split data into X and y
    my_analysis.split_data_X_y()

    # 3. train test split
    my_analysis.my_train_test_split()

    # 4. fit models 
    my_analysis.fit_models()

    # 5. save the models
    my_analysis.save_models()



if __name__=="__main__":
    main()
