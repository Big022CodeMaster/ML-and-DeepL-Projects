
from hr import HR

def main():

    myHR = HR()

    # 1. generate data
    myHR.generate_data()

    # 2. score data
    myHR.score_data_in_X()

    # 3. calculate decision
    myHR.calculate_decision()

    # 4. split datasets
    myHR.split_datasets()

    # 5. train model
    myHR.train_model()

    # 6. predict test values
    myHR.predict_test_values()

    # 7. save all datasets
    myHR.save_datasets()

    #8 print and save confusion matrix
    myHR.print_confusion_matrix()

    # 9. print coefficients
    myHR.print_coef_()


if __name__ =='__main__':
    main()
