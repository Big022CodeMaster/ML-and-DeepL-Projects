from analysis import Analysis


def main():
    my_analysis = Analysis()

    # 1. import data
    my_analysis.import_data()

    # 2. cross-validate and score models and store output in self.model_list
    my_analysis.cross_val_score_models()

    # 3. determine three best models
    my_analysis.best_models()

    #4. save decision in file
    my_analysis.save_decision()


if __name__=="__main__":
    main()