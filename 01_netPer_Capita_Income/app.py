
from netPCI import NetPCI

# bei time series nicht NA droppen
#TODO Spalten droppen => wo Na (any)

def main():
    my_netPCI = NetPCI()
    my_netPCI.import_data()
    my_netPCI.split_data()
    my_netPCI.regression_fit()
    my_netPCI.predict_values()
    my_netPCI.save_results()
    my_netPCI.print_coefficients()
    my_netPCI.print_scoring()
    my_netPCI.save_model_as_pickle()


if __name__ == '__main__':
    main()