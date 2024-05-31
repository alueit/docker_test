import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

target_col = 'binary_target'
feature_select = ['сумма', 'доход', 'частота_пополнения',
                  'сегмент_arpu', 'частота', 'объем_данных',
                  'on_net', 'продукт_1', 'продукт_2',
                  'секретный_скор',]


def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file)[feature_select]

    return input_df

# Main preprocessing function
def run_preproc(input_df):


    # Create dataframe 
    output_df = input_df.copy()
    output_df['income_freq_mul'] = output_df['доход'] * output_df['частота_пополнения']
    output_df['sum_repl_freq_mul'] = output_df['сумма'] * output_df['частота_пополнения']

    # Return resulting dataset values
    return output_df.values