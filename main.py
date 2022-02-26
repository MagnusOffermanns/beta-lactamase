import time

from Classes import util, Data_functions
import argparse
from pandas import read_pickle, concat
from numpy import ceil
from alive_progress import alive_bar


def main(mode):
    if mode == 'download':
        Data_functions.download_and_unzip_files()

    elif mode == 'prepare_dataset':
        print('==converted the downloaded csv to pandas DataFrame==')
        dataset = Data_functions.convert_csv_to_Dataframe()

        print('===eliminate molecules without canoncial smiles value===')
        dataset = Data_functions.remove_molecules_without_smiles(dataset)
        print(f'number of entries after eliminating all elements with smiles=NaN: {len(dataset)}')

        print("==eliminating molecules without pchembl value==")
        dataset = Data_functions.eliminate_values_with_pchembl_Nan(dataset)
        print(f'number of entries after eliminating all elements with pchembl=NaN: {len(dataset)}')

        print('==Merging molecules occuring mulitple times==')
        dataset = Data_functions.remove_and_combine_molecules(dataset)
        print(f'Entries after merging double molecules: {len(dataset)}')

        dataset = dataset[dataset['target_pref_name'] == 'Beta-lactamase AmpC']
        print(f'== eliminated elements that did not target Beta-lactamase AmpC {dataset.shape[0]} molecules remaining ==')

        print('==creating a output variable y for binary descriptors==')
        dataset = Data_functions.calculate_output_variable_binary(dataframe=dataset, binary_cutoff=6)
        print('==creating a 3 classes output for classification==')
        dataset = Data_functions.calculate_output_variable_classes(dataframe=dataset, classes_cutoff=[5, 6])

        print('==Calculating molecule representations from canonical_smiles string==')
        dataset = Data_functions.calculateMolfromSmiles(dataset)

        print(f'===Saving Dataset to disk===')
        dataset.to_pickle('./Data/dataset/cleaned_dataset.pkl')
        print('file successfully printed to: ./Data/dataset/cleaned_dataset.pkl')

    elif mode == 'calculate_fingerprints':
        print('===Reading the prepared Dataset from disk===')
        try:
            dataset = read_pickle('./Data/dataset/cleaned_dataset.pkl')
        except:
            print(
                'No dataset was written into ./Data/dataset/cleaned_dataset.pkl consider running prepare_dataset to create the file')

        print('===Calculating and saving Padel descriptors to disk===')
        Data_functions.calculate_descriptors(dataset)
        print(f'calculated descriptors and saved to: ./Data/fingerprints')

    elif mode == 'filter_descriptors':
        Data_functions.filter_descriptors()

    elif mode == 'create_train_validate_test':
        Data_functions.create_train_test_validate()
        Data_functions.normalize()

    elif mode == 'pca':
        Data_functions.calc_pca()

    elif mode == 'multi-layer-perceptron':
        Data_functions.multi_layer_perceptron()

    elif mode == 'keras-deep-model':
        Data_functions.keras_deep_model()

    elif mode == 'random-forest':
        Data_functions.random_forest()

    elif mode == 'descriptor_stats':
        Data_functions.show_descriptor_stats()

    elif mode == 'test':
        #util.missing_values_to_nan()
        Data_functions.grid_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose the work-step you want to perform')
    parser.add_argument('mode',
                        choices=['download',
                                 'prepare_dataset',
                                 'calculate_fingerprints',
                                 'descriptor_stats',
                                 'test',
                                 'filter_descriptors',
                                 'create_train_validate_test',
                                 'multi-layer-perceptron',
                                 'keras-deep-model',
                                 'random-forest'],
                        help='choose the mode the programm should run in.')

    main(parser.parse_args().mode)
