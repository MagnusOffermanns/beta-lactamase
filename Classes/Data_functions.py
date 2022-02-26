import math
import os
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import  TruePositives,FalsePositives,TrueNegatives,FalseNegatives,BinaryAccuracy,Precision,Recall,AUC
from tensorflow import convert_to_tensor

from Classes.util import read_in_csv_from_directory, read_all_all_filenames, FilterClass
from pandas import DataFrame, Series, merge, concat, read_pickle, to_numeric

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

from tqdm import tqdm
import pickle
import h5py
from matplotlib.pyplot import plot, show, title, xlabel, ylabel, savefig, gcf

from rdkit import Chem
from mordred import Calculator, descriptors


def download_and_unzip_files():
    print('=== Started Download ===')
    # downloads files
    os.system(
        'wget https://github.com/dataprofessor/beta-lactamase/raw/main/beta_lactamase_CHEMBL29.zip -O ./Data/raw_dataset/beta_lactamase_CHEMBL29.zip -q && echo download successfull')
    # unzips
    os.system('unzip -qq -o ./Data/raw_dataset/beta_lactamase_CHEMBL29.zip -d ./Data/raw_dataset/')
    # remove zip file
    os.system('rm ./Data/raw_dataset/beta_lactamase_CHEMBL29.zip')
    # shows
    number_of_files = len(
        [name for name in os.listdir("./Data/raw_dataset") if os.path.isfile("./Data/raw_dataset/" + name)])
    print(f'Number of files unpacked: {number_of_files} ')


def convert_csv_to_Dataframe():
    dataset = read_in_csv_from_directory('./Data/raw_dataset')
    # reindex so that each element has their own index
    dataset.index = list(range(0, len(dataset.index)))
    # Number of entries
    print(f'number of entries in the dataframe: {len(dataset)}')
    return dataset


def eliminate_values_with_pchembl_Nan(dataset):
    dataset.dropna(subset=["pchembl_value"], inplace=True)
    return dataset


def remove_molecules_without_smiles(dataframe: DataFrame):
    dataframe = dataframe[~dataframe['canonical_smiles'].isna()]
    return dataframe


def remove_and_combine_molecules(dataframe: DataFrame):
    # we extract the molecules that occur more often in the dataframe
    bool_unique_or_not = dataframe.duplicated(subset='molecule_chembl_id', keep=False)
    # all_duplicates contains now all molecules that occur multiple times
    all_duplicates = dataframe[bool_unique_or_not]
    # uniques contains all molecules that occur only once in the dataframe
    uniques = dataframe[~bool_unique_or_not]

    # calulates the sample standard deviation (ddof=1) of each molecule in the all_duplicates dataset
    std_deviation_of_double_molecules = all_duplicates.groupby('molecule_chembl_id')['pchembl_value'].std(
        ddof=0).to_frame()
    std_deviation_of_double_molecules = std_deviation_of_double_molecules.reset_index()
    std_deviation_of_double_molecules = std_deviation_of_double_molecules.rename(columns={'pchembl_value': 'std_dev'})

    # calculates the mean of the pchembl values of the molecules
    mean_of_double_molecules = all_duplicates.groupby('molecule_chembl_id')['pchembl_value'].mean().to_frame()
    mean_of_double_molecules = mean_of_double_molecules.reset_index()
    mean_of_double_molecules = mean_of_double_molecules.rename(columns={'pchembl_value': 'mean'})

    # match the std deviations to the fitting molecule
    all_duplicates = merge(all_duplicates, std_deviation_of_double_molecules, how='left', left_on='molecule_chembl_id',
                           right_on='molecule_chembl_id')

    # match the means to the fitting molecule
    all_duplicates = merge(all_duplicates, mean_of_double_molecules, how='left', left_on='molecule_chembl_id',
                           right_on='molecule_chembl_id')

    # Filter values based on the pchembl value
    cleaned_duplicates = all_duplicates[
        abs(all_duplicates['pchembl_value'] - all_duplicates['mean']) <= 2 * all_duplicates['std_dev']]

    # caluclates the mean of each molecule group
    calculated_mean_of_each_molecule = cleaned_duplicates.groupby('molecule_chembl_id', as_index=False)[
        'pchembl_value'].mean()

    # eliminates the double molecules in the cleaned_duplicates dataframe
    unique_duplicates = cleaned_duplicates.drop_duplicates(subset=['molecule_chembl_id'])

    # drops the pchembl_value of the unique_duplicates so that it can be filled up with the calucalted means
    unique_duplicates = unique_duplicates.drop(labels="pchembl_value", axis=1)

    # merges the calculated mean values of the molecules with the
    unique_duplicates = merge(unique_duplicates, calculated_mean_of_each_molecule, how='left',
                              left_on='molecule_chembl_id',
                              right_on='molecule_chembl_id')

    # eliminates the Nan when joining the unique elements to the double elements in the collumn 'std_dev' and 'mean'
    uniques.loc[:, 'std_dev'] = 0
    uniques.loc[:, 'mean'] = uniques.loc[:, 'pchembl_value']

    return concat([uniques, unique_duplicates], axis=0, ignore_index=True)


def calculate_output_variable_binary(dataframe: DataFrame, binary_cutoff: float) -> DataFrame:
    # Create two classes based on  the pchembl_value
    dataframe['binary'] = False
    dataframe.loc[dataframe['pchembl_value'] > binary_cutoff, 'binary'] = True

    return dataframe


def calculate_output_variable_classes(dataframe: DataFrame, classes_cutoff: list) -> DataFrame:
    # Create three classes based on the column pchembl_value
    dataframe['classes'] = 0
    for i in range(len(classes_cutoff) - 1):
        dataframe.loc[dataframe['pchembl_value'] > classes_cutoff[i], 'classes'] = i + 1

    dataframe.loc[dataframe['pchembl_value'] > classes_cutoff[len(classes_cutoff) - 1], 'classes'] = len(classes_cutoff)

    return dataframe


def calculateMolfromSmiles(dataframe: DataFrame):
    print('start')
    tqdm.pandas()
    dataframe['molecules'] = dataframe['canonical_smiles'].progress_apply(lambda x: Chem.MolFromSmiles(x))
    print('done')
    return dataframe


def calculate_descriptors(dataframe: DataFrame):
    '''
    :param dataframe: input dataframe containing the mol representation of molecules
    :return: The Dataframe appended with all unfiltered modred descriptors
    '''

    df = DataFrame()

    for i in range(5, 6):
        print(f'calculating: {descriptors.all[i].__name__[len("mordred."):]} ')
        fp = open('./Data/fingerprints/' + descriptors.all[i].__name__[len('mordred.'):] + '.pkl', 'w+')
        fp.close()
        calc = Calculator(ignore_3D=True)
        calc.register(descriptors.all[i])
        df = concat([dataframe['molecule_chembl_id'], calc.pandas(dataframe['molecules'], quiet=False, nproc=3)],
                    axis=1)
        print(f'calculated the: {descriptors.all[i].__name__} descriptor')
        df.to_pickle('./Data/fingerprints/' + descriptors.all[i].__name__[len('mordred.'):] + '.pkl')


def filter_descriptors():
    """
    :return:
    """
    filter_counter = 0
    filterobj = FilterClass()
    cleaned_dataset = read_pickle('/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt'
                                  '/Machine_learning_and_deep_learning/Data/dataset/cleaned_dataset.pkl')

    # cleaned_dataset = cleaned_dataset.sample(frac=0.70)

    zeros = DataFrame([[''] * cleaned_dataset.shape[1]], columns=cleaned_dataset.columns)
    cleaned_dataset = zeros.append(cleaned_dataset, ignore_index=True)
    cleaned_dataset = cleaned_dataset[["molecule_chembl_id", "binary", 'classes']]

    file_list = read_all_all_filenames(
        '/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt/Machine_learning_and_deep_learning/Data/fingerprints')

    if len(file_list) == 0:
        raise Exception('fingerprint folder is empty. Consider running the program with the arguments "download", '
                        '"prepare_dataset" or "calculate_fingerprints" before')

    print(f'======= filtering values =======')
    for index, file in enumerate(file_list):
        # if index > 60:
        #    break
        print(f'== {file.rsplit("/", 1)[-1][:-4]} == index: {index}')
        data = DataFrame(read_pickle(file))
        data = data.loc[:, data.columns != 'molecule_chembl_id']
        temp_data = DataFrame()
        for column in data:
            data[column] = to_numeric(data[column], errors='coerce')
            filterobj.filter_nan(data[column], 5000)
            filterobj.filter_not_data_type(data[column], int)

            if not filterobj.filtered:
                filter_counter = filter_counter + 1
                temp_data[column] = Series([file.rsplit("/", 1)[-1][:-4]]).append(data[column], ignore_index=True)
            filterobj.reset()
        cleaned_dataset = concat([cleaned_dataset, temp_data], axis=1)
        print(f'{filter_counter} columns of {data.shape[1]} columns eliminated')
        filter_counter = 0

    cleaned_dataset = cleaned_dataset.copy()
    print(f'We now have {cleaned_dataset.shape[0]} molecules with {cleaned_dataset.shape[1]} features')

    cleaned_dataset.dropna(inplace=True)
    print(f'===After eliminating nans {cleaned_dataset.shape[0]} molecules remain===')
    # df_1 = cleaned_dataset.iloc[:math.floor(len(cleaned_dataset.index)/3), :]
    # df_2 = cleaned_dataset.iloc[math.floor(len(cleaned_dataset.index)/3)+1:math.floor(len(cleaned_dataset.index)/3*2), :]
    # df_3 = cleaned_dataset.iloc[math.floor(len(cleaned_dataset.index)/3*2)+1:, :]
    # del cleaned_dataset

    # df_1.to_pickle('./Data/training/working_dataset1.pkl')
    # del df_1
    # print('df1 sucess')
    # df_2.to_pickle('./Data/training/working_dataset2.pkl')
    # del df_2
    # print('df2 success')
    # df_3.to_pickle('./Data/training/working_dataset3.pkl')
    # print('df3 success')
    cleaned_dataset.to_pickle('./Data/training/working_dataset_int.pkl')
    print('successfully written to drive')


def create_train_test_validate():
    train_ratio = 0.65
    validation_ratio = 0.15
    test_ratio = 0.20

    print('== creating test,train, and validation dataset ==')
    dataset = read_pickle('./Data/training/working_dataset_floats.pkl')
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[1:, 3:], dataset.iloc[1:, [1, 2]],
                                                        test_size=test_ratio + validation_ratio,
                                                        random_state=random.randint(0, 10000))

    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test,
                                                              test_size=test_ratio / (test_ratio + validation_ratio),
                                                              random_state=random.randint(0, 100000))

    print('Saving the train,test and validate datasets to disk')
    X_train.to_pickle('./Data/training/X_train.pkl')
    y_train.to_pickle('./Data/training/y_train.pkl')

    X_validate.to_pickle('./Data/training/X_validate.pkl')
    y_validate.to_pickle('./Data/training/y_validate.pkl')

    X_test.to_pickle('./Data/training/X_test.pkl')
    y_test.to_pickle('./Data/training/y_test.pkl')

    print('Sucessfully written test,validate and train dataset into the folder ./Data/training')

    return None


def normalize():
    print('== Normalizing dataset ==')
    print('loading train,test and validation datasets from disk')
    X_train = read_pickle('./Data/training/X_train.pkl')
    # X_validate = read_pickle('./Data/training/X_validate.pkl')
    X_test = read_pickle('./Data/training/X_test.pkl')

    print('scaling the dataset')
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = DataFrame(scaler.transform(X_train))
    # X_validate_scaled = DataFrame(scaler.transform(X_validate))
    X_test_scaled = DataFrame(scaler.transform(X_test))

    X_train_scaled.to_pickle('./Data/training/X_train.pkl')
    # X_validate_scaled.to_pickle('./Data/training/X_validate.pkl')
    X_test_scaled.to_pickle('./Data/training/X_test.pkl')
    print('written the scaled datasets back to  into ./Data/training')

    return None


def multi_layer_perceptron():
    print('== started estimating: Multi-Layer-Perceptron ==')
    X_train = read_pickle('./Data/training/X_train.pkl')
    y_train = read_pickle('./Data/training/y_train.pkl')

    learning_rate = 0.0005
    model = MLPClassifier(hidden_layer_sizes=(400, 400,),
                          random_state=random.randint(0, 999999999),
                          max_iter=200,
                          verbose=True,
                          tol=0.000001,
                          learning_rate_init=learning_rate)

    model.fit(X_train, y_train['binary'].astype(int))

    X_test = read_pickle('./Data/training/X_test.pkl')
    y_test = read_pickle('./Data/training/y_test.pkl')

    score = model.score(X_test, y_test["binary"].astype(int))

    print(f'score: {model.score(X_test, y_test["binary"].astype(int))}')
    fig = gcf()
    plot(model.loss_curve_)
    xlabel('iterations')
    ylabel('loss')
    title(f'learningrate: {learning_rate} score: {score}')
    show()
    fig.savefig(f'./plots_figures/losscurve_{str(learning_rate).replace(".", "_")}_nr_iterations_{model.n_iter_}.eps',
                format='eps')

    with open('./Data/models/MLPCClassifier', 'wb') as f:
        pickle.dump(model, f)

    return None


def random_forest():
    print('== started estimating: Random forest ==')
    X_train = read_pickle('./Data/training/X_train.pkl')
    y_train = read_pickle('./Data/training/y_train.pkl')

    model = RandomForestClassifier(n_estimators=200, verbose=1)
    model.fit(X_train, y_train['binary'].astype(int))

    X_test = read_pickle('./Data/training/X_test.pkl')
    y_test = read_pickle('./Data/training/y_test.pkl')

    print(f'random forest score: {model.score(X_test, y_test["binary"].astype(int))}')

    with open('./Data/models/RandomForestClassifier', 'wb') as f:
        pickle.dump(model, f)

    return None


def calc_pca(nr_features: int):
    print(f'== Applying PCA with {nr_features} remaining ==')
    dataset = read_pickle('./Data/training/working_dataset.pkl')
    size_original_dataset = dataset.shape[0]
    pca = PCA(n_components=nr_features)
    dataset = pca.fit_transform(dataset.iloc[1:, 3:])
    print(f'Sucessfully reduced the features from {size_original_dataset} to {nr_features}')

    dataset.to_pickle('./Data/training/working_dataset.pkl')
    print('Sucessfully saved dataset ./Data/training/working_dataset.pkl')

    return None


def keras_deep_model():

    def model_class():
        METRICS = [
            TruePositives(name='tp'),
            FalsePositives(name='fp'),
            TrueNegatives(name='tn'),
            FalseNegatives(name='fn'),
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

        model_seq = Sequential()
        model_seq.add(Dense(600, input_shape=(X_test.shape[1],)))
        model_seq.add(LeakyReLU(alpha=0.02))
        model_seq.add(Dense(400))
        model_seq.add(Dropout(0.25))
        model_seq.add(Dense(200))
        model_seq.add(Dense(50))
        model_seq.add(Dense(20))
        model_seq.add(Dense(1, activation='sigmoid'))
        model_seq.compile(loss='binary_crossentropy', optimizer='adam',run_eagerly=True, metrics=METRICS)
        return model_seq

    X_train = read_pickle('./Data/training/X_train.pkl')
    X_train=convert_to_tensor(X_train.to_numpy(dtype=np.float32))
    y_train = read_pickle('./Data/training/y_train.pkl')
    y_train = convert_to_tensor(y_train['binary'].to_numpy(dtype=np.bool_))

    X_valid = read_pickle('./Data/training/X_validate.pkl')
    X_valid=convert_to_tensor(X_valid.to_numpy(dtype=np.float32))
    y_valid = read_pickle('./Data/training/y_validate.pkl')
    y_valid = convert_to_tensor(y_valid['binary'].to_numpy(dtype=np.bool))

    X_test = read_pickle('./Data/training/X_test.pkl')
    X_test = convert_to_tensor(X_test.to_numpy(dtype=np.float32))
    y_test = read_pickle('./Data/training/y_test.pkl')
    y_test = convert_to_tensor(y_test['binary'].to_numpy(dtype=np.bool_))

    # generating the keras model
    model = model_class()
    checkpointer = ModelCheckpoint(filepath='./Data/models/current_keras_deep_model.h5',
                                   verbose=1, save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(X_train, y_train,
                        epochs=70, batch_size=4096,
                        verbose=1, validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer, lr_reduction])

    model.load_weights('./Data/models/current_keras_deep_model.h5')
    old_model = model_class()
    old_model.load_weights('./Data/models/keras_deep_model.h5')
    score_new = model.evaluate(X_test, y_test)
    score_old = old_model.evaluate(X_test, y_test)
    print(f'new score: {score_new} old score: {score_old}')
    if score_new > score_old:
        with open('./Data/models/evaluation.pkl', 'wb') as file:
            pickle.dump(history, file)


    # score_old = old_model.evaluate(X_test, y_test)


def grid_search():
    sourceFile = open('./grid_search_evaluation.txt', 'w+')
    print('test', file=sourceFile)
    sourceFile.close()

    X_train = read_pickle('./Data/training/X_train.pkl')
    y_train = read_pickle('./Data/training/y_train.pkl')

    learning_rate = 0.0005
    model = MLPClassifier(hidden_layer_sizes=(400, 400,),
                          random_state=random.randint(0, 999999999),
                          max_iter=200,
                          verbose=True,
                          tol=0.000001,
                          learning_rate_init=learning_rate)

    grid_search_obj = GridSearchCV(model, {'alpha': 10.0 ** -np.arange(1, 7), 'solver': ('lbfgs', 'sgd', 'adam')},
                                   verbose=1)
    grid_search_obj.fit(X_train, y_train['binary'].astype(int))

    sourceFile = open('./grid_search_evaluation', 'w+')
    print(str(sorted(grid_search_obj.cv_results_)), file=sourceFile)
    sourceFile.close()

    return None


def show_descriptor_stats():
    file_list = read_all_all_filenames(
        '/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt/Machine_learning_and_deep_learning/Data/fingerprints')
    if len(file_list) == 0:
        raise Exception('fingerprint folder is empty. Consider running the program with the arguments "download", '
                        '"prepare_dataset" or "calculate_fingerprints" before')
    print(f'=======stating analysis=======')
    for file in file_list:

        data = DataFrame(read_pickle(file))
        data = data.loc[:, data.columns != 'molecule_chembl_id']
        print(f'== {file.rsplit("/", 1)[-1][:-4]} nr-columns: {len(data.columns)} ==')

        for column in data:
            print()
            print(f'Name: {"{:<15}".format(column)}')
            print(
                f'max : {"{0:07f}".format(data[column].max(skipna=True))}      min: {"{0:07f}".format(data[column].min(skipna=True))}')
            print(
                f'mean: {"{0:07f}".format(data[column].mean(skipna=True))} std_dev: {"{0:07f}".format(data[column].std(skipna=True))}')
            print(
                f'uniq: {"{0:07d}".format(len(data[column].unique()))} num_nan: {"{0:07d}".format(data[column].isna().sum())}')
            print(
                f'com : {"{0:07f}".format(data[column].value_counts().keys()[0])} freq: {"{0:07d}".format(data[column].value_counts().iloc[0])} ')
            data_statistic = data[column].apply(type).value_counts(dropna=False)
            for i in range(len(data_statistic.index)):
                print(
                    f'type: {"{:<15}".format(str(data_statistic.keys()[i]))} | nr: {"{:<10}".format(str(data_statistic.values[i]))} ')

        print('\n')
