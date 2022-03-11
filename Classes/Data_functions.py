import os
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, \
    Recall, AUC

from tensorflow import convert_to_tensor

from Classes.util import read_in_csv_from_directory, read_all_all_filenames, FilterClass, create_confusion_matrix, \
    oversample_dataset
from pandas import DataFrame, Series, merge, concat, read_pickle, to_numeric

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


from tqdm import tqdm
import pickle
from matplotlib.pyplot import plot, show, title, xlabel, ylabel, savefig, gcf, legend

from rdkit import Chem
from mordred import Calculator, descriptors


def download_and_unzip_files():
    """
    Downloads the dataset
    :return:
    """
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
    """
    :return: Dataframe containing the data of the csv files
    """
    dataset = read_in_csv_from_directory('./Data/raw_dataset')
    # reindex so that each element has their own index
    dataset.index = list(range(0, len(dataset.index)))
    # Number of entries
    print(f'number of entries in the dataframe: {len(dataset)}')
    return dataset


def eliminate_values_with_pchembl_Nan(dataset):
    """
    :param dataframe: Dataframe containing molecules with and without pchembl_value
    :return: cleaned Dataframe with all molecules removed that do not have a pchembl_value
    """
    dataset.dropna(subset=["pchembl_value"], inplace=True)
    return dataset


def remove_molecules_without_smiles(dataframe: DataFrame):
    """
    :param dataframe: Dataframe containing molecules with and without smiles representation
    :return: cleaned Dataframe with all molecules removed that do not have a smiles representation
    """
    dataframe = dataframe[~dataframe['canonical_smiles'].isna()]
    return dataframe


def remove_and_combine_molecules(dataframe: DataFrame):
    """
    Eliminates empty and double molecules
    :param dataframe: Input dataframe containing molecules with pchembl_value
    :return: Cleaned pandas dataset
    """
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
    """
    :param dataframe: Pandas Dataframe containing the pchembl_value together with the molecules
    :param binary_cutoff: denotign the border value between the two classes  ie class_1 < binary cutoff < class_2
    :return: The input dataframe with a new column containing the classes
    """
    # Create two classes based on  the pchembl_value
    dataframe['binary'] = False
    dataframe.loc[dataframe['pchembl_value'] > binary_cutoff, 'binary'] = True

    return dataframe


def calculate_output_variable_classes(dataframe: DataFrame, classes_cutoff: list) -> DataFrame:
    """
    :param dataframe: Pandas Dataframe containing the pchembl_value together with the molecules
    :param classes_cutoff: list of values denoting the borders between the classes [3,4] would result in 3 klasses.
     The first being all molecules with pchembl value < 3, the second with all molecules with pchembl value 3< < 4, and
     the  last being all molecules with pchembl value  4 <
    :return: The input dataframe with a new column containing the classes.
    """
    # Create three classes based on the column pchembl_value
    dataframe['classes'] = 0
    for i in range(len(classes_cutoff) - 1):
        dataframe.loc[dataframe['pchembl_value'] > classes_cutoff[i], 'classes'] = i + 1

    dataframe.loc[dataframe['pchembl_value'] > classes_cutoff[len(classes_cutoff) - 1], 'classes'] = len(classes_cutoff)

    return dataframe


def calculateMolfromSmiles(dataframe: DataFrame):
    """
    Transforms the general smiles notation into scikitChem Mole representation with which
    :param dataframe: Pandas dataframe containing the smiles representation of the molecules
    :return: Pandas dataframe containing the Mol representation of the molecules
    """
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

    for i in range(0, 50):
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
    Filters the calcualted descriptors by mordred based on certain properties. The the FilterClass can be adapted to filter
    by othe properties
    """
    filter_counter = 0
    filterobj = FilterClass()
    cleaned_dataset = read_pickle('/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt'
                                  '/Machine_learning_and_deep_learning/Data/dataset/cleaned_dataset.pkl')


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
    cleaned_dataset.to_pickle('./Data/training/working_dataset_int.pkl')
    print('successfully written to drive')


def create_train_test_validate():
    """
    Splits a dataset into 3 parts, train test and validate
    """
    train_ratio = 0.65
    validation_ratio = 0.15
    test_ratio = 0.20

    print('== creating test,train, and validation dataset ==')
    dataset = read_pickle('./Data/training/dataset_oversampled.pkl')
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
    """
    Normalizes train,test and evaluate dataset using the Standard scaler saves the
    scaled datasets to the drive
    """
    print('== Normalizing dataset ==')
    print('loading train,test and validation datasets from disk')
    X_train = read_pickle('./Data/training/X_train.pkl')
    X_validate = read_pickle('./Data/training/X_validate.pkl')
    X_test = read_pickle('./Data/training/X_test.pkl')

    print('scaling the dataset')
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = DataFrame(scaler.transform(X_train))
    X_validate_scaled = DataFrame(scaler.transform(X_validate))
    X_test_scaled = DataFrame(scaler.transform(X_test))

    X_train_scaled.to_pickle('./Data/training/X_train.pkl')
    X_validate_scaled.to_pickle('./Data/training/X_validate.pkl')
    X_test_scaled.to_pickle('./Data/training/X_test.pkl')
    print('written the scaled datasets back to  into ./Data/training')

    return None


def Oversample():
    """
    Oversampling to have 50% active molecules & 50% inactive molecules in the dataset
    """
    print('Oversampling to have 50% active molecules & 50% inactive molecules')
    dataset = read_pickle('./Data/training/working_dataset_floats.pkl')

    dataset_active = dataset.loc[dataset['binary'] == 1]
    dataset_inactive = dataset.loc[dataset['binary'] == 0]

    dataset_oversampled = oversample_dataset(dataset_active, dataset_inactive)

    dataset_oversampled.to_pickle('./Data/training/dataset_oversampled.pkl')
    print('written the scaled datasets back to  into ./Data/training')
    return None


def multi_layer_perceptron():
    """
    Trains a multi_layer_perceptron classfier, evaluates it and saves the classifier to disk
    :return:
    """

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
    """
    Trains a random forest classfier, evaluates it and saves the classifier to disk
    :return: none
    """
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
    """
    Calculates a PCA and removes features with little information
    :param nr_features: number of features that should remain after the pca calculation
    :return: none -> data is saved to disk
    """
    print(f'== Applying PCA with {nr_features} remaining ==')
    dataset = read_pickle('./Data/training/working_dataset.pkl')
    size_original_dataset = dataset.shape[0]
    pca = PCA(n_components=nr_features)
    dataset = pca.fit_transform(dataset.iloc[1:, 3:])
    print(f'Sucessfully reduced the features from {size_original_dataset} to {nr_features}')

    dataset.to_pickle('./Data/training/working_dataset.pkl')
    print('Sucessfully saved dataset ./Data/training/working_dataset.pkl')

    return None


def create_model(input_size):
    """
    :param input_size: the size of one set of input data
    :return: a keras sequential model
    """

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
    model_seq.add(Dense(400, input_shape=(input_size,), activation='relu'))
    model_seq.add(Dense(400, activation='relu'))
    model_seq.add(Dense(1, activation='sigmoid'))

    model_seq.compile(loss='binary_crossentropy', optimizer='Adam', run_eagerly=True, metrics=METRICS)
    return model_seq


def keras_deep_model():
    """
    Trains and evaluates a neural network defined in the function create_model
    :return: None
    """
    X_train = read_pickle('./Data/training/X_train.pkl')
    X_train = convert_to_tensor(X_train.to_numpy(dtype=np.float32))
    y_train = read_pickle('./Data/training/y_train.pkl')
    y_train = convert_to_tensor(y_train['binary'].to_numpy(dtype=np.bool_))

    X_valid = read_pickle('./Data/training/X_validate.pkl')
    X_valid = convert_to_tensor(X_valid.to_numpy(dtype=np.float32))
    y_valid = read_pickle('./Data/training/y_validate.pkl')
    y_valid = convert_to_tensor(y_valid['binary'].to_numpy(dtype=np.bool))

    X_test = read_pickle('./Data/training/X_test.pkl')
    X_test = convert_to_tensor(X_test.to_numpy(dtype=np.float32))
    y_test = read_pickle('./Data/training/y_test.pkl')
    y_test = convert_to_tensor(y_test['binary'].to_numpy(dtype=np.bool_))

    # generating the keras model
    model = create_model(X_test.shape[1])
    checkpointer = ModelCheckpoint(filepath='./Data/models/current_keras_deep_model.h5',
                                   verbose=1, save_best_only=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.005)

    history = model.fit(X_train, y_train,
                        epochs=70, batch_size=4096,
                        verbose=1, validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer, lr_reduction])

    model.load_weights('./Data/models/current_keras_deep_model.h5')
    score_new = model.evaluate(X_test, y_test)

    print(f'new score: {score_new}')
    with open('./Data/models/evaluation.pkl', 'wb') as file:
        pickle.dump(history, file)



def grid_search():
    """
    Searches for the optimal solver and the best alpha value. Currently not used in this implementation
    :return:
    """
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
    """
    Prints descriptor stats of all descriptors. Info include min,max,most frequent value etc.
    """

    file_list = read_all_all_filenames(
        '/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt/Machine_learning_and_deep_learning/Data/fingerprints')
    if len(file_list) == 0:
        raise Exception('fingerprint folder is empty. Consider running the program with the arguments "download", '
                        '"prepare_dataset" or "calculate_fingerprints" before')
    print(f'=======stating analysis=======')
    for file in file_list:

        data = DataFrame(read_pickle(file))
        data = data.loc[:, data.columns != 'molecule_chembl_id']
        data = data.apply(to_numeric, errors='coerce')
        print(f'== {file.rsplit("/", 1)[-1][:-4]} nr-columns: {len(data.columns)} ==')

        for column in data:
            print()
            print(f'Name: {"{:<15}".format(column)}')
            if data[column].isna().sum() == len(data[column].index):
                print(f'column only contains Nans')
                continue

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


def evaluate_models():
    """
    Prints out the evaluation for the random forest, the MLP model and the keras-deep model. Furthermore confusion matrixes are
    displayed and losscurves are shown
    :return:
    """
    X_test = read_pickle('./Data/training/X_test.pkl')
    y_test = read_pickle('./Data/training/y_test.pkl')

    # Random forest classifier
    randomforestobj = read_pickle('./Data/models/RandomForestClassifier')
    score = randomforestobj.score(X_test, y_test["binary"].astype(int))
    plot_random_forest = create_confusion_matrix(y_test["binary"].astype(int), randomforestobj.predict(X_test))
    title(f'random forest score: {score}')
    show()

    # MLPCClassifier
    MLPCobj = read_pickle('./Data/models/MLPCClassifier')

    score = MLPCobj.score(X_test, y_test["binary"].astype(int))

    print(f'score: {MLPCobj.score(X_test, y_test["binary"].astype(int))}')
    # loss curve
    fig = gcf()
    plot(MLPCobj.loss_curve_)
    xlabel('iterations')
    ylabel('loss')
    title(f'Loss multi-layer-perceptron')
    show()
    # confusion matrix
    create_confusion_matrix(y_test["binary"].astype(int), MLPCobj.predict(X_test))
    title(f'Multi-layer-perceptron score: {score}')
    show()

    # keras deep model
    keras_model = create_model(X_test.shape[1])
    keras_model.load_weights('./Data/models/current_keras_deep_model.h5')
    evaluation = read_pickle('./Data/models/evaluation.pkl')
    score = keras_model.evaluate(X_test, y_test['binary'].astype(int))
    create_confusion_matrix(y_test["binary"].astype(int), np.rint(keras_model.predict(X_test)))
    title(f'Keras-deep-model score: {score[5]}')
    show()

    plot(evaluation.history['accuracy'])
    plot(evaluation.history['val_accuracy'])
    title('model accuracy')
    ylabel('accuracy')
    xlabel('epoch')
    legend(['train', 'validate'], loc='upper left')
    show()

    plot(evaluation.history['loss'])
    plot(evaluation.history['val_loss'])
    title('model loss')
    ylabel('loss')
    xlabel('epoch')
    legend(['train', 'validate'], loc='upper right')
    show()

    return None
