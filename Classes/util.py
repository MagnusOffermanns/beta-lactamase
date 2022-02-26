from os import listdir, remove, walk
from os.path import isfile, join

from keras.backend import sum, clip, epsilon
from tensorflow import cast

import mordred
import numpy as np
from mordred import error

import pandas as pd
from pandas import read_csv, concat, DataFrame, read_pickle
from typing import Callable
from gc import collect


def read_in_csv_from_directory(directorypath):
    csv_files = [f for f in listdir(directorypath) if isfile(join(directorypath, f))]
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    dataset = concat((read_csv(open(f'{directorypath}/{f}')) for f in csv_files))
    return dataset


def yes_or_no(message: str) -> bool:
    yes = {'yes', 'y'}
    no = {'no', 'n'}

    print(message)
    response = input('please input "yes" or "no":')
    while response not in set.union(yes, no):
        response = input('Not valid input please insert "yes" or "no":')

    if response in yes:
        return True
    else:
        return False


def read_all_all_filenames(path: str) -> list:
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break

    for i in range(len(f)):
        f[i] = dirpath + '/' + f[i]

    return f


def correct_fingerprints(path: str):
    """
    path str: is the path to the folder with the precalculated descriptors inside
    This functions adds the molecule chembl id to descriptors so that the descriptor can later be associated with its
    molecule id
    """

    file_list = read_all_all_filenames(path)

    cleaned_dataset = read_pickle('/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt'
                                  '/Machine_learning_and_deep_learning/Data/dataset/cleaned_dataset.pkl')

    for file in file_list:
        print(f'Loading file: {file}')
        data = pd.DataFrame(read_pickle(file))
        # data.loc[:, data.columns != 'molecule_chembl_id'] = data.loc[:, data.columns != 'molecule_chembl_id'].astype(float)
        print(f'size of dataframe. Collumns: {len(data.columns)} rows: {len(data.index)}')
        if not ('molecule_chembl_id' in data.columns):
            data.insert(0, 'molecule_chembl_id', cleaned_dataset['molecule_chembl_id'])
            data.to_pickle(file)

    return True


class FilterClass:

    def __init__(self):
        self.filtered = False

    def reset(self):
        self.filtered = False

    def filter_nan(self, series: pd.Series, max_number_nan: int):
        if series.isna().sum() > max_number_nan:
            self.filtered = True

    def filter_not_data_type(self, series: pd.Series, datatype: type):
        if not (series.dtype == datatype):
            self.filtered = True


class WorkChuncker:

    def __init__(self, func: Callable, path_to_file: str, path_to_result: str, size_chunck: int,
                 size_data_in_memory: int):

        if size_chunck > size_data_in_memory:
            raise Exception('size_chunk needs to be smaller than chunk in memory')

        self.func = func
        self.path_to_file: str = path_to_file
        self.path_to_result: str = path_to_result
        self.size_chunk: int = size_chunck
        self.size_data_in_memory: int = size_data_in_memory
        self.data_in_memory: DataFrame = pd.DataFrame()
        self.result_in_memory: DataFrame = pd.DataFrame()
        self.__memory_counter: int = 0
        self.__chunk_counter: int = 0
        self.__data_worked_of: bool = False
        self.__load_from_memory()
        fp = open(self.path_to_result, 'w')
        fp.close()
        pd.DataFrame().to_pickle(self.path_to_result)

    def start_new(self):
        self.__memory_counter = 0
        self.__chunk_counter = 0
        self.__data_worked_of = 0

    def is_finished(self) -> bool:
        if self.__data_worked_of + self.__memory_counter == -2:
            return True
        else:
            return False

    def __load_from_memory(self):
        if self.__memory_counter == -1:
            return

        dataset = read_pickle(self.path_to_file)

        if self.size_data_in_memory * (self.__memory_counter + 1) < len(dataset.index):
            self.data_in_memory = dataset.iloc[
                                  self.size_data_in_memory * self.__memory_counter:self.size_data_in_memory * (
                                          self.__memory_counter + 1), :]
            self.__memory_counter += 1
            self.__data_worked_of = 0
        else:
            self.data_in_memory = dataset.iloc[self.size_data_in_memory * self.__memory_counter:, :]
            self.__memory_counter = -1
            self.__data_worked_of = 0

        # deletes the big loaded dataset and frees it from memory
        del dataset
        collect()

    def get_result(self) -> pd.DataFrame:
        if self.is_finished():
            result = read_pickle(self.path_to_result)
            remove(self.path_to_result)
            return result
        else:
            raise Exception('Calculation is not done yet')

    def work(self):

        if self.size_chunk * (self.__chunk_counter + 1) < len(self.data_in_memory.index):

            temp_result = self.func(self.data_in_memory.iloc[self.size_chunk * self.__chunk_counter:self.size_chunk * (
                    self.__chunk_counter + 1)].copy())
            self.result_in_memory = concat([self.result_in_memory, temp_result])
            self.__chunk_counter += 1
        else:
            temp_result = self.func(self.data_in_memory.iloc[self.size_chunk * self.__chunk_counter:self.size_chunk * (
                    self.__chunk_counter + 1)].copy())
            self.result_in_memory = concat([self.result_in_memory, temp_result])
            entire_result = read_pickle(self.path_to_result)
            entire_result = concat([entire_result, self.result_in_memory])
            entire_result.to_pickle(self.path_to_result)
            self.result_in_memory = pd.DataFrame()
            self.__data_worked_of = -1
            self.__chunk_counter = 0
            self.__load_from_memory()
