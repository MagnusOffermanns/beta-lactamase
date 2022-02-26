import random
import unittest
import pandas as pd
import numpy as np
from Classes import Data_functions, util
from random import randint, uniform
from alive_progress import alive_bar


def create_test_data(num_molecules, doubles, uniques) -> pd.DataFrame:
    # create molecule_names
    molecule_names = []
    for i in range(num_molecules):
        molecule_names.append('mol' + str(i))
    molecule_names = molecule_names * doubles
    pchembl_values = list(range(num_molecules)) * doubles
    for i in range(0, uniques):
        molecule_names.extend(['mol' + str(randint(num_molecules + 1, 10000))])
        pchembl_values.extend([uniform(0, 1000)])

    # create a value that is outside 2*std_dev to be eliminated
    pchembl_values[0] = 1000
    pchembl_values[num_molecules] = 2
    # create a random set of values to check if the mean and the std_dev is calculated correctly
    for i in range(0, doubles):
        pchembl_values[i * num_molecules + 1] = uniform(0, 1000)

    # create smiles values with nans
    smiles = ['a'] * (num_molecules * doubles + uniques - 2) + [np.nan, np.nan]
    random.shuffle(smiles)

    test_data = {'molecule_chembl_id': molecule_names,
                 'canonical_smiles': smiles,
                 'standard_relation': ['='] * (num_molecules * doubles + uniques),
                 'standard_value': [randint(0, 10)] * (num_molecules * doubles + uniques),
                 'standard_units': ['nm'] * (num_molecules * doubles + uniques),
                 'standard_type': ['IC50'] * (num_molecules * doubles + uniques),
                 'pchembl_value': pchembl_values,
                 'target_pref_name': ['Beta_lactamase'] * (num_molecules * doubles + uniques),
                 'bao_lable': ['assay format'] * (num_molecules * doubles + uniques)
                 }

    # converting the dict into a dataframe
    dataframe = pd.DataFrame(data=test_data)

    # shuffeling and returning the dataframe
    return dataframe.sample(frac=1).reset_index(drop=True)


class DataCleaningTest(unittest.TestCase):

    def testremovingdoubles(self):
        test_DataFrame = create_test_data(4, 10, 5)

        result_DataFrame = Data_functions.remove_molecules_without_smiles(test_DataFrame)
        result_DataFrame = Data_functions.remove_and_combine_molecules(result_DataFrame)

        indexes_to_remove = []
        for index, row in test_DataFrame.iterrows():
            if pd.isna(row['canonical_smiles']):
                indexes_to_remove.append(index)
        test_DataFrame = test_DataFrame.drop(indexes_to_remove)

        # remove the molecules that are not unique
        # we extract the molecules that occur more often in the dataframe
        double_flag = test_DataFrame.duplicated(subset='molecule_chembl_id', keep=False)
        unique_molecules = test_DataFrame[~double_flag]
        test_DataFrame = test_DataFrame[double_flag]

        # add the column 'std_dev' and 'mean' to the uniques
        # unique_molecules.loc[:, 'std_dev'] = 0
        # unique_molecules.loc[:, 'mean'] = unique_molecules.loc[:, 'pchembl_value']

        # group duplicate molecules into lists
        molecule_groups = {}
        for index, row in test_DataFrame.iterrows():
            if not (row['molecule_chembl_id'] in molecule_groups.keys()):
                molecule_groups[row['molecule_chembl_id']] = [row]
            else:
                molecule_groups[row['molecule_chembl_id']].append(row)

        # Calculate Standard deviation and mean
        std_dev = {}
        mean = {}
        for molecules in molecule_groups:
            pchembl_values = []
            for i in range(len(molecule_groups[molecules])):
                pchembl_values.append(molecule_groups[molecules][i]['pchembl_value'])
            std_dev[molecules] = np.std(pchembl_values, ddof=0)
            mean[molecules] = np.mean(pchembl_values)

        # add the standard deviations and means from the unique molecules for later checking
        for index, row in unique_molecules.iterrows():
            mean[row['molecule_chembl_id']] = row['pchembl_value']
            std_dev[row['molecule_chembl_id']] = 0.0

        # eliminate molecules where 0 > abs(pchembl_value-mean)- 2 * std_dev
        indexes_to_be_removed = []
        for molecules in molecule_groups.items():
            for i, element in enumerate(molecules[1]):
                if abs(element['pchembl_value'] - mean[element['molecule_chembl_id']]) - 2 * std_dev[
                    element['molecule_chembl_id']] > 0:
                    indexes_to_be_removed.append(i)
            indexes_to_be_removed.reverse()
            for index in indexes_to_be_removed:
                del molecules[1][index]
            indexes_to_be_removed = []

        # combine double molecules to one molecule
        for molecules in molecule_groups.values():
            pchembl_values = []
            for molecule in molecules:
                pchembl_values.append(molecule['pchembl_value'])
            molecules[0]['pchembl_value'] = np.mean(pchembl_values)
            molecule_groups[molecule['molecule_chembl_id']] = molecules[0]

        # append the molecules that did not occur double and were removed before
        for index, row in unique_molecules.iterrows():
            molecule_groups[row['molecule_chembl_id']] = row

        # check if the means are calculated correctly
        for index, row in result_DataFrame.iterrows():
            self.assertAlmostEqual(row['std_dev'], std_dev[row['molecule_chembl_id']],
                                   msg=f'std deviation of {row["molecule_chembl_id"]} is not equal to the test value')
            self.assertAlmostEqual(row['mean'], mean[row['molecule_chembl_id']],
                                   msg=f'mean of {row["molecule_chembl_id"]} is not equal to the test value')

        # check if the number of elements in the test data and the result dataframe are the same
        self.assertEqual(result_DataFrame.shape[0], len(molecule_groups), 'Number of molecules is not the same')

        # check if the new calculated pchembl values are the same
        for index, row in result_DataFrame.iterrows():
            self.assertAlmostEqual(row['pchembl_value'], molecule_groups[row['molecule_chembl_id']]['pchembl_value'])

        # check if there are any elements that have  'canonical_smiles'==nan
        self.assertEqual(result_DataFrame['canonical_smiles'].isna().sum(), 0,
                         'Values with canonical_smiles == nan are still present after running')
        self.assertEqual(test_DataFrame['canonical_smiles'].isna().sum(), 0,
                         'Values with canonical_smiles == nan are still present in the test_dataframe')

        return True

    def test_calculatingMol(self):

        dataset = pd.read_pickle('./Data/dataset/cleaned_dataset.pkl')
        dataset = dataset.head(1000)
        dataset = Data_functions.calculateMolfromSmiles(dataset)
        return True

    def test_calculating_descriptors(self):
        dataset = pd.read_pickle('./Data/dataset/cleaned_dataset.pkl')
        dataset = dataset.head(100)

        dataset = Data_functions.calculate_descriptors(dataset)

    def test_creating_outputs(self):
        test_DataFrame = create_test_data(4, 10, 5)
        dataframe = Data_functions.calculate_output_variable(dataframe=test_DataFrame, binary_cutoff=500,
                                                             classes_cutoff=[200, 400, 600, 800])
        return True

    def test_WorkChuncker(self):
        def list_comprehension_sum(dataframe: pd.DataFrame) -> pd.DataFrame:
            dataframe.loc[:, 'sum'] = dataframe.apply(np.sum, axis=1)
            return dataframe

        dataset = pd.DataFrame(np.random.randint(0, 100, size=(4010, 4)), columns=list('ABCD'))
        dataset.to_pickle('./temp/random_data.pkl')

        workchunker_obj = util.WorkChuncker(list_comprehension_sum, './temp/random_data.pkl',
                                            './temp/result_workchuncker_tester.pkl', 500,
                                            2000)

        with alive_bar(int(np.ceil(4000 / workchunker_obj.size_chunk))) as bar:
            while not (workchunker_obj.is_finished()):
                workchunker_obj.work()
                bar()

        result = workchunker_obj.get_result()

    def test_correct_fingerprints(self):
        util.correct_fingerprints(
            '/media/magnus/Main_volume/moechtegerndesktop/Universitaet_Klagenfurt/Machine_learning_and_deep_learning/Data/fingerprints')
