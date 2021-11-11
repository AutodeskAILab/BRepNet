"""
This code should be used to generate a random training and validation split
for the npz files generated using `pipeline/extract_feature_data.py`

The feature standardization will be computed from the features in the randomly 
split out training data only.

The train_test.json file is passed in to ensure the official test set
is also held out.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import tqdm

import utils.data_utils as data_utils
from pipeline.running_stats import RunningStats

def stats_to_json(stats):
    data = []
    for stat in stats:
        data.append(
            {
                "mean": stat.mean(),
                "standard_deviation": stat.standard_deviation()
            }
        )
    return data

def append_to_stats(arr, stats):
    num_entities = arr.shape[0]
    num_features = arr.shape[1]
    if len(stats) == 0:
        for i in range(num_features):
            stats.append(RunningStats())
    else:
        assert len(stats) == num_features

    for i in range(num_entities):
        for j in range(num_features):
            stats[j].push(arr[i,j])


def find_standardization(train_files):
    face_feature_stats = []
    edge_feature_stats = []
    coedge_feature_stats = []
    for file in tqdm.tqdm(train_files):
        data = data_utils.load_npz_data(file)
        append_to_stats(data["face_features"], face_feature_stats)
        append_to_stats(data["edge_features"], edge_feature_stats)
        append_to_stats(data["coedge_features"], coedge_feature_stats)

    data = {
        "face_features": stats_to_json(face_feature_stats),
        "edge_features": stats_to_json(edge_feature_stats),
        "coedge_features": stats_to_json(coedge_feature_stats),
    }
    return data

def check_stats_for_zero_standard_deviation(stats):
    """
    When adding new features, especially with Open Cascade,
    it is possible to call functions which always return the 
    same value.  Here we print a big warning about that, so 
    unsuspecting users of the code will not make the mistake
    of using the bad features
    """
    eps = 1e-7
    for feature in stats:
        if feature["standard_deviation"] < eps:
            print("WARNING! - At least one feature has zero standard deviation")

def check_for_zero_standard_deviation(standardization_data):
    check_stats_for_zero_standard_deviation(standardization_data["face_features"])
    check_stats_for_zero_standard_deviation(standardization_data["edge_features"])
    check_stats_for_zero_standard_deviation(standardization_data["coedge_features"])


def file_stems(file_list):
    stems = []
    for file in file_list:
        stems.append(file.stem)
    return stems

def check_files_exist(file_list, npz_folder):
    npz_pathnames = []
    for file in file_list:
        npz_pathname = npz_folder / f"{file}.npz"
        if npz_pathname.exists():
            npz_pathnames.append(npz_pathname)
    return npz_pathnames

def get_train_test_lists_from_file(train_test_file):
    train_test = data_utils.load_json_data(train_test_file)
    if not "train" in train_test:
        print("The train/test file must contain key 'train'")
        sys.exit(1)

    if not "test" in train_test:
        print("The train/test file must contain key 'test'")
        sys.exit(1)

    # Copy the 
    train_val_files = train_test["train"]
    test_files = train_test["test"]
    return train_val_files, test_files


def get_train_test_lists_from_split(npz_folder, test_split):
    files = [ f.stem for f in npz_folder.glob("*.npz")]
    train_val_files, test_files = train_test_split(files, test_size=test_split, random_state=234)
    output_train_test_file = npz_folder / "train_test.json"
    if output_train_test_file.exists():
        print(f"Error!  The file {output_train_test_file} already exists.  Delete it if you want to regenerate it")
        sys.exit(1)
    train_test = {}
    train_test["train"] = train_val_files
    train_test["test"] = test_files
    data_utils.save_json_data(output_train_test_file, train_test)
    return train_val_files, test_files


def build_dataset_file(
        npz_folder, 
        dataset_file, 
        validation_split, 
        train_test_file=None,
        test_split=None
    ):
    if train_test_file is not None:
        train_val_files, test_files = get_train_test_lists_from_file(train_test_file)
    else:
        train_val_files, test_files = get_train_test_lists_from_split(npz_folder, test_split)

    train_val_files = check_files_exist(train_val_files, npz_folder)
    test_files = check_files_exist(test_files, npz_folder)

    train_files, validation_files = train_test_split(train_val_files, test_size=validation_split, random_state=567)

    standardization_data = find_standardization(train_files)
    check_for_zero_standard_deviation(standardization_data)
    data = {
        "training_set": file_stems(train_files),
	    "validation_set": file_stems(validation_files),
	    "test_set": file_stems(test_files),
        "feature_standardization": standardization_data
    }
    data_utils.save_json_data(dataset_file, data)
    print("Completed pipeline/build_dataset_file.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_folder", type=str, required=True, help="Path to the folder containing npz files from extract_feature_data")
    parser.add_argument("--train_test", type=str, help="Pathname to the file containing the train/test split")
    parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of the data to add to the test set")
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.18, 
        help="The fraction of examples from the available training file for the validation set"
    )
    parser.add_argument("--dataset_file", type=str, required=True, help="Pathname to save the generated dataset file")
    args = parser.parse_args()

    npz_folder = Path(args.npz_folder)
    if not npz_folder.exists():
        print("The npz folder does not exist")
        sys.exit(1)

    test_split = None
    train_test_file = None
    if args.train_test is not None:
        train_test_file = Path(args.train_test)
        if not train_test_file.exists():
            print("The train test file does not exist")
            sys.exit(1)
    else:
        test_split = args.test_split

    dataset_file = Path(args.dataset_file)
    

    build_dataset_file(npz_folder, dataset_file,  args.validation_split, train_test_file, test_split)