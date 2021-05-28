"""
Run the model on each test file in the given folder.

The dataset_file needs to contain the standardization which you want to use
The dataset_dir needs to be the place where the files are
"""
import argparse
from pytorch_lightning import Trainer
from pathlib import Path

from models.brepnet import BRepNet
import utils.data_utils as data_utils

def copy_standardization(original_dataset):
    dataset = {}
    if "feature_standardization" in original_dataset:
        dataset["feature_standardization"] = original_dataset["feature_standardization"]
    elif "feature_normalization" in original_dataset:
        dataset["feature_normalization"] = original_dataset["feature_normalization"]
    else:
        assert False, "Dataset file must contain either feature_standardization or feature_normalization"
    return dataset


def find_file_stems(dataset_dir, use_old_dataloader):
    file_stems = []
    if use_old_dataloader:
        topology_files = [ f for f in dataset_dir.glob("*_topology.json")]
        for file in topology_files:
            split_stem = file.stem.rpartition("_topology.json")[0]
            file_stems.append(split_stem)
    else:
        assert False, "Not implemented yet"
    return file_stems

def create_new_test_set(dataset_dir, file_stems):
    assert False, "Not implemented"
    pass

def create_test_old_set(dataset_dir, file_stems):
    batches = []
    current_batch = []
    num_faces_in_batch = 0
    max_faces_in_batch = 999
    for file_stem in file_stems:
        topology_file = dataset_dir / (file_stem + "_topology.json")
        top = data_utils.load_json_data(original_dataset_file)
        num_faces = len(top["topology"]["faces"])
        if num_faces_in_batch == 0:
            current_batch.append(file_stem)
            num_faces_in_batch += num_faces
        elif num_faces_in_batch + num_faces > max_faces_in_batch:
            batches.append(current_batch)
            current_batch = [ file_stem ]
            num_faces_in_batch = num_faces
        else:
            current_batch.append(file_stem)
            num_faces_in_batch += num_faces

    return {"batches": batches}



def build_dataset_file(args):
    original_dataset_file = Path(args.dataset_file)
    original_dataset = data_utils.load_json_data(original_dataset_file)

    dataset_dir = Path(args.dataset_dir)
    dest = dataset_dir / (original_dataset_file.stem + ".json")

    new_dataset = copy_standardization(original_dataset)

    if args.use_old_dataloader:
        new_dataset["test_set"] = create_test_old_set(dataset_dir, file_stems)
    else:
        new_dataset["test_set"] = create_new_test_set(dataset_dir, file_stems)
    
    data_utils.save_json_data(dest, new_dataset)
    return dest


def do_eval(args):

    # We need to build a new dataset file with the 
    dataset_file = build_dataset_file(args)
    args.dataset_file = dataset_file

    brepnet = BRepNet.load_from_checkpoint(args.model, opts=args)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(brepnet)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BRepNet.add_model_specific_args(parser)
    parser.add_argument("--model", type=str, required=True,  help="Model to load use for evaluation")
    args = parser.parse_args()

    do_eval(args)