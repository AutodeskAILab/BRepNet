"""
Run the model on each test file in the given folder.

Args:
   --dataset_file  The pathname of a file containing the standardization 
                   which you want to use.  The standardization needs to 
                   be compatible with the input features used to generate 
                   dataset and train the model

   --dataset_dir   The dataset directory.  This will typically be a folder 
                   containing step or stp files.

  --model          The pathname to the pretained model to use

  --input_features If the model was trained with non-standard features
                   then the feature list file needs to be given here.
                   The features and the feature standardization values
                   need to be compatible.  If you used feature_lists/all.json
                   (the default) then you don't need to add anything here.

"""
import argparse
from pytorch_lightning import Trainer
from pathlib import Path
import shutil

from models.brepnet import BRepNet
import utils.data_utils as data_utils
from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step

def copy_standardization(original_dataset):
    dataset = {}
    if "feature_standardization" in original_dataset:
        dataset["feature_standardization"] = original_dataset["feature_standardization"]
    elif "feature_normalization" in original_dataset:
        dataset["feature_normalization"] = original_dataset["feature_normalization"]
    else:
        assert False, "Dataset file must contain either feature_standardization or feature_normalization"
    return dataset


def find_file_stems_for_old_files(dataset_dir):
    file_stems = []
    topology_files = [ f for f in dataset_dir.glob("*_topology.json")]
    for file in topology_files:
        split_stem = file.stem.rpartition("_topology")[0]
        file_stems.append(split_stem)
    return file_stems

def create_new_test_set(dataset_dir, working_dir, feature_list_path):
    
    # First convert the step to the intermediate npz format
    extract_brepnet_data_from_step(
        dataset_dir, 
        working_dir, 
        feature_list_path=feature_list_path,
        force_regeneration=False
    )

    # Now file the file_stems for the intermediate files
    file_stems = [ f.stem for f in working_dir.glob("*.npz")]

    return file_stems


def create_old_test_set(dataset_dir):
    # Remove any cache dir
    cache_dir = dataset_dir / "cache"
    shutil.rmtree(cache_dir, ignore_errors=True)

    file_stems = find_file_stems_for_old_files(dataset_dir)
    batches = []
    current_batch = []
    num_faces_in_batch = 0
    max_faces_in_batch = 999
    for file_stem in file_stems:
        topology_file = dataset_dir / (file_stem + "_topology.json")
        top = data_utils.load_json_data(topology_file)
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

    if len(current_batch) > 0:
        batches.append(current_batch)

    return {"batches": batches}



def build_dataset_file(args):
    original_dataset_file = Path(args.dataset_file)
    original_dataset = data_utils.load_json_data(original_dataset_file)

    dataset_dir = Path(args.dataset_dir)
    working_dir = dataset_dir / "temp_working"
    if not working_dir.exists():
        working_dir.mkdir()

    dataset_file = working_dir / (original_dataset_file.stem + ".json")
    new_dataset = copy_standardization(original_dataset)

    if args.use_old_dataloader:
        new_dataset["test_set"] = create_old_test_set(dataset_dir)
    else:
        new_dataset["test_set"] = create_new_test_set(dataset_dir, working_dir, args.input_features)
        dataset_dir = working_dir
    
    data_utils.save_json_data(dataset_file, new_dataset)
    return dataset_file, dataset_dir


def do_eval(args):

    # We need to build a new dataset file with the 
    dataset_file, dataset_dir = build_dataset_file(args)
    args.dataset_file = dataset_file
    args.dataset_dir = dataset_dir

    logit_dir = dataset_dir / "logits"
    if not logit_dir.exists():
        logit_dir.mkdir()
    args.logit_dir = logit_dir

    embeddings_dir = dataset_dir / "embeddings"
    if not embeddings_dir.exists():
        embeddings_dir.mkdir()
    args.embeddings_dir = embeddings_dir

    if args.model is not None:
        brepnet = BRepNet.load_from_checkpoint(args.model, opts=args)
    else:
        print("WARNING!! No pre-trained model given.  Are you sure you want to evaluate with an untrained model?")
        brepnet = BRepNet(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(brepnet)

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BRepNet.add_model_specific_args(parser)
    parser.add_argument("--model", type=str, help="Model to load use for evaluation")
    return parser

def evaluate_folder(
        step_folder, 
        feature_standardization, 
        model=None, 
        input_features=None, 
        extra_args=None
    ):
    # We need to set up all the default brepnet arguments.  The easiest
    # way to do it is to use the same argument parser
    parser = get_argument_parser()
    args_to_parse = [
        "--dataset_dir", str(step_folder),
        "--dataset_file", str(feature_standardization),
        "--segment_names", "example_files/pretrained_models/segment_names.json"
    ]
    if model is None:
        print("Warning! No pretrained model given.  Using random network!")
    else:
        args_to_parse.extend(
            [ "--model", str(model) ]
        )

    if extra_args is not None:
        args_to_parse.extend(extra_args)

    if input_features is not None:
        args_to_parse.append("--input_features")
        args_to_parse.append(str(input_features))
    args = parser.parse_args(args_to_parse)
    do_eval(args)


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()

    do_eval(args)