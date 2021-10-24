"""
A script to perform all the required processing on the Fusion 360 Gallery 
segmentation dataset.  This script just needs to be run once before you
start training.

Instructions for using the script

1) Download the Fusion 360 Gallery segmentation dataset from 
   https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/segmentation/s2.0.0/s2.0.0.zip

2) Unzip the dataset

3) Run the script like this

    python -m pipeline.quickstart /path/to/s2.0.0

4) You can then train the model using the command line printed in the shell

"""
import argparse
from pathlib import Path
import sys

from pipeline.build_dataset_file import build_dataset_file
from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the folder you unzipped the segmentation dataset")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of threads for processing")
    parser.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists")
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.3, 
        help="The fraction of examples from the available training file for the validation set"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"The dataset directory is not found at {dataset_dir}")
        sys.exit(1)


    step_path = dataset_dir / "breps/step"
    if not step_path.exists():
        print(f"The step directory is not found at {step_path}.  Step data was not included before version s2.0.0")
        sys.exit(1)



    seg_dir =  dataset_dir / "breps/seg"
    if not seg_dir.exists():
        print(f"The segmentation directory is not found at {seg_dir}.  Please use dataset version s2.0.0 or later")
        sys.exit(1)


    train_test_file = dataset_dir / "train_test.json"
    if not train_test_file.exists():
        print(f"The file {train_test_file} is missing.  If you are building a new dataset please read docs/building_your_own_dataset.md")
        sys.exit(1)

    # This is where the intermediate files will be generated
    processed_dir = dataset_dir / "processed"
    if not processed_dir.exists():
        processed_dir.mkdir()

    # The new dataset file will be created in the processed folder
    # with the name dataset.json
    dataset_file = processed_dir / "dataset.json"

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    # This script converts the STEP files into an intermediate format
    extract_brepnet_data_from_step(
        step_path, 
        processed_dir, 
        seg_dir=seg_dir, 
        feature_list_path=feature_list_path, 
        num_workers=args.num_workers
    )

    # This script created train/validation split.   The held out
    # test set is defined by the train_test_file.
    # It also computes the feature standardization for the dataset
    build_dataset_file(
        processed_dir, 
        dataset_file,  
        args.validation_split, 
        train_test_file
    )

    if not dataset_file.exists():
        print(f"Error! Failed to generate {dataset_file}")
        sys.exit(1)
    else:
        print("Processing complete")
        print("You are now ready to train the model using the command")
        print("python -m train.train \\")
        print(f"  --dataset_file {dataset_file} \\")
        print(f"  --dataset_dir {processed_dir} \\")
        print(f"  --max_epochs 200")
        print(f" ")
        print(f"To reproduce the results in the paper  ")
        print(f"BRepNet: A Topological Message Passing System for Solid Models")
        print(f"please run the script")
        print(f"train/reproduce_paper_results.sh {dataset_dir}")