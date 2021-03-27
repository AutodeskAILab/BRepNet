# BRepNet

## Input data format
The input data for the network is obtained from 3 files.   
- The topology file
- The features file
- Then labels file
For examples of each file format please see the `example_files` folder.   
The train/validation/test split is defined in a dataset file.  An example is shown in `example_files/example_dataset.json`

## Training the network

```
python train.py --dataset_file /path/to/dataset_file.json --dataset_dir /path/to/data_dir
```

## Testing the network

```
python test.py \
  --dataset_file /path/to/dataset_file.json \
  --dataset_dir /path/to/data_dir \
  --test_with_pretrained_model /path/to/pretrained_model.pt
```