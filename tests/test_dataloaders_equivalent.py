# System
import json
import numpy as np
from pathlib import Path
import shutil

import torch

from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
from dataloaders.brepnet_dataset_old import BRepNetDatasetOld
from pipeline.extract_brepnet_data_from_json import BRepNetJsonExtractor
import utils.data_utils as data_utils

from tests.test_base import TestBase
import unittest

class TestDataloadersEquivalent(TestBase):

    def equivalent_dataloaders_dir(self):
        return self.data_dir() / "equivalent_dataloaders"
        
    def input_feature_list(self):
        return self.equivalent_dataloaders_dir() / "original_feature_list.json"

    def create_old_dataset(self, dataset_file, dataset_dir):
        input_features = self.input_feature_list()
        opts = self.create_dummy_options(dataset_file, dataset_dir, input_features)
        return BRepNetDatasetOld(opts, "training_set")


    def create_new_dataset(self, dataset_file, dataset_dir):
        input_features = self.input_feature_list()
        opts = self.create_dummy_options(dataset_file, dataset_dir, input_features)     
        return BRepNetDataset(opts, "training_set")

    def check_for_batch(
            self, 
            old_dataset_file,
            batch_index,
            new_dataset_file,
            brep_indices
        ):
        self.remove_folder(self.working_dir())
        # Make sure the dataloaders are not working with 
        # cache data
        self.remove_cache_folder(self.equivalent_dataloaders_dir())

        # Create the old data loader
        data_dir = self.equivalent_dataloaders_dir()
        old_dataset = self.create_old_dataset(old_dataset_file, data_dir)

        # Process files into the new format and create a new data
        # loader to process them
        self.generate_npz_files_from_json(new_dataset_file, data_dir)
        new_dataset = self.create_new_dataset(new_dataset_file, self.working_dir())

        # Extract a batch from the old data loader
        old_batch = old_dataset[batch_index]
        
        # Extract solids from the new data loader
        new_brep_data = []
        for brep_index in  brep_indices:
            new_brep_data.append(new_dataset[brep_index])
        new_batch = brepnet_collate_fn(new_brep_data)
        self.check_batches_same(old_batch, new_batch)

    def check_batches_same(self, old_batch, new_batch):
        self.check_face_features(old_batch, new_batch)
        self.check_edge_features(old_batch, new_batch)
        self.check_coedge_features(old_batch, new_batch)
        self.check_face_kernel_tensor(old_batch, new_batch)
        self.check_edge_kernel_tensor(old_batch, new_batch)
        self.check_coedges_of_edges(old_batch, new_batch)
        self.check_coedges_of_small_faces(old_batch, new_batch)
        self.check_coedges_of_big_faces(old_batch, new_batch)
        self.check_labels(old_batch, new_batch)


    def check_face_features(self, old_batch, new_batch):
        old_face_features = old_batch["face_features"]
        new_face_features = new_batch["face_features"]

        # Now the face features will have different permutations
        # so we need to split the batch and then compare
        old_split_batch = old_batch["split_batch"]
        new_split_batch = new_batch["split_batch"]

        for old_split_solid, new_split_solid in zip(old_split_batch, new_split_batch):
            old_split_face_features = old_face_features[old_split_solid["face_indices"]]
            new_split_face_features = new_face_features[new_split_solid["face_indices"]]
            self.assertTrue(torch.allclose(old_split_face_features, new_split_face_features))


    def check_edge_features(self, old_batch, new_batch):
        old_edge_features = old_batch["edge_features"]
        new_edge_features = new_batch["edge_features"]
        self.assertTrue(torch.allclose(old_edge_features, new_edge_features))

    def check_coedge_features(self, old_batch, new_batch):
        old_coedge_features = old_batch["coedge_features"]
        new_coedge_features = new_batch["coedge_features"]
        self.assertTrue(torch.allclose(old_coedge_features, new_coedge_features))

    def check_face_kernel_tensor(self, old_batch, new_batch):
        old_face_index_to_original_face_index, old_face_index_to_solid_index = self.find_face_mapping(old_batch)
        new_face_index_to_original_face_index, new_face_index_to_solid_index = self.find_face_mapping(new_batch)
        old_face_kernel_tensor = old_batch["face_kernel_tensor"]
        new_face_kernel_tensor = new_batch["face_kernel_tensor"]
        old_original_face_indices = old_face_index_to_original_face_index[old_face_kernel_tensor]
        new_original_face_indices = new_face_index_to_original_face_index[new_face_kernel_tensor]
        self.assertTrue(np.all(old_original_face_indices == new_original_face_indices))

    def check_edge_kernel_tensor(self, old_batch, new_batch):
        old_edge_kernel_tensor = old_batch["edge_kernel_tensor"]
        new_edge_kernel_tensor = new_batch["edge_kernel_tensor"]
        self.assertTrue(torch.all(old_edge_kernel_tensor == new_edge_kernel_tensor))

    def check_coedges_of_edges(self, old_batch, new_batch):
        old_coedges_of_edges = old_batch["coedges_of_edges"]
        new_coedges_of_edges = new_batch["coedges_of_edges"]
        self.assertTrue(torch.all(old_coedges_of_edges == new_coedges_of_edges))

    def check_coedges_of_small_faces(self, old_batch, new_batch):
        old_coedges_of_small_faces = old_batch["coedges_of_small_faces"]
        new_coedges_of_small_faces = new_batch["coedges_of_small_faces"]

        old_split_batch = old_batch["split_batch"]
        new_split_batch = new_batch["split_batch"]

        for old_split_solid, new_split_solid in zip(old_split_batch, new_split_batch):
            old_all_face_indices = old_split_solid["face_indices"]
            new_all_face_indices = new_split_solid["face_indices"]
            for old_face_index, new_face_index in zip(old_all_face_indices, new_all_face_indices):
                if old_face_index >= old_coedges_of_small_faces.size(0):
                    # We should be in the big faces
                    self.assertGreaterEqual(new_face_index, new_coedges_of_small_faces.size(0))
                else:
                    old_split_coedges_of_small_face = old_coedges_of_small_faces[old_face_index]
                    new_split_coedges_of_small_face = new_coedges_of_small_faces[new_face_index]
                    self.check_multiplicity_of_indices_in_row(old_split_coedges_of_small_face, new_split_coedges_of_small_face)

    def check_multiplicity_of_indices_in_row(self, old, new):
        self.assertEqual(old.size(), new.size())
        old_multiplicity = self.build_multiplicity(old)
        new_multiplicity = self.build_multiplicity(new)
        self.assertEqual(len(old_multiplicity), len(new_multiplicity))
        for key in old_multiplicity:
            self.assertIn(key, new_multiplicity)
            self.assertEqual(old_multiplicity[key], new_multiplicity[key])

    def build_multiplicity(self, row):
        mult = {}
        for value in row:
            if not value in mult:
                mult[value.item()] = 0
            mult[value.item()] += 1
        return mult

    def check_coedges_of_big_faces(self, old_batch, new_batch):
        old_coedges_of_big_faces = old_batch["coedges_of_big_faces"]
        new_coedges_of_big_faces = new_batch["coedges_of_big_faces"]
        self.assertEqual(len(old_coedges_of_big_faces), len(new_coedges_of_big_faces))
        num_big_faces = len(old_coedges_of_big_faces)

        old_coedges_of_small_faces = old_batch["coedges_of_small_faces"]
        new_coedges_of_small_faces = new_batch["coedges_of_small_faces"]

        # Check the number of small faces are the same in both cases
        self.assertEqual(old_coedges_of_small_faces.size(0), new_coedges_of_small_faces.size(0))
        num_small_faces = old_coedges_of_small_faces.size(0)

        old_split_batch = old_batch["split_batch"]
        new_split_batch = new_batch["split_batch"]

        old_face_index_to_original_face_index, old_face_index_to_solid_index = self.find_face_mapping(old_batch)

        for old_split_solid in old_split_batch:
            old_all_face_indices = old_split_solid["face_indices"]
            for big_face_index in range(num_big_faces):
                old_face_index_in_batch = num_small_faces + big_face_index
                old_original_face_index = old_face_index_to_original_face_index[old_face_index_in_batch]
                old_original_solid_index = old_face_index_to_solid_index[old_face_index_in_batch]
                new_face_index_in_batch = \
                    new_split_batch[old_original_solid_index]["face_indices"][old_original_face_index]
                old_split_coedges_of_big_face = old_coedges_of_big_faces[big_face_index]
                new_split_coedges_of_big_face = new_coedges_of_big_faces[new_face_index_in_batch-num_small_faces]
                self.check_multiplicity_of_indices_in_row(
                    old_split_coedges_of_big_face, 
                    new_split_coedges_of_big_face
                )


    def check_labels(self, old_batch, new_batch):
        old_lables = old_batch["labels"]
        new_labels = new_batch["labels"]

        # Now the face features will have different permutations
        # so we need to split the batch and then compare
        old_split_batch = old_batch["split_batch"]
        new_split_batch = new_batch["split_batch"]

        for old_split_solid, new_split_solid in zip(old_split_batch, new_split_batch):
            old_split_lables = old_lables[old_split_solid["face_indices"]]
            new_split_labels = new_labels[new_split_solid["face_indices"]]
            self.assertTrue(torch.allclose(old_split_lables, new_split_labels))


    def find_face_mapping(self, batch):
        split_batch = batch["split_batch"]

        num_faces = batch["face_features"].size(0)
        face_index_to_original_face_index = np.zeros((num_faces), dtype=np.int64)
        face_index_to_solid_index = np.zeros((num_faces), dtype=np.int64)

        for solid_index, solid in enumerate(split_batch):
            face_indices = solid["face_indices"]
            for old_face_index, new_face_index in enumerate(face_indices):
                face_index_to_original_face_index[new_face_index] = old_face_index
                face_index_to_solid_index[new_face_index] = solid_index
        return face_index_to_original_face_index, face_index_to_solid_index

    
    def generate_npz_files_from_json(self, dataset_pathname, data_dir):
        # Load the dataset file
        dataset = data_utils.load_json_data(dataset_pathname)
        train_filestems = dataset["training_set"]

        working_dir = self.working_dir()
        feature_schema = data_utils.load_json_data(data_dir / "original_feature_list.json")
        for file_stem in train_filestems:
            npz_pathname = working_dir / (file_stem + ".npz")
            if not npz_pathname.exists():
                self.extract_brepnet_data(file_stem, data_dir, working_dir, feature_schema)

                # We should now have generated the npz file
                self.assertTrue(npz_pathname.exists())

            seg_file = working_dir / (file_stem + ".seg")
            if not seg_file.exists():
                label_file = data_dir / (file_stem + "_labels.json")
                self.make_labels_from_json(label_file, seg_file)

    def extract_brepnet_data(self, file_stem, data_dir, output_path, feature_schema):
        topology_file = data_dir / (file_stem + "_topology.json")
        topology = data_utils.load_json_data(topology_file)["topology"]
        features_pathname = data_dir / (file_stem + "_features.json")
        features = data_utils.load_json_data(features_pathname)["feature_data"]
        extractor = BRepNetJsonExtractor(topology, features, feature_schema)
        data = extractor.process()
        output_pathname = output_path / f"{file_stem}.npz"
        data_utils.save_npz_data_without_uvnet_features(output_pathname, data)


    def make_labels_from_json(self, label_file, output_file):
        labels = data_utils.load_json_data(label_file)
        num_faces = len(labels["face_labels"])
        self.assertGreater(num_faces, 0)
        labels_arr = np.zeros((num_faces), dtype=np.int64)

        for face_index, face_labels in enumerate(labels["face_labels"]):
            for label_index, label_obj in enumerate(face_labels["labels"]):
                label_value = label_obj["label_value"]
                if label_value > 0.5:
                    labels_arr[face_index] = label_index
        np.savetxt(output_file, labels_arr, fmt='%i', delimiter="\n")


    def test_dataloaders_equivalent(self):
        self.remove_folder(self.working_dir())
        data_dir = self.equivalent_dataloaders_dir()
        old_dataset_file = data_dir / "dummy_old_dataset_single_brep.json"
        new_dataset_file = data_dir / "dummy_new_dataset_single_brep.json"
        self.check_for_batch(
            old_dataset_file,
            0,
            new_dataset_file,
            [ 0 ]
        )

    def test_full_dataset(self):
        self.remove_folder(self.working_dir())
        data_dir = self.equivalent_dataloaders_dir()
        old_dataset_file = data_dir / "dummy_old_dataset_without_standardization.json"
        new_dataset_file = data_dir / "dummy_new_dataset_without_standardization.json"
        self.cross_check_datasets(old_dataset_file, new_dataset_file)

        old_dataset_file = data_dir / "dummy_old_dataset_with_standardization.json"
        new_dataset_file = data_dir / "dummy_new_dataset_with_standardization.json"
        self.cross_check_datasets(old_dataset_file, new_dataset_file)

    def cross_check_datasets(self, old_dataset_file, new_dataset_file):
        shutil.rmtree(self.working_dir(), ignore_errors=True)
        old_dataset = data_utils.load_json_data(old_dataset_file)
        new_dataset = data_utils.load_json_data(new_dataset_file)
        file_stem_to_index = {}
        for file_index, file_stem in enumerate(new_dataset["training_set"]):
            file_stem_to_index[file_stem] = file_index

        batches_tested = 0
        for batch_index, batch in enumerate(old_dataset["training_set"]["batches"]):
            indices_in_new_dataset = []
            batch_ok = True
            for file_stem in batch:
                if not file_stem in file_stem_to_index:
                    print(f"Warning -- File {file_stem} missing from new dataset")
                    batch_ok = False
                else:
                    file_index = file_stem_to_index[file_stem]
                    indices_in_new_dataset.append(file_index)

            # We test for every batch without missing files
            if batch_ok:
                self.check_for_batch(
                    old_dataset_file,
                    batch_index,
                    new_dataset_file,
                    indices_in_new_dataset
                )
                batches_tested += 1

        print(f"Successfully tested {batches_tested} batches")


if __name__ == '__main__':
    unittest.main()