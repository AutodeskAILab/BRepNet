
# System
import json
import numpy as np
from pathlib import Path

import torch

from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
from pipeline.extract_brepnet_data_from_step import BRepNetExtractor
import utils.data_utils as data_utils

from tests.test_base import TestBase
import unittest

class TestBRepNetDataset(TestBase):

    def create_dataset(self):
        self.remove_cache_folder(self.simple_solid_test_data_dir())
        self.remove_folder(self.working_dir())

        self.generate_npz_files()
        opts = self.create_dummy_options(
            self.simple_solid_dataset_file(),
            self.working_dir(),
            self.feature_list_file()
        )     
        return BRepNetDataset(opts, "train")

    def simple_solid_test_data_dir(self):
        return Path(__file__).parent / "test_data/simple_solids/"

    def simple_solid_dataset_file(self):
        return self.simple_solid_test_data_dir()/"simple_solid_dataset.json"

    def load_feature_schema(self):
        return data_utils.load_json_data(self.feature_list_file())

    def generate_npz_files(self):
        # Load the dataset file
        dataset = data_utils.load_json_data(self.simple_solid_dataset_file())
        train_filestems = dataset["train"]

        working_dir = self.working_dir()
        simple_solid_dir = self.simple_solid_test_data_dir()
        feature_schema = self.load_feature_schema()
        step_dir = self
        for file_stem in train_filestems:
            npz_pathname = working_dir / (file_stem + ".npz")
            if not npz_pathname.exists():
                step_pathname = simple_solid_dir / (file_stem + ".step")
                self.assertTrue(step_pathname.exists())
                brepnet_extractor = BRepNetExtractor(
                    step_pathname, 
                    working_dir, 
                    feature_schema
                )
                brepnet_extractor.process()

                # We should now have generated the npz file
                self.assertTrue(npz_pathname.exists())

            # Make dummy label files as well
            self.make_dummy_labels(npz_pathname, working_dir)


    def make_dummy_labels(self, npz_pathname, working_dir):
        data = data_utils.load_npz_data(npz_pathname)
        num_faces = data["face_features"].shape[0]
        dummy_labels = np.arange(num_faces, dtype=np.int64)

        label_pathname = npz_pathname.with_suffix(".seg")
        np.savetxt(label_pathname, dummy_labels, fmt='%i', delimiter="\n")

    def load_npz_and_labels(self, file_stem):
        working_dir = self.working_dir()
        npz_pathname = working_dir / (file_stem + ".npz")
        npz_data = data_utils.load_npz_data(npz_pathname)
        label_pathname = npz_pathname.with_suffix(".seg")
        labels = data_utils.load_labels(label_pathname)
        return npz_data, labels


    def test_full_dataloader(self):
        dataset_filestems = data_utils.load_json_data(self.simple_solid_dataset_file())
        train_filestems = dataset_filestems["train"]
        dataset = self.create_dataset()
        self.assertEqual(len(dataset), len(train_filestems))
        num_files = len(dataset)
        self.assertEqual(num_files, len(train_filestems))
        for file_stem, data in zip(train_filestems, dataset):
            npz_data, labels = self.load_npz_and_labels(file_stem)
            self.cross_check_data_for_solid(data, npz_data, labels)


    def cross_check_data_for_solid(self, data, npz_data, labels):
        self.cross_check_face_features(data, npz_data)
        self.cross_check_edge_features(data, npz_data)
        self.cross_check_coedge_features(data, npz_data)

        kernel = data_utils.load_json_data(self.kernel_file())
        self.cross_check_face_kernel_tensor(data, npz_data, kernel)
        self.cross_check_edge_kernel_tensor(data, npz_data, kernel)
        self.cross_check_coedge_kernel_tensor(data, npz_data, kernel)
        self.cross_check_coedges_of_edges_tensor(data, npz_data)
        self.cross_check_coedges_of_faces_tensors(data, npz_data)
        self.cross_check_labels(data, npz_data, labels)

    def cross_check_face_features(self, data, npz_data):
        face_features_from_dl = data["face_features"]
        face_features_from_npz = npz_data["face_features"]
        self.assertEqual(face_features_from_dl.size(0), face_features_from_npz.shape[0])
        self.assertEqual(face_features_from_dl.size(1), face_features_from_npz.shape[1])

        coedges_of_small_faces = data["coedges_of_small_faces"]
        coedges_of_big_faces = data["coedges_of_big_faces"]
        total_num_faces = coedges_of_small_faces.size(0) + len(coedges_of_big_faces)
        self.assertEqual(face_features_from_dl.size(0), total_num_faces)

        # For this test dataset we have defined the standardization 
        # to do nothing to make the face features easier to test.

        # The faces will however be in a different order to the one
        # used in the original.
        old_to_new_face_indices = data["old_to_new_face_indices"]
        num_faces = face_features_from_npz.shape[0]
        num_features = face_features_from_npz.shape[1]
        reordered_face_features_from_npz = torch.zeros(
            num_faces, 
            num_features, 
            dtype=face_features_from_dl.dtype
        )

        # Do the re-ordering the slow but easy to understand way
        for old_face_index in range(num_faces):
            for j in range(num_features):
                new_face_index = old_to_new_face_indices[old_face_index]
                value = face_features_from_npz[old_face_index, j]
                reordered_face_features_from_npz[new_face_index, j] = value

        self.assertTrue(torch.allclose(face_features_from_dl, reordered_face_features_from_npz))


    def cross_check_edge_features(self, data, npz_data):
        edge_features_from_dl = data["edge_features"]
        edge_features_from_npz = torch.from_numpy(npz_data["edge_features"]).float()

        # Check the features have the expected size
        self.assertEqual(edge_features_from_dl.size(0), edge_features_from_npz.size(0))
        self.assertEqual(edge_features_from_dl.size(1), edge_features_from_npz.size(1))

        # Cross check against the topology 
        coedges_of_edges = data["coedges_of_edges"]
        self.assertEqual(coedges_of_edges.size(0), edge_features_from_dl.size(0))

        # The edge features should now be comparable 
        self.assertTrue(torch.allclose(edge_features_from_dl, edge_features_from_npz))


    def cross_check_coedge_features(self, data, npz_data):
        coedge_features_from_dl = data["coedge_features"]
        coedge_features_from_npz = torch.from_numpy(npz_data["coedge_features"]).float()

        # Check the features have the expected size
        num_coedges = npz_data["coedge_to_next"].size
        self.assertEqual(coedge_features_from_dl.size(0), num_coedges)
        self.assertEqual(coedge_features_from_npz.shape[0], num_coedges)
        self.assertEqual(coedge_features_from_dl.size(1), coedge_features_from_npz.size(1))

        # The kernel tensors should have the same size
        self.assertEqual(data["face_kernel_tensor"].size(0), num_coedges)
        self.assertEqual(data["edge_kernel_tensor"].size(0), num_coedges)
        self.assertEqual(data["coedge_kernel_tensor"].size(0), num_coedges)

        self.assertTrue(torch.allclose(coedge_features_from_dl, coedge_features_from_npz))
        

    def execute_walk(self, n, p, m, e, f, instructions):
        """
        In this function we are going to evaluate the instructions in a 
        topological walk in the way that one would in a CAD system.

        For each element in the walk we follow the "pointers" stored
        in the arrays n, p, m, e and f.  These talk us from the current
        coedge to the next, previous or mating coedge or owner edge or face.

        Using numpy it is possible to execute all the lookups at once with 
        super fast array indexing operations.  That's how the dataloader works.
        In this function we are just checking the results, so we do things the 
        slow and reliable way.
        """
        # All these lists of indices should have the same size
        self.assertEqual(n.size, p.size)
        self.assertEqual(n.size, m.size)
        self.assertEqual(n.size, e.size)
        self.assertEqual(n.size, f.size)

        final_ent_index = torch.zeros(n.size, dtype=torch.int64)
        for i in range(n.size):
            # We start at coedge i
            entity_index = i

            # Then for each instruction in the list we jump 
            # to an adjacent coedge, edge or face
            for instruction in instructions:
                if instruction == "n":
                    entity_index = n[entity_index]
                elif instruction == "p":
                    entity_index = p[entity_index]
                elif instruction == "m":
                    entity_index = m[entity_index]
                elif instruction == "e":
                    entity_index = e[entity_index]
                elif instruction == "f":
                    entity_index = f[entity_index]
                else:
                    self.assertTrue(False) # Unknown instruction

            # Then we store the final index for the entity we end up sitting on
            final_ent_index[i] = entity_index
        return final_ent_index


    def find_inverse_permutation_slow(self, perm):
        """
        Find the inverse of a permutation for a numpy array
        the really slow way
        """
        if isinstance(perm, np.ndarray):
            inv = np.zeros(perm.size, perm.dtype)
            num_elements = perm.size
        else:
            self.assertTrue(isinstance(perm, torch.Tensor))
            inv = torch.zeros(perm.size(), dtype=torch.int64)
            num_elements = perm.size(0)
        check_set = set()
        for old_index, new_index in enumerate(perm):
            inv[new_index] = old_index
            # Check this is really a permutation
            self.assertNotIn(new_index, check_set)
            check_set.add(new_index)
            self.assertLess(new_index.item(),num_elements)
        self.assertEqual(len(check_set), num_elements)
        return inv


    def build_kernel_tensor(self, npz_data, walks):
        n = npz_data["coedge_to_next"]
        m = npz_data["coedge_to_mate"]
        e = npz_data["coedge_to_edge"]
        f = npz_data["coedge_to_face"]

        p = self.find_inverse_permutation_slow(n)

        num_coedges = n.size
        self.assertEqual(num_coedges, p.size)
        self.assertEqual(num_coedges, m.size)
        self.assertEqual(num_coedges, e.size)
        self.assertEqual(num_coedges, f.size)
        final_ents = []
        for walk in walks:
            ent_at_end_of_walk = self.execute_walk(n, p, m, e, f, walk) 
            final_ents.append(ent_at_end_of_walk)
        final_ents = torch.stack(final_ents)
        kernel_tensor = torch.transpose(final_ents, 0, 1)
        self.assertEqual(kernel_tensor.size(0), n.size)
        self.assertEqual(kernel_tensor.size(1), len(walks))
        return kernel_tensor


    def cross_check_face_kernel_tensor(self, data, npz_data, kernel):
        Kf_from_dl = data["face_kernel_tensor"]
        Kf_from_npz = self.build_kernel_tensor(npz_data, kernel["faces"])

        self.assertEqual(Kf_from_dl.size(0), Kf_from_npz.size(0))
        self.assertEqual(Kf_from_dl.size(1), Kf_from_npz.size(1))

        num_coedges = npz_data["coedge_to_next"].size
        num_faces_in_kernel = len(kernel["faces"])
        self.assertEqual(Kf_from_dl.size(0), num_coedges)
        self.assertEqual(Kf_from_dl.size(1), num_faces_in_kernel)

        # Now the faces array have been re-arranged so we
        # need to map the indices in the kernel to the new values
        old_to_new_face_indices = data["old_to_new_face_indices"]
        num_faces = npz_data["face_features"].shape[0]
        self.assertEqual(old_to_new_face_indices.size(0), num_faces)

        Kf_from_npz_mapped = torch.zeros(Kf_from_dl.shape, dtype=Kf_from_dl.dtype)
        for i in range(num_coedges):
            for j in range(num_faces_in_kernel):
                old_face_index = Kf_from_npz[i,j]
                new_face_index = old_to_new_face_indices[old_face_index]
                Kf_from_npz_mapped[i,j] = new_face_index
        self.assertTrue(torch.all(Kf_from_dl == Kf_from_npz_mapped))


    def cross_check_edge_kernel_tensor(self, data, npz_data, kernel):
        num_coedges = data["coedge_features"].shape[0]
        num_edges_in_kernel = len(kernel["edges"])

        Ke_from_dl = data["edge_kernel_tensor"]
        self.assertEqual(Ke_from_dl.size(0), num_coedges)
        self.assertEqual(Ke_from_dl.size(1), num_edges_in_kernel)

        Ke_from_npz = self.build_kernel_tensor(npz_data, kernel["edges"])
        self.assertEqual(Ke_from_npz.size(0), num_coedges)
        self.assertEqual(Ke_from_dl.shape, Ke_from_npz.shape)
        self.assertTrue(torch.all(Ke_from_dl == Ke_from_npz))


    def cross_check_coedge_kernel_tensor(self, data, npz_data, kernel):
        num_coedges = data["coedge_features"].shape[0]
        num_coedges_in_kernel = len(kernel["coedges"])

        Kc_from_dl = data["coedge_kernel_tensor"]
        self.assertEqual(num_coedges, Kc_from_dl.size(0))
        self.assertEqual(num_coedges_in_kernel, Kc_from_dl.size(1))

        Kc_from_npz = self.build_kernel_tensor(npz_data, kernel["coedges"])
        self.assertEqual(num_coedges, Kc_from_npz.size(0))
        self.assertEqual(Kc_from_dl.shape, Kc_from_npz.shape)
        self.assertTrue(torch.all(Kc_from_dl == Kc_from_npz))


    def cross_check_coedges_of_edges_tensor(self, data, npz_data):
        """
        We can check for consistency between the mating coedges and the
        edge information.
        """
        coedges_of_edges = data["coedges_of_edges"]
        m = npz_data["coedge_to_mate"]
        for edge_index in range(coedges_of_edges.size(0)):
            left_coedge = coedges_of_edges[edge_index, 0]
            right_coedge = coedges_of_edges[edge_index, 1]

            left_mate = m[left_coedge]
            right_mate = m[right_coedge]

            self.assertEqual(left_coedge, right_mate)
            self.assertEqual(right_coedge, left_mate)


    def cross_check_coedges_of_faces_tensors(self, data, npz_data):
        """
        We can cross check the in two ways

        - For every coedge, we know the index of it's face from array f.
          We can then check that the coedge is in the appropriate array once 
          and only once.   We can also check the next coedge in the loop is 
          on the same face.

        - For every face we can look at the coedges in the face and check that
          the face index agrees with the coedge index  
        """
        # Get the pointers to the next coedges in the loop.  These must lie on the
        # same face as the current coedge
        n = npz_data["coedge_to_next"]
        num_coedges = n.size

        # Get the pointers from coedges to their parent faces
        f = npz_data["coedge_to_face"]

        # The face array gets rearranged.  This table maps from the old
        # face index to the new face index.  We find the inverse of this 
        # mapping as well
        old_to_new_face_indices = data["old_to_new_face_indices"]
        new_to_old_face_indices = self.find_inverse_permutation_slow(old_to_new_face_indices)

        # Now the tensors used for the face pooling are split into two bits
        # We have a threshhold for the number of coedges which can be around a face
        # and for us to still consider this as "small"
        max_coedges_per_face = 30 
        # For faces with a small number of coedges we have the tensor
        # Cf.size() = (num_small_faces, max_coedges_per_face).  The value are padded
        # with the value num_coedges and an extra row of zeros gets concatenated 
        # to the coedge array when Cf is used
        Cf = data["coedges_of_small_faces"]
        self.assertEqual(Cf.size(1), max_coedges_per_face)

        # For faces which have more than max_coedges_per_face, the coedge indices are
        # in individual tensors
        Csf = data["coedges_of_big_faces"]
        for coedges in Csf:
            self.assertGreaterEqual(coedges.size(0), max_coedges_per_face)

        num_small_faces = Cf.size(0)
        total_num_faces = num_small_faces + len(Csf)
        self.assertEqual(old_to_new_face_indices.size(0), total_num_faces)

        for coedge_index, face_index in enumerate(f):
            next_coedge = n[coedge_index]
            self.assertEqual(face_index, f[next_coedge])
            self.check_coedge_is_in_correct_face(
                coedge_index, 
                face_index, 
                Cf, 
                Csf, 
                old_to_new_face_indices
            )
            self.check_coedge_is_in_correct_face(
                next_coedge, 
                face_index, 
                Cf, 
                Csf, 
                old_to_new_face_indices
            )

        # Now we try to perform the same check the other way around
        # i.e. we look at the coedges for each face and check they
        # map to that face
        for small_face_index in range(num_small_faces):
            for j in range(max_coedges_per_face):
                coedge_index = Cf[small_face_index, j]
                if coedge_index == num_coedges:
                    # This is the padding
                    continue
                old_face_index = f[coedge_index]
                new_face_index = old_to_new_face_indices[old_face_index]
                self.assertEqual(new_face_index, small_face_index)

        # This is the loop over the faces which have more than 
        # max_coedges_per_face coedges around the face
        for big_face_index, coedges in enumerate(Csf):
            new_face_index = big_face_index + num_small_faces
            num_coedges_on_face = coedges.size(0)
            for i in range(num_coedges_on_face):
                coedge_index = coedges[i]
                old_face_index = f[coedge_index]
                new_face_index_from_coedge = old_to_new_face_indices[old_face_index]
                self.assertEqual(new_face_index, new_face_index_from_coedge)


    def check_coedge_is_in_correct_face(
            self, 
            coedge_index, 
            old_face_index, 
            Cf, 
            Csf, 
            old_to_new_face_index
        ):
        new_face_index = old_to_new_face_index[old_face_index]
        num_small_faces = Cf.size(0)

        if new_face_index >= num_small_faces:
            # This data will be in the single faces array Csf
            single_face_index = new_face_index-num_small_faces
            self.assertLess(single_face_index, len(Csf))
            num_occurrences = (Csf[single_face_index] == coedge_index).sum()
            self.assertEqual(num_occurrences, 1)
        else:
            # The data is in a row of Cf
            num_occurrences = (Cf[new_face_index] == coedge_index).sum()
            self.assertEqual(num_occurrences, 1)


    def cross_check_labels(self, data, npz_data, labels):
        num_faces = npz_data["face_features"].shape[0]
        self.assertEqual(labels.size, num_faces)

        labels_from_dl = data["labels"]
        self.assertEqual(labels_from_dl.size(0), num_faces)
        old_to_new_face_indices = data["old_to_new_face_indices"]

        for i in range(num_faces):
            label_from_labels = labels[i]
            new_face_index = old_to_new_face_indices[i]
            label_from_dl = labels_from_dl[new_face_index]
            self.assertEqual(label_from_labels, label_from_dl)


    def test_build_coedges_of_edges_tensor(self):
        """
        Test for the BRepNetDataset.build_coedges_of_edges_tensor() function
        """
        dataset = self.create_dataset()
        coedge_to_edge = np.array([0,0, 1,1, 2,2, 3,3])
        dummy_edge_features = np.zeros(4)
        body_data = {
            "coedge_to_edge": coedge_to_edge,
            "edge_features": dummy_edge_features
        }
        coedges_of_edges = dataset.build_coedges_of_edges_tensor(body_data)

        expected_coedges_of_edges = torch.tensor(
            [[0, 1],
             [2, 3],
             [4, 5],
             [6, 7]], 
            dtype=torch.int64
        )
        self.assertTrue(torch.all(coedges_of_edges == expected_coedges_of_edges))


    def test_build_coedges_of_faces_tensor(self):
        """
        Test for the BRepNetDataset.build_coedges_of_faces_tensor() function
        """
        dataset = self.create_dataset()
        #                          0 1 2 3  4 5  6 7 8 9 10 11   12  13
        coedge_to_face = np.array([0,0,0,0, 1,1, 2,2,2,2,2, 2,   3,  3])
        dummy_face_features = np.zeros(4)
        body_data = {
            "coedge_to_face": coedge_to_face,
            "face_features": dummy_face_features
        }
        
        max_coedges_per_face = 3
        Cf, Csf, new_to_old_face_index = dataset.build_coedges_of_faces_tensor(
            body_data, 
            max_coedges_per_face         
        )


        # We want Cf.size() = [ num_small_faces x max_coedges ]
        expected_num_small_faces = 2
        num_coedges = coedge_to_face.size
        expected_Cf = torch.tensor(
            [[4, 5, num_coedges],
             [12, 13, num_coedges]]
        )
        self.assertTrue(torch.all(Cf == expected_Cf))

        # Then the array of tensors 
        # Csf = [
        #    Csf.size() = [ num_coedges_in_face_1 ],
        #    ...
        #]
        expected_Csf = [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([6, 7, 8, 9, 10, 11])
        ]
        self.assertEqual(len(expected_Csf), len(Csf))
        for t1, t2 in zip(Csf, expected_Csf):
            self.assertTrue(torch.all(t1 == t2))


    def test_find_face_permutation(self):
        """
        Test for the BRepNetDataset.find_face_permutation() function.
        """
        dataset = self.create_dataset()
        coedge_to_face = np.array([0,0,0, 1,1,1, 2,2,2, 3,3,3])
        num_dummy_faces = 4
        dummy_face_features = np.zeros((num_dummy_faces, 7))
        body_data = {
            "coedge_to_face": coedge_to_face,
            "face_features": dummy_face_features
        }
        max_coedges_per_face = 10
        
        fp, small_faces_size = dataset.find_face_permutation(body_data, max_coedges_per_face)
        identity = np.arange(num_dummy_faces, dtype=np.int64)
        self.assertTrue(np.all(fp == identity))

        # Try another example where we rearrange the array
        coedge_to_face = np.array([0,0,0,0,0, 1,1,1, 2,2,2])
        num_dummy_faces = 3
        dummy_face_features = np.zeros((num_dummy_faces, 7))
        body_data = {
            "coedge_to_face": coedge_to_face,
            "face_features": dummy_face_features
        }
        max_coedges_per_face = 4
        fp, small_faces_size = dataset.find_face_permutation(body_data, max_coedges_per_face)
        expected_permutation = np.array([1, 2, 0])
        self.assertTrue(np.all(fp == expected_permutation))

    def test_standardize_features(self):
        stats = [
            {
                "mean": 1.0,
                "standard_deviation": 2.0
            },
            {
                "mean": 2.0,
                "standard_deviation": 1.0
            },
            {
                "mean": 3.0,
                "standard_deviation": 5.0
            }
        ]
        num_features = len(stats)
        num_ents = 10
        feature_tensor = torch.rand(num_ents, num_features)
        dataset = self.create_dataset()
        std_features = dataset.standardize_features(feature_tensor, stats)
                
        test_tensor = torch.zeros(num_ents, num_features)
        for i in range(num_ents):
            for j in range(num_features):
                value = (feature_tensor[i,j] - stats[j]["mean"])/stats[j]["standard_deviation"]
                test_tensor[i,j] = value
        eps = 1e-7
        self.assertTrue(torch.allclose(std_features, test_tensor, eps))
        


    def test_find_inverse_permutation_np(self):
        perm = np.random.permutation(100)
        dataset = self.create_dataset()
        inv_perm = dataset.find_inverse_permutation(perm)

        for mapped_from, mapped_to in enumerate(perm):
            self.assertEqual(inv_perm[mapped_to], mapped_from)


    def test_find_inverse_permutation_torch(self):
        perm = torch.randperm(100)
        dataset = self.create_dataset()
        inv_perm = dataset.find_inverse_permutation(perm)

        for mapped_from, mapped_to in enumerate(perm):
            self.assertEqual(inv_perm[mapped_to], mapped_from)

    def build_face_mapping(self, data):
        mapping = []
        face_offset = 0

        # First loop over the small faces (faces with a small number of coedges)
        for solid_data in data:
            face_mapping = []
            num_small_faces = solid_data["coedges_of_small_faces"].size(0)
            for face_index in range(num_small_faces):
                face_mapping.append(face_index + face_offset) 
            face_offset += num_small_faces
            mapping.append(face_mapping)

        # Now loop over the big faces (faces with a large number of coedges)
        for solid_index, solid_data in enumerate(data):
            num_big_faces = len(solid_data["coedges_of_small_faces"])
            for face_index in range(num_big_faces):
                mapping[solid_index].append(face_index + face_offset) 
            face_offset += num_big_faces

        return mapping

    def build_edge_mapping(self, data):
        mapping = []
        edge_offset = 0
        for solid_data in data:
            edge_mapping = []
            num_edges = solid_data["edge_features"].size(0)
            for edge_index in range(num_edges):
                edge_mapping.append(edge_index + edge_offset) 
            edge_offset += num_edges
            mapping.append(edge_mapping)
        return mapping

    def build_coedge_mapping(self, data):
        mapping = []
        coedge_offset = 0
        for solid_data in data:
            coedge_mapping = []
            num_coedges = solid_data["coedge_features"].size(0)
            for coedge_index in range(num_coedges):
                coedge_mapping.append(coedge_index + coedge_offset) 
            coedge_offset += num_coedges
            mapping.append(coedge_mapping)
        return mapping

    def check_collate(self, data):
        batch = brepnet_collate_fn(data)
        face_mapping = self.build_face_mapping(data)
        edge_mapping = self.build_edge_mapping(data)
        coedge_mapping = self.build_coedge_mapping(data)

        batch_face_features = batch["face_features"]
        batch_edge_features = batch["edge_features"]
        batch_coedge_features = batch["coedge_features"]
        batch_face_kernel_tensor = batch["face_kernel_tensor"]
        batch_edge_kernel_tensor = batch["edge_kernel_tensor"]
        batch_coedge_kernel_tensor = batch["coedge_kernel_tensor"]
        batch_coedges_of_edges = batch["coedges_of_edges"]
        batch_coedges_of_small_faces = batch["coedges_of_small_faces"]
        batch_coedges_of_big_faces = batch["coedges_of_big_faces"]
        batch_labels = batch["labels"]
        file_stems = batch["file_stems"]
        split_batch = batch["split_batch"]

        self.assertEqual(len(file_stems), len(data))
        self.assertEqual(len(split_batch), len(data))

        # This should be the value for the padding in the 
        # batch_coedges_of_small_faces tensor
        batch_num_coedges = batch_coedge_features.size(0)

        # Loop over the solid data
        for solid_index, solid_data in enumerate(data):
            # Get at the tensors
            face_features = solid_data["face_features"]
            edge_features = solid_data["edge_features"]
            coedge_features = solid_data["coedge_features"]
            face_kernel_tensor = solid_data["face_kernel_tensor"]
            edge_kernel_tensor = solid_data["edge_kernel_tensor"]
            coedge_kernel_tensor = solid_data["coedge_kernel_tensor"]
            coedges_of_edges = solid_data["coedges_of_edges"]
            coedges_of_small_faces = solid_data["coedges_of_small_faces"]
            coedges_of_big_faces = solid_data["coedges_of_big_faces"]
            labels = solid_data["labels"]
            old_to_new_face_indices = solid_data["old_to_new_face_indices"]

            # Find the number of faces, edges and coedges
            num_faces = face_features.size(0)
            num_edges = edge_features.size(0)
            num_coedges = coedge_features.size(0)

            # Loop over the faces in the solid
            for solid_face_index in range(num_faces):
                batch_face_index = face_mapping[solid_index][solid_face_index]
                self.assertTrue(
                    torch.allclose(face_features[solid_face_index], batch_face_features[batch_face_index])
                )

                solid_num_small_faces = coedges_of_small_faces.size(0)
                batch_num_small_faces = batch_coedges_of_small_faces.size(0)
                if solid_face_index < solid_num_small_faces:
                    solid_coedges_of_small_face = coedges_of_small_faces[solid_face_index]
                    batch_coedges_of_small_face = batch_coedges_of_small_faces[batch_face_index]
                    for solid_coedge_index, batch_coedge_index in zip(solid_coedges_of_small_face,batch_coedges_of_small_face):
                        self.assertLessEqual(solid_coedge_index, num_coedges)
                        if solid_coedge_index == num_coedges:
                            # This is in the padding
                            self.assertEqual(batch_coedge_index, batch_num_coedges)
                            continue
                        mapped_coedge_index = coedge_mapping[solid_index][solid_coedge_index]
                        self.assertEqual(batch_coedge_index, mapped_coedge_index)
                        self.assertTrue(
                            torch.allclose(coedge_features[solid_coedge_index], batch_coedge_features[batch_coedge_index])
                        )
                else:
                    solid_coedges_of_big_face = coedges_of_big_faces[solid_face_index-solid_num_small_faces]
                    batch_coedges_of_big_face = batch_coedges_of_big_faces[batch_face_index-batch_num_small_faces]
                    for solid_coedge_index, batch_coedge_index in zip(solid_coedges_of_big_face,batch_coedges_of_big_face):
                        mapped_coedge_index = coedge_mapping[solid_index][solid_coedge_index]
                        self.assertEqual(batch_coedge_index, mapped_coedge_index)
                        self.assertTrue(
                            torch.allclose(coedge_features[solid_coedge_index], batch_coedge_features[batch_coedge_index])
                        )


            # Checks for things mapped by edges
            for solid_edge_index in range(num_edges):
                batch_edge_index = edge_mapping[solid_index][solid_edge_index]
                self.assertTrue(
                    torch.allclose(edge_features[solid_edge_index], batch_edge_features[batch_edge_index])
                )

                solid_coedges_of_edge = coedges_of_edges[solid_edge_index]
                batch_coedges_of_edge = batch_coedges_of_edges[batch_edge_index]
                for solid_coedge_index, batch_coedge_index in zip(solid_coedges_of_edge,batch_coedges_of_edge):
                    # Check based on the predicted mapping
                    mapped_coedge_index = coedge_mapping[solid_index][solid_coedge_index]
                    self.assertEqual(batch_coedge_index, mapped_coedge_index)

                    # Check the coedge features map
                    self.assertTrue(
                        torch.allclose(coedge_features[solid_coedge_index], batch_coedge_features[batch_coedge_index])
                    )


            
            # Checks for things which are mapped from coedges
            self.assertEqual(num_coedges, face_kernel_tensor.size(0))
            self.assertEqual(num_coedges, edge_kernel_tensor.size(0))
            self.assertEqual(num_coedges, coedge_kernel_tensor.size(0))
            for solid_coedge_index in range(num_coedges):
                # Check the coedge indices are mapped correctly so we find
                # the correct coedge feature 
                batch_coedge_index = coedge_mapping[solid_index][solid_coedge_index]
                self.assertTrue(
                    torch.allclose(coedge_features[solid_coedge_index], batch_coedge_features[batch_coedge_index])
                )

                # Check the face indices in the face kernel tensor get mapped 
                # correctly
                solid_face_indices = face_kernel_tensor[solid_coedge_index]
                batch_face_indices = batch_face_kernel_tensor[batch_coedge_index]
                for solid_face_index, batch_face_index in zip(solid_face_indices, batch_face_indices):
                    # Test based on the expected face mapping
                    mapped_face_index = face_mapping[solid_index][solid_face_index]
                    self.assertEqual(mapped_face_index, batch_face_index)

                    # So check we get the expected face features 
                    self.assertTrue(
                        torch.allclose(face_features[solid_face_index], batch_face_features[batch_face_index])
                    )

                    # Check we get the expected segment indices (labels)
                    self.assertTrue(
                        torch.allclose(labels[solid_face_index], batch_labels[batch_face_index])
                    )

                # Check the edge indices in the edge kernel tensor get mapped
                # correctly
                solid_edge_indices = edge_kernel_tensor[solid_coedge_index]
                batch_edge_indices = batch_edge_kernel_tensor[batch_coedge_index]
                for solid_edge_index, batch_edge_index in zip(solid_edge_indices, batch_edge_indices):
                    # Check the index based on the expected mapping
                    mapped_edge_index = edge_mapping[solid_index][solid_edge_index]
                    self.assertEqual(mapped_edge_index, batch_edge_index)
                                        
                    # Check we get the expected edge features 
                    self.assertTrue(
                        torch.allclose(edge_features[solid_edge_index], batch_edge_features[batch_edge_index])
                    )

                # Check the indices in the coedge kernel tensor get mapped 
                # correctly
                solid_coedge_indices = coedge_kernel_tensor[solid_coedge_index]
                batch_coedge_indices = batch_coedge_kernel_tensor[batch_coedge_index]
                for solid_coedge_index, batch_coedge_index in zip(solid_coedge_indices, batch_coedge_indices):
                    # Check based on the expected mapping
                    mapped_coedge_index = coedge_mapping[solid_index][solid_coedge_index]
                    self.assertEqual(mapped_coedge_index, batch_coedge_index)

                    # Check based on the coedge features
                    self.assertTrue(
                        torch.allclose(coedge_features[solid_coedge_index], batch_coedge_features[batch_coedge_index])
                    )

        # Now check we can unpack the solid and get back to the original 
        # lables
        solid_split_batch = split_batch[solid_index]
        new_to_old_face_indices = solid_split_batch["face_indices"]
        labels_for_old_face_order = batch_labels[new_to_old_face_indices]
        labels_for_new_face_order = old_to_new_face_indices[labels_for_old_face_order]
        self.assertTrue(torch.all(labels_for_new_face_order == labels))

        # Now we want to reload the original npz data and check that the 
        # batch splitter will really let us get back to the original 
        # data in the npz files
        loaded_data, loaded_labels = self.load_npz_and_labels(file_stems[solid_index])
        self.assertTrue(torch.all(labels_for_old_face_order == torch.from_numpy(loaded_labels)))

        loaded_face_features = torch.from_numpy(loaded_data["face_features"]).float()
        loaded_edge_features = torch.from_numpy(loaded_data["edge_features"]).float()
        loaded_coedge_features = torch.from_numpy(loaded_data["coedge_features"]).float()
        split_face_features = batch_face_features[new_to_old_face_indices]
        split_edge_features = batch_edge_features[solid_split_batch["edge_indices"]]
        split_coedge_features = batch_coedge_features[solid_split_batch["coedge_indices"]]
        self.assertTrue(torch.allclose(loaded_face_features, split_face_features))
        self.assertTrue(torch.allclose(loaded_edge_features, split_edge_features))
        self.assertTrue(torch.allclose(loaded_coedge_features, split_coedge_features))



    def test_brepnet_collate_fn(self):
        dataset = self.create_dataset()
        data = [
            dataset[1],
            dataset[2]
        ]
        self.check_collate(data)
        data = [
            dataset[0],
            dataset[1],
            dataset[0],
            dataset[2],
            dataset[0],
            dataset[3],
            dataset[0],
            dataset[4],
            dataset[0],
            dataset[5],
            dataset[0],
            dataset[6]
        ]


if __name__ == '__main__':
    unittest.main()