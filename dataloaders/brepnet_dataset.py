"""
The standard dataloader for BRepNet.

For each BRep model the following tensors are created.

    Xf - The face features

        A tensor of size [ num_faces x num_face_features ]

         These correspond to the feature data extracted from faces.  
         See BRepNetExtractor.extract_face_features_from_body() in
         pipeline/extract_brepnet_data_from_step.py for details of how the 
         values are computed.  

         The values will be standardized based on information from the 
         dataset file.

         The faces are not in the same order as they are in Open Cascade.
         For a batch of data you must use 
         
           Xf[split_batch[solid_index]["face_indices"]] 
           
         to restore these to their original order. 

    Xe - The edge features

        A tensor of size [ num_edges x num_edge_features ]

         See BRepNetExtractor.extract_edge_features_from_body() in
         pipeline/extract_brepnet_data_from_step.py for details of how the 
         values are computed.  

    Xc - The coedge features

         A tensor of size [ num_coedges x num_coedge_features ]

         See BRepNetExtractor.extract_coedge_features_from_body() in
         pipeline/extract_brepnet_data_from_step.py for details of how the 
         values are computed.  

    Kf - The face kernel tensor

         This is an index tensor of size [ num_coedges x num_faces_in_kernel ]

         For every coedge in the model the kernel will include some number of
         nearby faces whos hidden states will be combined in the convolution.

         This tensor contains the indices of these faces as they are in the
         Xf array.  

    Ke - The edge kernel tensor

         This is an index tensor of size [ num_coedges x num_edges_in_kernel ]

         For every coedge in the model the kernel will include some number of
         nearby edges whos hidden states will be combined in the convolution.

         This tensor contains the indices of these edges. 

    Kc - The coedge kernel tensor

         This is an index tensor of size [ num_coedges x num_coedges_in_kernel ]

         For every coedge in the model the kernel will include some number of
         nearby coedges whos hidden states will be combined in the convolution.

         This tensor contains the indices of these coedges. 

    Ce - The coedges of edges tensor

         This is an index tensor of size [ num_edges x 2]

         For every edge in the model, this tensor contains the indices
         of its two child coedges.

         The tensor is required to perform pooling of coedge hidden
         states onto the parent edges

    Cf - The coedges of "small" faces tensor

         There can be any arbitrary number of coedges around a single face.
         When we perform pooling of the coedge hidden states into faces we
         need to know which coedges are in the loops around each face.

         This information is split into two sets of tensors.  For faces with 
         less than max_coedges_per_face coedges on each face, the indices are
         written into the Cf index tensor with size

         [ num_small_faces x max_coedges_per_face ]

         A special "padding" index which is equal to the number of coedges in 
         the model is used for faces which have less than max_coedges_per_face
         coedges around the face.

         The faces in the model get re-ordered so that the faces with less
         than max_coedges_per_face coedges per face come before
         the other faces.

    Csf - The coedges of "large" faces
         
         For faces which have more than max_coedges_per_face around them
         the indices are stored in an array
          
            Csf = [
                Csf.size() = [ num_coedges_in_face_1 ],
                Csf.size() = [ num_coedges_in_face_2 ],
                ...
            ]

        In models/brepnet.py find_max_feature_vectors_for_each_face()
        the Cf and Csf index tensors get used for the max pooling operation.

        Notice that the re-ordering of the faces in Xf and Kf allows the new
        hidden states to be efficiently computed by concatenating tensors.

    labels - These are the segment indices for each face

       Notice that these are also re-ordered as described above. 

       Use labels[split_batch[solid_index]["face_indices"]] to extract
       the labels in the order of the faces in Open Cascade

    file_stem - This tells you the stem of the filename of each solid in the 
                batch


Splitting batches

BRepNet processes multiple solids in batches.  brepnet_collate_fn()
does the job of combining multiple solids into batches and keeps track
of how the faces, edges and coedges of each solid were combined into the
tensors for the batch.  This information is contained in the split_batch
array.

For each solid in the batch, split_batch contains the indices in the batch 
which must be accessed to understand the logits for faces, edges and coedges



split_batch = [
    {
        "face_indices":  [ num_faces_for_solid_1],
        "edge_indices": [ num_edges_for_solid_1],
        "coedge_indices" [ num_coedges_for_solid_1]:
    },
    ...
]

Given some logits for the batch they can be decoded like this

face_logits_for_solid_1 = face_logits_for_batch[split_batch[1]["face_indices"]]

"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import math
import numpy as np
import hashlib
import copy
import pickle

import utils.data_utils as data_utils

class BRepNetDataset(Dataset):
    """
    Dataset which loads the processed step data generated by
    pipeline/extract_brepnet_data_from_step.py
    """

    def __init__(self, opts, train_val_or_test):
        super(BRepNetDataset, self).__init__()
        self.opts = opts

        # Load the topological walks in to be used in the kernel
        self.kernel = data_utils.load_json_data(self.opts.kernel)

        # Load the list of input features to be used
        self.feature_lists = data_utils.load_json_data(self.opts.input_features)
        
        # The dataset file holds the full list of batches and the
        # feature normalization information
        dataset_info = data_utils.load_json_data(self.opts.dataset_file)

        self.bodies = dataset_info[train_val_or_test]
        self.feature_standardization = dataset_info["feature_standardization"]
        self.dataset_dir = Path(self.opts.dataset_dir)
        self.label_dir = self.find_label_dir(opts, train_val_or_test)
        self.cache_dir = self.create_cache_dir(self.dataset_dir)


    def __len__(self):
        """
        Get the number of bodies in the dataset
        """
        return len(self.bodies)


    def __getitem__(self, idx):
        """
        Load the data for a body.  Use a cache if its there.  
        Cache the binary data if not.
        """ 
        assert idx < len(self.bodies)
        cache_pathname = self.get_cache_pathname(idx)
        if cache_pathname.exists():
            body_data = self.load_body_from_cache(cache_pathname)
        else:
            body_data = self.load_and_cache_body(idx, cache_pathname)
        return body_data


    def hash_strings_in_list(self, string_list):
        """
        Create a hash of the strings in the list
        """
        single_str = str().join(string_list)
        single_str = single_str.encode('ascii', 'replace')

        return hashlib.sha224(single_str).hexdigest()


    def hash_data_for_body(self, body_filestem):
        """
        We want to be sure that the cache is correctly built given
        all the hyper-parameters of the network.
        Here we make a hash of these hyper-parameters and 
        use this as the pathname for the cache
        """
        string_list = [ body_filestem ]
        for ent in self.kernel:
            for walk in self.kernel[ent]:
                string_list.append(walk)
        for ent in self.feature_lists.values():
            feature_list = list(ent)
            sorted_feature_list = sorted(feature_list)
            for feature in sorted_feature_list:
                string_list.append(feature)
        if self.label_dir is not None:
            string_list.append("with_labels")
        return self.hash_strings_in_list(string_list)


    def get_cache_pathname(self, idx):
        """
        Create a pathname for the cache file for the batch with idx
        """
        body_filestem = self.bodies[idx]
        hstring = self.hash_data_for_body(body_filestem)
        return (self.cache_dir/hstring).with_suffix(".p")

    
    def cache_body(self, cache_pathname, data):
        """
        Save the cache data for the body
        """
        torch.save(data, cache_pathname)


    def load_body_from_cache(self, cache_pathname):
        """
        Load cache data for the body with the given pathname
        """
        return torch.load(cache_pathname)


    def load_and_cache_body(self, idx, cache_pathname):
        """
        Load the body with idx from the
        STEP data, then save a cache of the 
        binary tensors
        """
        body_data = self.load_body(idx)
        self.cache_body(cache_pathname, body_data)
        return body_data


    def create_cache_dir(self, dataset_dir):
        cache_dir = dataset_dir / "cache"
        if not cache_dir.exists():
            cache_dir.mkdir()
        assert cache_dir.exists(), "Check we were able to create the cache dir"
        return cache_dir


    def find_label_dir(self, opts, train_val_or_test):
        """
        Try to locate the dir where the labels are stored if this is not
        given explicitely 
        """
        if opts.label_dir:
            return Path(opts.label_dir)

        assert len(self.bodies) > 0
        first_file_stem = self.bodies[0]
        dataset_dir = Path(opts.dataset_dir)

        # Look for the seg files in the same folder as the 
        # step files.  This assumes the intermediate files
        # were created in a subfolder of the dataset folder
        # as is done by the quickstart script
        cadidate_label_file = dataset_dir.parent / (first_file_stem + ".seg")
        if cadidate_label_file.exists():
            print(f"Using labels from {dataset_dir}")
            return dataset_dir.parent

        # When using the Fusion Gallery segmentation dataset from the  
        # quickstart script, the seg files in the path 
        # s2.0.0/breps/seg and the dataset folder will be
        # s2.0.0/processed
        seg_dir = dataset_dir.parent / "breps/seg"
        cadidate_label_file = seg_dir / (first_file_stem + ".seg")
        if cadidate_label_file.exists():
            print(f"Using labels from {seg_dir}")
            return seg_dir

        # When using the model to evaluate on data without labels
        # the test set if used.  Hence if we are building this
        # datalaoder for the test set then we allow the labels to be
        # None without asserting
        if train_val_or_test == "test_set":
            print(" ")
            print("Warning!! - No labels are provided.  This can happen when you")
            print("are evaluating with a pre-trained model on an unlabelled dataset.")
            print("Please disregard and accuracy and IoU values which get logged.")
            print(" ")
            return None

        print(" ")
        print("Error!  Failed to find any label files.")
        print("These files should have the extension .seg and contain the")
        print("segment index for each face in each brep")
        print(" ")
        print("You can use the --label_dir option to point the model at the appropriate folder")
        print(" ")
        assert False, "Failed to locate labels (seg files)"


    def load_body(self, idx):
        """
        Load the data for a body.  
        """
        assert idx < len(self.bodies)
        file_stem = self.bodies[idx]
        npz_pathname = self.dataset_dir / (file_stem + ".npz")
        body_data = data_utils.load_npz_data(npz_pathname)
        Xf, Xe, Xc = self.build_input_feature_tensors(body_data)
        Kf, Ke, Kc = self.build_kernel_tensors(body_data)

        # Gf is the face point grids tensor in the order
        # the faces appear in the solid
        #
        # Gc is the coedge point grids in the order the 
        # coedges appear in the solid.  The points
        # in the grid are ordered based on the topological
        # direction of the coedge
        #
        # lcs is the local coordinate systems for each coedge
        Gf, Gc, lcs = self.build_point_grids(body_data)

        # We need to rearrange the order of the faces so that
        # faces with more than max_coedges_per_face are at the 
        # end of the array
        max_coedges_per_face = 30
        Ce = self.build_coedges_of_edges_tensor(body_data)

        # Try building point grids from the left coedges 
        # of each edge
        Ge = self.build_edge_grids_from_left_coedges(Gc, Ce, body_data)

        Cf, Csf, new_to_old_face_indices = self.build_coedges_of_faces_tensor(
            body_data, 
            max_coedges_per_face         
        )

        old_to_new_face_indices = self.find_inverse_permutation(new_to_old_face_indices)

        Kf_perm = old_to_new_face_indices[Kf]
        Xf_perm = Xf[new_to_old_face_indices]
        Gf_perm = Gf[new_to_old_face_indices]

        # If we are evaluating a pre-trained model on a dataset
        # with no labels then the label_dir will be none.  In this
        # case we provide some "dummy" label values here.
        if self.label_dir is not None:
            labels = self.load_labels(file_stem)
            labels_perm = labels[new_to_old_face_indices]
        else:
            labels_perm = torch.zeros(Xf_perm.size(0), dtype=torch.int64)

        data = {
            "face_features": Xf_perm,
            "face_point_grids": Gf_perm,
            "edge_features": Xe,
            "edge_point_grids": Ge,
            "coedge_features": Xc,
            "coedge_point_grids": Gc,
            "coedge_lcs": lcs,
            "face_kernel_tensor": Kf_perm,
            "edge_kernel_tensor": Ke,
            "coedge_kernel_tensor": Kc,
            "coedges_of_edges": Ce,
            "coedges_of_small_faces": Cf,
            "coedges_of_big_faces": Csf,
            "labels": labels_perm,
            "old_to_new_face_indices": old_to_new_face_indices,
            "file_stem": file_stem
        }
        return data

    def build_kernel_tensors(self, body_data):
        n = body_data["coedge_to_next"]
        m = body_data["coedge_to_mate"]
        e = body_data["coedge_to_edge"]
        f = body_data["coedge_to_face"]

        # The permutation array n, m, e and f have the following meanings
        # For a coedge with index c:
        #   The next coedge around the loop is n[c]
        #   The mating coedge is m[c]
        #   The index of the parent edge is e[c]
        #   The index of the parent face is f[c]

        # All these arrays should have the same size. i.e. the number of coedges 
        # in the body
        num_coedges = n.size
        assert num_coedges == m.size
        assert num_coedges == e.size
        assert num_coedges == f.size

        # Now we need to recover the permutation p with the meaning
        # p[c] is index of the previous coedge in the loop to the 
        # coedge with index c.   This is a little like finding the
        # inverse (also the transpose) of a permutation matrix.
        p = self.find_inverse_permutation(n)

        # Now we can build the kernel tensors for the faces, edges and coedges
        Kf = self.build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel["faces"])
        Ke = self.build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel["edges"])
        Kc = self.build_kernel_tensor_from_topology(n, p, m, e, f, self.kernel["coedges"])

        return Kf, Ke, Kc


    def build_input_feature_tensors(self, body_data):
        """
        Convert the feature tensors for faces, edges and coedges
        from numpy to pytorch
        """
        Xf = torch.from_numpy(body_data["face_features"])
        Xe = torch.from_numpy(body_data["edge_features"])
        Xc = torch.from_numpy(body_data["coedge_features"])

        Xf = self.standardize_features(
            Xf, 
            self.feature_standardization["face_features"]
        )
        Xe = self.standardize_features(
            Xe, 
            self.feature_standardization["edge_features"]
        )
        Xc = self.standardize_features(
            Xc, 
            self.feature_standardization["coedge_features"]
        )

        return Xf, Xe, Xc


    def standardize_features(self, feature_tensor, stats):
        num_features = len(stats)
        assert feature_tensor.size(1) == num_features
        means = torch.zeros(num_features, dtype=feature_tensor.dtype)
        sds = torch.zeros(num_features, dtype=feature_tensor.dtype)
        eps = 1e-7
        for index, s in enumerate(stats):
            assert s["standard_deviation"] > eps, "Feature has zero standard deviation"
            means[index] = s["mean"]
            sds[index] = s["standard_deviation"]

        # We need to broadcast means and sds over the number of entities
        means.unsqueeze(0)
        sds.unsqueeze(0)
        feature_tensor_zero_mean = feature_tensor - means
        feature_tensor_standadized = feature_tensor_zero_mean / sds

        # Test code to check this works
        num_ents = feature_tensor.size(0) 
        test_tensor = torch.zeros((num_ents, num_features), dtype=feature_tensor.dtype)
        for i in range(num_ents):
            for j in range(num_features):
                value = (feature_tensor[i,j] - means[j])/sds[j]
                test_tensor[i,j] = value
        assert torch.allclose(feature_tensor_standadized, test_tensor, eps)

        # Convert the tensors to floats after standardization 
        return feature_tensor_standadized.float()

    def build_point_grids(self, body_data):
        """
        Read the point grid and LCS data and convert it to
        pytorch
        """
        face_point_grids = torch.from_numpy(body_data["face_point_grids"])
        coedge_point_grids = torch.from_numpy(body_data["coedge_point_grids"])
        coedge_lcs = torch.from_numpy(body_data["coedge_lcs"])
        return face_point_grids.float(), coedge_point_grids.float(), coedge_lcs.float()


    def build_edge_grids_from_left_coedges(self, Gc, Ce, body_data):
        """
        Try building point grids from the left coedges 
        of each edge
        """
        # Lets do this the slow way to start with
        num_edges = Ce.size(0)
        left_coedge_indices = torch.zeros(num_edges, dtype=torch.int64)
        reverse_flags = body_data["coedge_reverse_flags"]
        
        for edge_index in range(num_edges):
            coedges = Ce[edge_index, :]
            first_coedge_index = coedges[0]
            second_coedge_index = coedges[1]
            if reverse_flags[second_coedge_index]==1:
                left_coedge_indices[edge_index] = first_coedge_index
            else: 
                left_coedge_indices[edge_index] = second_coedge_index
        return Gc[left_coedge_indices]


    def	build_kernel_tensor_from_topology(self, n, p, m, e, f, kernel):
        """
        A BRepNet kernel is defined by a list of topological walks.  
        These are used to create the index tensors Kf, Ke and Kc
        for faces edges and coedges separately.  
        
        The index tensor Kf, Ke and Kc are essentially permutations
        on the arrays of coedges.  The BRepNet paper describes these 
        as permutation matrices,  but actually we don't need a full
        matrix.
        
        This function build the permutations as follows.

        - We start off with just a list of indices in order
          [ 0, 1, 2, ...  num_coedges-1]

        - For each entity in the kernel we need to find its index.
          This will be found by following a topological walk.  
          The list of instructions in the walk is defined in the kernel.
          To execute one instruction we simple use the current coedge
          indices as the index of either the next, previous, mate, 
          edge or face permutation.  i.e.

          c = n[c]

          will set the permutations in c to c->next()

        - We do this for all the instructions in the walk.
        """
        # The permutation array n, p, m, e and f have the following meanings
        # For a coedge with index c:
        #   The next coedge around the loop is n[c]
        #   The previous coedge around the loop is p[c]
        #   The mating coedge is m[c]
        #   The index of the parent edge is e[c]
        #   The index of the parent face is f[c]
        # Each of the arrays n, p, m, e and f should have the same
        # length.  i.e. the number of coedges in the B-Rep
        num_coedges = n.size
        assert num_coedges == p.size
        assert num_coedges == m.size
        assert num_coedges == e.size
        assert num_coedges == f.size

        # We want to build an integer tensor of size
        # [ num_coedges x num_entities_in_kernel ]
        # We will build this one column at a time
        kernel_tensor_cols = []
        for walk_instructions in kernel:
            # This is like starting with the identity matrix.
            # The identity permutation is just the indices 
            # [0, 1, 2, ...] 
            c = np.arange(num_coedges, dtype=n.dtype)
 
            # Loop over the instructions and execute each one
            for instruction in walk_instructions:
                if instruction == "n":
                    c = n[c]
                elif instruction == "p":
                    c = p[c]
                elif instruction == "m":
                    c = m[c]
                elif instruction == "f":
                    c = f[c]
                elif instruction == "e":
                    c = e[c]
                else:
                    assert False, "Unknown instruction"

            kernel_tensor_col = torch.from_numpy(c.astype(dtype=np.int64))
            kernel_tensor_cols.append(kernel_tensor_col)

        kernel_tensor = torch.transpose(torch.stack(kernel_tensor_cols), 0, 1)

        assert kernel_tensor.size(0) == num_coedges
        assert kernel_tensor.size(1) == len(kernel)
        return kernel_tensor

    def find_face_permutation(self, body_data, max_coedges_per_face):
        """
        We will need to rearrange the array of faces so that 
        faces with more than max_coedges_per_face coedges 
        appear at the end of the faces array
        """
        num_faces = body_data["face_features"].shape[0]
        coedge_to_face = body_data["coedge_to_face"]
        coedges_per_face = np.bincount(coedge_to_face)
        small_face_indices = np.where(coedges_per_face <= max_coedges_per_face)[0]
        big_face_indices = np.where(coedges_per_face > max_coedges_per_face)[0]
        if small_face_indices.size == 0:
            face_permutation = big_face_indices
        elif big_face_indices.size == 0:
            face_permutation = small_face_indices
        else:
            face_permutation = np.concatenate((small_face_indices, big_face_indices))
        return face_permutation, small_face_indices.size


    def build_coedges_of_edges_tensor(self, body_data):
        """
        We want to build an index tensor Ce with 

            Ce.size() = [ num_edges x 2]

        The ith row of Ce contains the indices of the two
        coedges belonging to the ith parent edge
        """
        coedge_to_edge = body_data["coedge_to_edge"]
        num_edges = body_data["edge_features"].shape[0]
        coedges_of_edges = [ [] for i in range(num_edges)]
        for coedge_index, edge_index in enumerate(coedge_to_edge):
            coedges_of_edges[edge_index].append(coedge_index)
        
        for coedges in coedges_of_edges:
            assert len(coedges) == 1 or len(coedges) == 2
            if len(coedges) == 1:
                # OK.  This is the special case of a sphere.  Here
                # we have two coedges at te poles.  They link to themselves.
                # As they are only used one each they don't have a second
                # coedge in the coedges_of_edges array.  We add the same
                # coedge twice here
                coedges.append(coedges[0])

        coedges_of_edges = torch.tensor(coedges_of_edges, dtype=torch.int64)
        return coedges_of_edges


    def build_coedges_of_faces_tensor(
            self, 
            body_data,
            max_coedges_per_face      
        ):
        """
        For faces with less than or equal to max_coedges_per_face we 
        want to build a tensor

        Cf.size() = [ num_small_faces x max_coedges ]
        
        The tensor is padded with the index num_coedges.  A row of
        zeros will be concatenated to the mlp output tensor to allow 
        this padding to work.

        For faces with more than max_coedges coedges we have
        an array of tensors of different sizes
    
        Csf = [
            Csf.size() = [ num_coedges_in_face_1 ],
            ...
        ]
        """
        coedge_to_face = body_data["coedge_to_face"]
        num_faces = body_data["face_features"].shape[0]
        num_coedges = coedge_to_face.size

        face_to_coedges = {}
        for coedge_index, face_index in enumerate(coedge_to_face):
            if not face_index in face_to_coedges:
                face_to_coedges[face_index] = []
            face_to_coedges[face_index].append(coedge_index)
        
        small_face_indices = []
        big_face_indices = []
        Cf = []
        Csf = []
        for face_index in range(num_faces):
            assert face_index in face_to_coedges
            assert len(face_to_coedges[face_index]) > 0
            if len(face_to_coedges[face_index]) > max_coedges_per_face:
                # This is the case where we have faces with lots of
                # coedges around them.  We place these long lists of
                # indices into an index tensor for each face
                Csf.append(torch.tensor(face_to_coedges[face_index], dtype=torch.int64))
                big_face_indices.append(face_index)
            else:
                # This is the case where we have only a small number
                # of coedges around a face.  We want to build an 
                # index tensor
                #
                # Cf.size() = [ num_small_faces x max_coedges ]
                # 
                # with the rows padded with the value 'num_coedges'
                coedges_of_face = face_to_coedges[face_index]

                # We need to pad the array with the value 'num_coedges'
                padding_size = max_coedges_per_face - len(coedges_of_face)
                coedges_of_face.extend([num_coedges] * padding_size)

                # Append the row to the list.  We will stack this once it
                # has been accumulated
                Cf.append(torch.tensor(coedges_of_face, dtype=torch.int64))
                small_face_indices.append(face_index)

        # Stack the rows for the "small faces"
        Cf = torch.stack(Cf)

        # Finally we need to define a "permutation" index tensor.  We want
        # re-arrange the face features Xf and and face indices in Kf so that the 
        # big faces come at the end of the tensors 
        small_face_indices = torch.tensor(small_face_indices, dtype=torch.int64)
        big_face_indices = torch.tensor(big_face_indices, dtype=torch.int64)
        face_permutation = torch.cat([small_face_indices, big_face_indices])

        return Cf, Csf, face_permutation


    def load_labels(self, file_stem):
        """
        Load the segmentation from the seg file
        """
        label_pathname = self.label_dir / (file_stem + ".seg")
        face_labels = np.loadtxt(label_pathname, dtype=np.int64)
        face_labels_tensor = torch.from_numpy(face_labels)
        if face_labels_tensor.ndim == 0:
            face_labels_tensor = torch.unsqueeze(face_labels_tensor, 0)
        return face_labels_tensor


    def find_inverse_permutation(self, perm):
        assert perm.ndim == 1

        # We create the identity permutation.  i.e. [0, 1, 2, ...]
        if isinstance(perm, np.ndarray):
            identity = np.arange(perm.size, dtype=perm.dtype)
            inv_perm = np.zeros(perm.size, dtype=perm.dtype)
        else:
            assert isinstance(perm, torch.Tensor)
            identity = torch.arange(end=perm.size(0), dtype=perm.dtype)
            inv_perm = torch.zeros(perm.size(0), dtype=perm.dtype)

        # Now we know that inv_perm[perm] is the identity.
        # We can "assign" values of inv_perm[perm] from the 
        # identity array
        inv_perm[perm] = identity
        return inv_perm
        
def unsqueeze_single_dim_tensors(tensor_list):
    unsqueezed = []
    for t in tensor_list:
        if t.ndim == 1:
            unsqueezed.append(torch.unsqueeze(t, 0))
        else:
            unsqueezed.append(t)
    return unsqueezed

def concatenate_tensor_arrays(small, big, unsqueeze):
    all = []
    if unsqueeze:
        small = unsqueeze_single_dim_tensors(small)
        big = unsqueeze_single_dim_tensors(big)
    all.extend(small)
    all.extend(big)
    return torch.cat(all)

def add_offset_to_face_index(
        face_indices, 
        num_small_faces, 
        small_face_offset, 
        big_face_offset
    ):
    face_index_offset = small_face_offset*(face_indices < num_small_faces)

    # The array was re-ordered so that small faces are always before
    # big faces.
    # The first big face has index "num_small_faces" are we need to map this to
    # big_face_offset
    face_index_offset += (big_face_offset-num_small_faces)*(face_indices >= num_small_faces)
    return face_indices + face_index_offset


def add_offset_to_coedge_index_with_padding(
        padded_coedge_indices,
        pad_value,
        coedge_index_offset,
        new_pad_value
    ):
    """
    In the coedges_of_small_faces array we have the indices
    of coedges with padding at the end of the array.   The 
    value of the padding in the input tensor is pad_value.
    We want to change the values of the valid coedge indices 
    and the padding separately.  The valid values want to 
    be increased by  `coedge_index_offset`.   The padding
    needs to be replaced by `new_pad_value`
    """
    coedge_index_offset = coedge_index_offset*(padded_coedge_indices != pad_value)
    coedge_index_offset += (new_pad_value-pad_value)*(padded_coedge_indices == pad_value)
    return padded_coedge_indices + coedge_index_offset


def brepnet_collate_fn(data_list):
    """
    Collate the data from multiple bodies into a single
    set of tensors.
    Here I call the faces with a small number of coedges
    "small faces" and the coedges with a large number of
    coedges "big faces".
    """
    Xf_small_faces = []
    Xf_big_faces = []
    Gf_small_faces = []
    Gf_big_faces = []
    labels_small_faces = []
    labels_big_faces = []

    Xe = []
    Ge = []
    Xc = []
    Gc = []
    lcs = []

    Ke = [] # Edge indices for each coedge
    Kc = [] # Coedge indices for each coedge

    Ce = []   # Coedge indices for coedges owned by each edge
    Csf = []  # Coedge indices for coedges owned by "big faces"

    # Keep track of which file each B-Rep came from
    file_stems = []

    # Keep track of the data we need to split the batch
    split_batch = []

    # We need to make two passes through the data.
    # In the first pass we process things which depend on
    # coedge index and small faces.  In the second pass
    # we work on modifying indices which depend on big faces
    face_offset = 0
    edge_offset = 0
    coedge_offset = 0

    for data in data_list:
        # These are input features.  We just wan to concatentate 
        # the tensors for each B-Rep
        num_edges = data["edge_features"].shape[0]
        Xe.append(data["edge_features"])
        Ge.append(data["edge_point_grids"])

        num_coedges = data["coedge_features"].shape[0]
        Xc.append(data["coedge_features"])
        Gc.append(data["coedge_point_grids"])
        lcs.append(data["coedge_lcs"])

        # For edge and coedge indices things are easy.  We just need to 
        # add the offsets to the arrays
        Ke.append(data["edge_kernel_tensor"] + edge_offset)
        Ce.append(data["coedges_of_edges"] + coedge_offset)
        Kc.append(data["coedge_kernel_tensor"] + coedge_offset)
        
        for single_face_coedges in data["coedges_of_big_faces"]:
            Csf.append(single_face_coedges + coedge_offset)
        
        # Face features are a little more complicated.  We want
        # to keep the faces with a small number of coedges
        # at the start of the array and the faces with a large
        # number of coedges in another array which will get
        # appended to the end.

        num_small_faces = data["coedges_of_small_faces"].shape[0]
        num_big_faces = len(data["coedges_of_big_faces"])
        Xf = data["face_features"]
        Gf = data["face_point_grids"]
        assert num_small_faces + num_big_faces == Xf.shape[0]
        assert num_small_faces + num_big_faces == Gf.shape[0]
        
        # We need to slice the face feature tensor
        Xf_small_faces.append(Xf[:num_small_faces])      
        Xf_big_faces.extend(Xf[num_small_faces:])

        Gf_small_faces.append(Gf[:num_small_faces])
        Gf_big_faces.append(Gf[num_small_faces:])

        labels = data["labels"]
        labels_small_faces.append(labels[:num_small_faces])
        labels_big_faces.append(labels[num_small_faces:])

        file_stems.append(data["file_stem"])

        split_batch_data_for_brep = {
            "edge_indices": torch.arange(edge_offset, edge_offset+num_edges, dtype=torch.int64),
            "coedge_indices": torch.arange(coedge_offset, coedge_offset+num_coedges, dtype=torch.int64)
        }
        split_batch.append(split_batch_data_for_brep)

        face_offset += num_small_faces
        edge_offset += num_edges
        coedge_offset += num_coedges
    
    # This is the second pass through the data.  We need to
    # set the indices of faces in Kf and the new_indices_of_brep_faces
    # for each face in each B-Rep
    Kf = []

    # The coedge indices for coedges owned by "small faces" also needs to 
    # be processed here as we need to apply different values to the 
    # coedge indices and the padding
    new_padding_value = coedge_offset
    coedge_offset = 0
    Cf = []   
    small_face_offset = 0
    for solid_index, data in enumerate(data_list):
        num_small_faces = data["coedges_of_small_faces"].shape[0]
        num_big_faces = len(data["coedges_of_big_faces"])
        old_to_new_face_index = data["old_to_new_face_indices"]
        num_coedges = data["coedge_features"].shape[0]

        # We need to add on the offsets for the new face indices
        # Here this is done for the indices which allow us to get
        # back from the combined batch to the original face indices
        # in each B-Rep
        offset_old_to_new_face_index = add_offset_to_face_index(
            old_to_new_face_index, 
            num_small_faces, 
            small_face_offset, 
            face_offset
        )

        offset_coedges_of_small_faces = add_offset_to_coedge_index_with_padding(
            data["coedges_of_small_faces"],
            num_coedges,
            coedge_offset,
            new_padding_value
        )
        Cf.append(offset_coedges_of_small_faces)

        split_batch[solid_index]["face_indices"] = offset_old_to_new_face_index

        # Here we add on the offsets for the kernel Kf
        brep_Kf = data["face_kernel_tensor"]
        offset_Kf = add_offset_to_face_index(
            brep_Kf, 
            num_small_faces, 
            small_face_offset, 
            face_offset
        )
        Kf.append(offset_Kf)

        small_face_offset += num_small_faces
        face_offset += num_big_faces
        coedge_offset += num_coedges
        
    batch_data = {
        "face_features": concatenate_tensor_arrays(Xf_small_faces, Xf_big_faces, unsqueeze=True),
        "face_point_grids": concatenate_tensor_arrays(Gf_small_faces, Gf_big_faces, unsqueeze=True),
        "edge_features": torch.cat(Xe),
        "edge_point_grids": torch.cat(Ge),
        "coedge_features": torch.cat(Xc),
        "coedge_point_grids": torch.cat(Gc),
        "coedge_lcs": torch.cat(lcs),
        "face_kernel_tensor": torch.cat(Kf),
        "edge_kernel_tensor": torch.cat(Ke),
        "coedge_kernel_tensor": torch.cat(Kc),
        "coedges_of_edges": torch.cat(Ce),
        "coedges_of_small_faces": torch.cat(Cf),
        "coedges_of_big_faces": Csf,
        "labels": concatenate_tensor_arrays(labels_small_faces, labels_big_faces, unsqueeze=False),
        "split_batch": split_batch,
        "file_stems": file_stems
    }
    return batch_data