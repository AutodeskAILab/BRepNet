import torch
from torch.utils.data import Dataset
from pathlib import Path
import math
import hashlib
import copy
import pickle

import utils.data_utils as data_utils

class BRepNetDatasetOld(Dataset):
    """Dataset of BRepNet data"""
        
    def __init__(self, opts, train_val_or_test):
        """
        Construct the dataset from the given options
        """
        super(BRepNetDatasetOld).__init__()
        self.opts = opts

        # Load the topological walks in to be used in the kernel
        self.kernel = data_utils.load_json_data(self.opts.kernel)

        # Load the list of input features to be used
        self.feature_lists = data_utils.load_json_data(self.opts.input_features)
        
        # The dataset file holds the full list of batches and the
        # feature normalization information
        dataset_info = data_utils.load_json_data(self.opts.dataset_file)

        self.batches = dataset_info[train_val_or_test]["batches"]
        self.feature_normalization = dataset_info["feature_normalization"]
        self.root_dir = Path(self.opts.dataset_dir)

        # The cache dir contains a binary copy of all the data.
        # we make sure that different hyper-parameters generate
        # different cache files as appropriate.
        # Using the cache gives a massive speedup when training 
        # the network
        self.cache_dir = self.root_dir / "cache"
        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
        

    def __len__(self):
        """
        The dataloader returns the dataset already divided into
        mini-batches.
        """
        return len(self.batches)

    def __getitem__(self, batch_idx):
        """
        Load a batch of data.  Use a cache if its there.  
        Cache the binary data if not.
        """ 
        cache_pathname = self.get_cache_pathname(batch_idx)
        if cache_pathname.exists():
            batch = self.load_batch_from_cache(cache_pathname)
        else:
            batch = self.load_and_cache_batch(batch_idx, cache_pathname)
        return batch


    def hash_strings_in_list(self, string_list):
        """
        Create a hash of the strings in the list
        """
        single_str = str().join(string_list)
        single_str = single_str.encode('ascii', 'replace')

        return hashlib.sha224(single_str).hexdigest()


    def hash_data_for_batch(self, batch_basenames):
        """
        We want to be sure that the cache is correctly built given
        all the hyper-parameters of the network.
        Here we make a hash of these hyper-parameters and 
        use this as the pathname for the cache
        """
        string_list = copy.deepcopy(batch_basenames)
        string_list.append("BRepNetDatasetOld")
        for ent in self.kernel:
            for walk in self.kernel[ent]:
                string_list.append(walk)
        for ent in self.feature_lists.values():
            feature_list = list(ent)
            sorted_feature_list = sorted(feature_list)
            for feature in sorted_feature_list:
                string_list.append(feature)
        return self.hash_strings_in_list(string_list)

    def get_cache_pathname(self, batch_idx):
        """
        Create a pathname for the cache file for the batch with 
        batch_idx
        """
        batch_basenames = self.batches[batch_idx]
        hstring = self.hash_data_for_batch(batch_basenames)
        return (self.cache_dir/hstring).with_suffix(".p")



    def save_batch_cache(self, cache_pathname, data):
        """
        Save the cache data for the batch
        """
        fh = open(cache_pathname, "wb")
        pickle.dump(data, fh)
        fh.close()

    def load_batch_from_cache(self, cache_pathname):
        """
        Load cache data for the batch with the given pathname
        """
        fh = open(cache_pathname,"rb")
        data = pickle.load(fh)
        fh.close()
        return data

    def load_and_cache_batch(self, batch_idx, cache_pathname):
        """
        Load the batch with batch_idx from the
        raw json data, then save a cache of the 
        binary tensors
        """
        batch_data = self.load_batch(batch_idx)
        self.save_batch_cache(cache_pathname, batch_data)
        return batch_data


    def load_feature_data(self, basename):
        """
        Load feature file
        """
        feature_data_pathname = self.root_dir / (basename+"_features.json")
        return data_utils.load_json_data(feature_data_pathname)


    def load_topology_file(self, basename):
        """
        Load the topology file
        """
        topology_pathname = self.root_dir / (basename + "_topology.json")
        return data_utils.load_json_data(topology_pathname)


    def load_face_label_file(self, basename):
        """
        Load face label file
        """
        label_file_pathname = self.root_dir / ( basename+"_labels.json")

        # If we are evaluating the model then the labels will not be
        # present
        if not label_file_pathname.exists():
            return None
        return data_utils.load_json_data(label_file_pathname)


    def	build_kernel_tensor_from_top(self, topology, kernel):
        """
        A BRepNet kernel is defined by a list of topological walks.  
        These are used to create the index tensors Kf, Ke and Kc
        for faces edges and coedges separately.  In this function
        we build one of Kf, Ke or Kc.

        Each walk is a string containing a series of instructions. 
        We execute the instructions in order to walk over the entities and
        arrive at the destination.  The index of the destination entity
        is then added to the index tensor.
        """

        # We want to build an integer tensor of size
        # [ num_coedges x num_entities_in_kernel ]
        num_coedges = len(topology["coedges"])
        num_ents_in_kernel = len(kernel)
        kernel_tensor = torch.LongTensor(num_coedges, num_ents_in_kernel)

        # Loop over the coedges in the topology
        for coedge_index, coedge in enumerate(topology["coedges"]):

            # Loop over the list of walks.  This is like a loop over the 
            # faces, edges or coedges which will appear in the kernel
            for walk_index, walk_instructions in enumerate(kernel):

                # Start the walk
                walked_on_entity = coedge_index

                # Loop over the instructions and execute each one
                for instruction in walk_instructions:
                    if instruction == "n":
                        walked_on_entity = self.get_index_of_next(topology, walked_on_entity)
                    elif instruction == "p":
                        walked_on_entity = self.get_index_of_previous(topology, walked_on_entity)
                    elif instruction == "m":
                        walked_on_entity = self.get_index_of_mate(topology, walked_on_entity)
                    elif instruction == "f":
                        walked_on_entity = self.get_index_of_face(topology, walked_on_entity)
                    elif instruction == "e":
                        walked_on_entity = self.get_index_of_edge(topology, walked_on_entity)
                    else:
                        assert False, "Unknown instruction"
                # Now we have completed the walk and arrived at the appropriate
                # entity we just need to set its index into the index tensor 
                kernel_tensor[coedge_index, walk_index] = walked_on_entity
        return kernel_tensor


    def build_coedges_of_faces_tensors(
            self, 
            topology, 
            num_coedges_per_face, 
            max_coedges, 
            pad_row
        ):
        """
        Build the index tensor telling which coedges belong 
        to which face.  As the number of coedges in a face 
        can vary then we split the tensor into two parts.
        
        There will be num_small_faces faces where the 
        number of coedges is smaller than or equal to
        max_coedges.
        
        There will be num_big_faces where the number of
        coedges per face is greater than max_coedges
        
        coedges_of_faces - size  = [ num_small_faces x  max_coedges ]
        
        If a face has less than max_coedges then we make 
        add the pad_row index to pad out the tensor
        
        coedges_of_single_faces
        An array
            [
            [ num_coedges_of_big_face_1 ],
            [ num_coedges_of_big_face_2 ],
            .
            .
            [ num_coedges_of_big_face_num_big_faces-1 ]
            ]
        """
        num_small_faces = 0
        for c in num_coedges_per_face:
            if c <= max_coedges:
                num_small_faces += 1

        coedges_of_faces_tensor = torch.LongTensor(num_small_faces, max_coedges)
        coedges_of_single_faces = []
        row = 0
        for i, face in enumerate(topology["faces"]):
            col = 0
            if num_coedges_per_face[i] <= max_coedges:
                for loop_id in face["loops"]:
                    loop = topology["loops"][loop_id]
                    for coedge in loop["coedges"]:
                        coedges_of_faces_tensor[row, col] = coedge
                        col += 1
                while col < max_coedges:
                    coedges_of_faces_tensor[row, col] = pad_row
                    col += 1
                row += 1
            else:
                coedge_ids = []
                for loop_id in face["loops"]:
                    loop = topology["loops"][loop_id]
                    for coedge in loop["coedges"]:
                        coedge_ids.append(coedge)
                coedge_tensor = torch.LongTensor(coedge_ids)
                coedges_of_single_faces.append(coedge_tensor)
        return coedges_of_faces_tensor, coedges_of_single_faces


    def build_coedges_of_edges_tensor(self, topology):
        """
        Build the index tensor telling us which coedges 
        belong to a given edge
        size = [ num_edges x 2 ]
        """
        num_edges = len(topology["edges"])
        coedges_of_edges_tensor = torch.LongTensor(num_edges, 2)
        for i, edge in enumerate(topology["edges"]):
            assert len(edge["coedges"]) == 2
            for j, coedge in enumerate(edge["coedges"]):
                coedges_of_edges_tensor[i,j] = coedge
        return coedges_of_edges_tensor

    def get_index_of_next(self, topology, coedge_index):
        """
        Get the index of the next coedge
        """
        coedges = topology["coedges"]
        assert coedge_index < len(coedges)
        coedge = coedges[coedge_index]
        return coedge["next"]


    def get_index_of_previous(self, topology, coedge_index):
        """
        Get the index of the previous coedge
        """
        coedges = topology["coedges"]
        assert coedge_index < len(coedges)
        coedge = coedges[coedge_index]
        return coedge["previous"]


    def get_index_of_mate(self, topology, coedge_index):
        """
        Get the index of the mating coedge
        """
        coedges = topology["coedges"]
        assert coedge_index < len(coedges)
        coedge = coedges[coedge_index]
        return coedge["partner"]

    def get_index_of_edge(self, topology, coedge_index):
        """
        Get the index of the edge
        """
        coedges = topology["coedges"]
        assert coedge_index < len(coedges)
        coedge = coedges[coedge_index]
        return coedge["edge"]

    def get_index_of_face(self, topology, coedge_index):
        """
        Get the index of the face to which this coedge belongs
        """
        coedges = topology["coedges"]
        loops = topology["loops"]
        assert coedge_index < len(coedges)
        coedge = coedges[coedge_index]
        loop_index = coedge["loop"]
        loop = loops[loop_index]
        return loop["face"]


    def load_batch(self, batch_idx):
        """
        Load data from the json files and generate a binary cache
        of this information
        """
        batch_basenames = self.batches[batch_idx]

        # First we load and standardize the batch
        batch, batch_face_labels = self.load_batch_and_standardize(batch_basenames)


        # Now we create the single dictionary which concatenates all the 
        # data into a single solid
        single_batch_solid = self.create_empty_batch_solid()

        for index, data in enumerate(batch):
            self.concatenate_entities(single_batch_solid, data, index)

        # Concatenate all the batch labels into a single tensor
        all_batch_face_labels = torch.cat(batch_face_labels)

        # Sort the faces in order of the number of coedges in each face
        permutation = self.sort_faces_by_num_coedges(single_batch_solid)
        perm_all_batch_face_labels = all_batch_face_labels[permutation]

        # First build the feature tensors
        # Xf - face_features 
        # Xe - edge features
        # Xc - coedge features
        feature_data = single_batch_solid["feature_data"]
        face_features = self.build_feature_tensor(
            feature_data["face_features"], 
            self.feature_lists["face_features"]
        )
        edge_features = self.build_feature_tensor(
            feature_data["edge_features"], 
            self.feature_lists["edge_features"]
        )
        coedge_features = self.build_feature_tensor(
            feature_data["coedge_features"], 
            self.feature_lists["coedge_features"]
        )

        # Get the one topology for the entire solid
        topology = single_batch_solid["topology"]

        # Next we build the kernel tensors for faces, edges and coedges
        # Kf size [ num_coedges x num_faces_in_kernel ]
        # Ke size [ num_coedges x num_edges_in_kernel ]
        # Kc size [ num_coedges x num_coedges_in_kernel ]
        face_kernel_tensor = self.build_kernel_tensor_from_top(
            topology, 
            self.kernel["faces"]
        )
        edge_kernel_tensor =  self.build_kernel_tensor_from_top(
            topology,  
            self.kernel["edges"]
        )
        coedge_kernel_tensor = self.build_kernel_tensor_from_top(
            topology, 
            self.kernel["coedges"]
        )

        # Create the coedges of edges tensor
        coedges_of_edges = self.build_coedges_of_edges_tensor(topology) 

        # Finally we build the tensors which map the coedges to
        # their owner faces
        max_coedges = 30
        coedges_of_faces_tensor, coedges_of_single_faces = \
            self.build_coedges_of_faces_tensors(
                topology, 
                single_batch_solid["num_coedges_per_face"], 
                max_coedges, 
                len(topology["coedges"])
            )

        split_batch = self.build_face_index_tensors(
            single_batch_solid["id_lookup"], 
            batch
        )

        return {
            "face_features": face_features,
            "edge_features": edge_features,
            "coedge_features": coedge_features, 
            "face_kernel_tensor": face_kernel_tensor,
            "edge_kernel_tensor": edge_kernel_tensor, 
            "coedge_kernel_tensor": coedge_kernel_tensor, 
            "coedges_of_edges": coedges_of_edges, 
            "coedges_of_small_faces": coedges_of_faces_tensor,
            "coedges_of_big_faces": coedges_of_single_faces,
            "labels": perm_all_batch_face_labels,
            "split_batch": split_batch,
            "file_stems": batch_basenames
        }


    def standardize_features_for_entity_type(self, feature_data, feature_normalization):
        """
        Standardize feature data for faces, edges or coedges
        """
        all_feature_names = set()
        for name in self.feature_lists["face_features"]:
            all_feature_names.add(name)
        for name in self.feature_lists["edge_features"]:
            all_feature_names.add(name)
        for name in self.feature_lists["coedge_features"]:
            all_feature_names.add(name)
        for entity in feature_data:
            for feature in entity["features"]:
                if feature["feature_name"] in all_feature_names:
                    mean = feature_normalization[feature["feature_name"]]["mean"]
                    variance = feature_normalization[feature["feature_name"]]["variance"]

                    original_value = feature["feature_value"]
                    new_value = original_value-mean

                    # Check for very small variation in the feature.
                    # It could be that in a given (small) dataset then all the values
                    # are the same and the feature basically has no value at all
                    if abs(variance) > 1e-7:
                        new_value /= math.sqrt(variance)

                    # Store the Standardized value back in the feature
                    feature["feature_value"] = new_value


    def standardize_brep_feature_data(self, feature_data):
        """
        Standardize brep feature data
        """
        self.standardize_features_for_entity_type(
            feature_data["face_features"], 
            self.feature_normalization["face_features"]
        )
        self.standardize_features_for_entity_type(
            feature_data["edge_features"], 
            self.feature_normalization["edge_features"]
        )
        self.standardize_features_for_entity_type(
            feature_data["coedge_features"], 
            self.feature_normalization["coedge_features"]
        )


    def load_batch_and_standardize(self, batch_basenames):
        """
        Load the data for the batch and apply the normalization
        """
        batch = []
        batch_face_labels = []
        for basename in batch_basenames:
            data_loaded = self.load_feature_data(basename)
            self.standardize_brep_feature_data(data_loaded["feature_data"])

            # Also load the topology data
            top_data = self.load_topology_file(basename)
            data_loaded["topology"] = top_data["topology"]

            face_label_data_loaded = self.load_face_label_file(basename)
            if face_label_data_loaded is not None:
                num_faces = len(face_label_data_loaded["face_labels"])
                num_labels_per_face = len(face_label_data_loaded["face_labels"][0]["labels"])
                face_labels = torch.IntTensor(num_faces, num_labels_per_face)

                for i in range(num_faces):
                    for j in range(num_labels_per_face):
                        face_labels[i,j] = face_label_data_loaded["face_labels"][i]["labels"][j]["label_value"]
            else:
                # This is the case where we don't have any labels
                # probably because we are evaluating the model
                print("Warning - Working without labels.  This is for evaluation only")
                num_faces = len(data_loaded["feature_data"]["face_features"])
                num_labels_per_face = len(self.feature_lists["face_features"])
                face_labels = torch.zeros(num_faces, num_labels_per_face, dtype=torch.int64)

            segment_indices = torch.argmax(face_labels, dim=1)
            batch_face_labels.append(segment_indices)



            batch.append(data_loaded)
            
        return batch, batch_face_labels


    def create_empty_batch_solid(self):
        """
        Create a dictionary which we can contatenate with
        other data to build up the single solid for the batch 
        """
        return {
            "feature_data": {
                "face_features": [],
                "edge_features": [],
                "coedge_features": []
            },
            "id_lookup": {
                "orig_face_ids": [],
                "orig_edge_ids": [],
                "orig_coedge_ids": [],
                "orig_loop_ids": [],
                "orig_vertex_ids": []
            },
            "topology": {
                "faces": [],
                "edges": [],
                "coedges": [],
                "loops": [],
                "vertices": []
            }
        }


    def build_feature_tensor(self, ent_features, feature_list):
        """
        Build feature tensor
        size = [ num_ents x num_features ]
        """
        num_ents = len(ent_features)
        num_features = len(feature_list)
        feature_tensor = torch.Tensor(num_ents, num_features)
        for i, ent in enumerate(ent_features):
            features = ent["features"]
            feature_index = 0
            for j, feature in enumerate(features):
                feature_name = feature["feature_name"]
                if feature_name in feature_list:
                    value = feature["feature_value"]
                    feature_tensor[i,feature_index] = value
                    feature_index += 1
            assert feature_index == num_features
        return feature_tensor


    def find_num_coedges(self, face, topology):
        """
        Find the number of coedges on the given face
        """
        num_coedges = 0
        loops = topology["loops"]
        for loop in face["loops"]:
            num_coedges += len(loops[loop]["coedges"])
        return num_coedges


    def find_num_coedges_per_face(self, batch_data):
        """
        Find the number of coedges on each face
        """
        top_data = batch_data["topology"]
        faces = top_data["faces"]
        loops = top_data["loops"]
        num_coedges_per_face = []
        for face in faces:
            num_coedges = self.find_num_coedges(face, top_data)
            num_coedges_per_face.append(num_coedges)
        return num_coedges_per_face


    def sort_faces_by_num_coedges(self, batch_data):
        """
        Sort the faces in the batch data by the number of 
        coedges on each face.  Apply the permutation 
        to the entirety of the data
        """
        num_coedges_per_face = self.find_num_coedges_per_face(batch_data)
        permutation = sorted(range(len(num_coedges_per_face)), key=lambda k: num_coedges_per_face[k])
        
        # permutation is a list
        # [
        #	index to move to position 0,
        #	index to move to position 1,
        #	...
        # ]
        #
        # We also need an index map from old ids to new ids
        index_map = [ None ] * len(permutation)
        for new_index, p in enumerate(permutation):
            index_map[p] = new_index

        # Apply the permutation all the lists
        old_face_features = batch_data["feature_data"]["face_features"]
        topology = batch_data["topology"]

        new_face_features = []
        new_top_faces = []
        perm_orig_face_ids = []
        perm_num_coedges_per_face = []

        for to_index, from_index in enumerate(permutation):

            # Re-order the face features
            face_feature = old_face_features[from_index]
            assert face_feature["entity"] == from_index
            copy_face_feature = copy.deepcopy(face_feature)
            copy_face_feature["entity"] = to_index
            new_face_features.append(face_feature)

            # Re-order the faces in the topology
            face = topology["faces"][from_index]
            copy_face = copy.deepcopy(face)
            new_top_faces.append(copy_face)

            # Re-order the original face ids
            orig_face_id = batch_data["id_lookup"]["orig_face_ids"][from_index]
            copy_orig_face_id = copy.deepcopy(orig_face_id)
            perm_orig_face_ids.append(copy_orig_face_id)

            # Re-order the num_coedges_per_face array
            orig_num_coedges_per_face = num_coedges_per_face[from_index]
            perm_num_coedges_per_face.append(orig_num_coedges_per_face)

        batch_data["feature_data"]["face_features"] = new_face_features
        topology["faces"] = new_top_faces
        batch_data["id_lookup"]["orig_face_ids"] = perm_orig_face_ids
        batch_data["num_coedges_per_face"] = perm_num_coedges_per_face

        for loop in topology["loops"]:
            old_face_id = loop["face"]
            new_face_id = index_map[old_face_id]
            loop["face"] = new_face_id


        return permutation


    def location_in_batch(self, entity_id, solid_in_batch):
        """
        We need to record what the original batch and id
        was for each entity
        """
        return {
            "entity_id": entity_id,
            "solid_in_batch": solid_in_batch
        }


    def concatenate_entities(self, acc_batch, batch_to_cat, solid_in_batch):
        """
        Append all the data from batch_to_cat into acc_batch
        We need to re-map the indices to do this.  
        Essentially we combine all the solids in the batch 
        into a single disjoint solid
        """
        dest_feature_data = acc_batch["feature_data"]
        first_face_index = len(dest_feature_data["face_features"])
        first_edge_index = len(dest_feature_data["edge_features"])
        first_coedge_index = len(dest_feature_data["coedge_features"])

        id_lookup = acc_batch["id_lookup"]
        orig_face_ids = id_lookup["orig_face_ids"]
        orig_edge_ids = id_lookup["orig_edge_ids"]
        orig_coedge_ids = id_lookup["orig_coedge_ids"]
        orig_loop_ids = id_lookup["orig_loop_ids"]
        orig_vertex_ids = id_lookup["orig_vertex_ids"]
        assert len(orig_face_ids) == first_face_index
        assert len(orig_edge_ids) == first_edge_index
        assert len(orig_coedge_ids) == first_coedge_index


        dest_topology = acc_batch["topology"]
        assert len(dest_topology["faces"]) == first_face_index
        assert len(dest_topology["edges"]) == first_edge_index
        assert len(dest_topology["coedges"]) == first_coedge_index

        first_loop_index = len(dest_topology["loops"])
        first_vertex_index = len(dest_topology["vertices"])
        assert len(orig_loop_ids) == first_loop_index
        assert len(orig_vertex_ids) == first_vertex_index

        # Now we need to concatenate the feature data
        src_feature_data = batch_to_cat["feature_data"]

        # Concatenate the face features
        for old_face_id, face_feature in enumerate(src_feature_data["face_features"]):
            face_feature["entity"] += first_face_index
            new_face_id = len(dest_feature_data["face_features"])
            assert face_feature["entity"] == new_face_id
            orig_face_ids.append(self.location_in_batch(old_face_id, solid_in_batch))
            dest_feature_data["face_features"].append(face_feature)

        # Concatenate the edge features
        for old_edge_id, edge_feature in enumerate(src_feature_data["edge_features"]):
            edge_feature["entity"] += first_edge_index
            new_edge_id = len(dest_feature_data["edge_features"])
            assert edge_feature["entity"] == new_edge_id
            orig_edge_ids.append(self.location_in_batch(old_edge_id, solid_in_batch))
            dest_feature_data["edge_features"].append(edge_feature)

        # Concatenate the coedge features
        for old_coedge_id, coedge_feature in enumerate(src_feature_data["coedge_features"]):
            coedge_feature["entity"] += first_coedge_index
            new_coedge_id = len(dest_feature_data["coedge_features"])
            assert coedge_feature["entity"] == new_coedge_id
            orig_coedge_ids.append(self.location_in_batch(old_coedge_id, solid_in_batch))
            dest_feature_data["coedge_features"].append(coedge_feature)

        # Concatenate the topology data
        scr_topology = batch_to_cat["topology"]
        assert len(scr_topology["faces"]) == len(src_feature_data["face_features"])
        assert len(scr_topology["edges"]) == len(src_feature_data["edge_features"])
        assert len(scr_topology["coedges"]) == len(src_feature_data["coedge_features"])

        for face in scr_topology["faces"]:
            face_copy = copy.deepcopy(face)
            for i, loop in enumerate(face_copy["loops"]):
                face_copy["loops"][i] += first_loop_index
            dest_topology["faces"].append(face_copy)

        assert len(dest_topology["faces"]) == len(orig_face_ids), "Sanity check the length of the original face ids array"

        for edge in scr_topology["edges"]:
            edge_copy = copy.deepcopy(edge)
            for i, coedge in enumerate(edge["coedges"]):
                edge_copy["coedges"][i] += first_coedge_index
            for i, vertex in enumerate(edge["vertices"]):
                edge_copy["vertices"][i] += first_vertex_index
            dest_topology["edges"].append(edge_copy)

        assert len(dest_topology["edges"]) == len(orig_edge_ids), "Sanity check the length of the original edge ids array"

        for old_loop_id, loop in enumerate(scr_topology["loops"]):
            loop_copy = copy.deepcopy(loop)
            for i, coedge in enumerate(loop["coedges"]):
                loop_copy["coedges"][i] += first_coedge_index
            loop_copy["face"] += first_face_index
            new_loop_id = len(dest_topology["loops"])
            dest_topology["loops"].append(loop_copy)
            assert new_loop_id == len(orig_loop_ids)
            orig_loop_ids.append(self.location_in_batch(old_loop_id, solid_in_batch))

        for old_coedge_id, coedge in enumerate(scr_topology["coedges"]):
            coedge_copy = copy.deepcopy(coedge)
            new_coedge_id = len(dest_topology["coedges"])
            coedge_copy["loop"] += first_loop_index
            coedge_copy["edge"] += first_edge_index
            coedge_copy["next"] += first_coedge_index
            coedge_copy["previous"] += first_coedge_index
            coedge_copy["partner"] += first_coedge_index
            dest_topology["coedges"].append(coedge_copy)
            
        assert len(dest_topology["coedges"]) == len(orig_coedge_ids), "Sanity check the length of the original coedge ids array"

        for old_vertex_id, vertex in enumerate(scr_topology["vertices"]):
            vertex_copy = copy.deepcopy(vertex)
            new_vertex_id = len(dest_topology["vertices"])
            assert new_vertex_id == len(orig_vertex_ids)
            dest_topology["vertices"].append(vertex_copy)
            orig_vertex_ids.append(self.location_in_batch(old_vertex_id, solid_in_batch))


    def build_face_index_tensors(self, id_lookup, batch):
        """
        Build the index tensors single sorted faces array
        back to the original faces in the different solids
        """
        tensors = []

        # Create an array of tensors of the correct sizes
        for solid in batch:
            topology = solid["topology"]
            tensors.append(
                {
                "face_indices": torch.LongTensor(len(topology["faces"]))
                }
            )

        orig_face_ids = id_lookup["orig_face_ids"]
        for index, face_id in enumerate(orig_face_ids):
            bid = face_id["solid_in_batch"]
            fid = face_id["entity_id"]
            tensors[bid]["face_indices"][fid] = index
            
        return tensors