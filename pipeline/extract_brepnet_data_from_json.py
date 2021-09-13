
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import utils.data_utils as data_utils

class BRepNetJsonExtractor:
    def __init__(self, topology, features, feature_schema):
        self.topology = topology
        self.features = features
        self.feature_schema = feature_schema
        
    def process(self):
        """
        Process the file and extract the derivative data
        """
        face_features = self.extract_features(self.features["face_features"], self.feature_schema["face_features"])
        edge_features = self.extract_features(self.features["edge_features"], self.feature_schema["edge_features"])
        coedge_features = self.extract_features(self.features["coedge_features"], self.feature_schema["coedge_features"])

        next, mate, face, edge = self.build_incidence_lists()

        return {
            "face_features": face_features,
            "edge_features": edge_features,
            "coedge_features": coedge_features,
            "coedge_to_next": next,
            "coedge_to_mate": mate,
            "coedge_to_face": face,
            "coedge_to_edge": edge
        }

    def find_feature_index(self, feature_name, features_of_ent):
        for index, feature_data in enumerate(features_of_ent):
            if feature_data["feature_name"] == feature_name:
                return index
        assert False, "Didn't find a feature with the given name"
        return None


    def extract_features(self, features, feature_schema):
        num_ents = len(features)
        num_features = len(feature_schema)
        feature_tensor = np.zeros((num_ents, num_features))
        for ent_index, features_of_ent in enumerate(features):
            assert ent_index == features_of_ent["entity"]
            feature_data = features_of_ent["features"]
            for arr_feature_index, feature_name in enumerate(feature_schema):
                index_of_feature = self.find_feature_index(feature_name, feature_data)
                feature_tensor[ent_index, arr_feature_index] = feature_data[index_of_feature]["feature_value"]
        return feature_tensor
        
        
    def build_incidence_lists(self):
        coedges = self.topology["coedges"]
        loops = self.topology["loops"]
        num_coedges = len(coedges)
        next = np.zeros(num_coedges, dtype=np.int64)
        mate = np.zeros(num_coedges, dtype=np.int64)
        face = np.zeros(num_coedges, dtype=np.int64)
        edge = np.zeros(num_coedges, dtype=np.int64)
        for coedge_index, coedge in enumerate(coedges):
            next[coedge_index] = coedge["next"]
            mate[coedge_index] = coedge["partner"]
            edge[coedge_index] = coedge["edge"]
            loop_index = coedge["loop"]
            loop = loops[loop_index]
            face_index = loop["face"]
            face[coedge_index] = face_index
        return next, mate, face, edge

def check_faces(topology):
    faces = topology["faces"]
    for face in faces:
        # Check for faces without loops
        if len(face["loops"]) <= 0:
            return False
    return True


def check_topology(topology):
    if not check_faces(topology):
        return False
    return True


def find_stem(file):
    return file.stem.rpartition("_topology")[0]

def check_seg_file(file, seg_path):
    parent_dir = file.parent
    stem = find_stem(file)
    label_file = parent_dir / (stem + "_labels.json")
    seg_file = seg_path / (stem + ".seg")
    if not seg_file.exists():
        return False
    seg_indices = data_utils.load_labels(seg_file)
    labels = data_utils.load_json_data(label_file)
    num_faces = len(labels["face_labels"])
    if seg_indices.size != num_faces:
        print(f"Warning! number of faces in {stem} doesn't match")
        return False
    for face_index, face_labels in enumerate(labels["face_labels"]):
        segment_index = seg_indices[face_index]
        for label_index, label_obj in enumerate(face_labels["labels"]):
            label_value = label_obj["label_value"]
            label_ok = True
            if label_value > 0.5:
                if segment_index != label_index:
                    label_ok = False
            else:
                if segment_index == label_index:
                    label_ok = False
            if not label_ok:
                print(f"Warning!! seg file and label file don't match for {stem}")
                return False
    return True


def extract_brepnet_data(file, seg_path, output_path, feature_schema):
    if check_seg_file(file, seg_path):
        topology = data_utils.load_json_data(file)["topology"]
        if not check_topology(topology):
            print(f"Warning! {file} has invalid topology")
            return

        features_pathname = file.parent / (find_stem(file) + "_features.json")
        features = data_utils.load_json_data(features_pathname)["feature_data"]
        feature_schema = data_utils.load_json_data(feature_schema)
        extractor = BRepNetJsonExtractor(topology, features, feature_schema)
        data = extractor.process()
        output_pathname = output_path / f"{find_stem(file)}.npz"
        data_utils.save_npz_data_without_uvnet_features(output_pathname, data)


def run_worker(worker_args):
    file = worker_args[0]
    output_path = worker_args[1]
    seg_path = worker_args[2]
    feature_schema =  worker_args[3]

    extract_brepnet_data(file, output_path, seg_path, feature_schema)

def extract_brepnet_data_from_json(
        json_path,
        seg_path,
        output_path,
        feature_schema,
        num_workers=1
    ):
    files = [ f for f in json_path.glob("**/*_topology.json")]

    use_many_threads = num_workers > 1
    if use_many_threads:
        worker_args = [(f, output_path) for f in files]
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(run_worker, worker_args), total=len(worker_args)))
    else:
        for file in files:
            extract_brepnet_data(file, seg_path, output_path,feature_schema)

    print("Completed pipeline/extract_feature_data_from_json.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--seg_path", type=str, required=True, help="Path to the seg files to cross check")
    parser.add_argument("--output", type=str, required=True, help="Path to the save validator results")
    parser.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")

    args = parser.parse_args()

    json_path = Path(args.json_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    seg_path = None
    if args.seg_path is not None:
        seg_path = Path(args.seg_path)

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    extract_brepnet_data_from_json(json_path, seg_path, output_path, feature_list_path, args.num_workers)