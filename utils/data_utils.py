import json
import numpy as np

def load_json_data(pathname):
    """Load data from a json file"""
    with open(pathname, encoding='utf8') as data_file:
        return json.load(data_file)
        

def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

def save_npz_data_without_uvnet_features(output_pathname, data):
    num_faces = data["face_features"].shape[0]
    num_coedges = data["coedge_features"].shape[0]

    dummy_face_point_grids = np.zeros((num_faces, 10, 10, 7))
    dummy_coedge_point_grids = np.zeros((num_coedges, 10, 12))
    dummy_coedge_lcs = np.zeros((num_coedges, 4, 4))
    dummy_coedge_scale_factors = np.zeros((num_coedges))
    dummy_coedge_reverse_flags = np.zeros((num_coedges))
    np.savez(
        output_pathname, 
        face_features = data["face_features"],
        face_point_grids = dummy_face_point_grids,
        edge_features = data["edge_features"],
        coedge_features = data["coedge_features"], 
        coedge_point_grids = dummy_coedge_point_grids,
        coedge_lcs = dummy_coedge_lcs,
        coedge_scale_factors = dummy_coedge_scale_factors,
        coedge_reverse_flags = dummy_coedge_reverse_flags,
        next = data["coedge_to_next"],
        mate = data["coedge_to_mate"],
        face = data["coedge_to_face"],
        edge = data["coedge_to_edge"],
        savez_compressed = True
    ) 

def load_npz_data(npz_file):
    with np.load(npz_file) as data:
        npz_data = {
            "face_features": data["face_features"],
            "face_point_grids": data["face_point_grids"],
            "edge_features": data["edge_features"],
            "coedge_features": data["coedge_features"], 
            "coedge_point_grids": data["coedge_point_grids"],
            "coedge_lcs": data["coedge_lcs"],
            "coedge_scale_factors": data["coedge_scale_factors"],
            "coedge_reverse_flags": data["coedge_reverse_flags"],
            "coedge_to_next": data["next"], 
            "coedge_to_mate": data["mate"], 
            "coedge_to_face": data["face"], 
            "coedge_to_edge": data["edge"]
        }
    return npz_data


def load_labels(label_pathname):
    labels = np.loadtxt(label_pathname, dtype=np.int64)
    if labels.ndim == 0:
        labels = np.expand_dims(labels, 0)
    return labels