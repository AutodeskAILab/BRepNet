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

def load_npz_data(npz_file):
    with np.load(npz_file) as data:
        assert len(data) == 8
        npz_data = {
            "face_features": data["arr_0"],
            "edge_features": data["arr_1"],
            "coedge_features": data["arr_2"], 
            "coedge_to_next": data["arr_3"], 
            "coedge_to_mate": data["arr_4"], 
            "coedge_to_face": data["arr_5"], 
            "coedge_to_edge": data["arr_6"]
        }
    return npz_data


def load_labels(label_pathname):
    labels = np.loadtxt(label_pathname, dtype=np.int64)
    if labels.ndim == 0:
        labels = np.expand_dims(labels, 0)
    return labels