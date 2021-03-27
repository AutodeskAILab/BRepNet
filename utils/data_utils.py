import json

def load_json_data(pathname):
    """Load data from a json file"""
    with open(pathname, encoding='utf8') as data_file:
        return json.load(data_file)
        

def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)