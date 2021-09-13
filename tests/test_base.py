# System
from pathlib import Path
import shutil
import tempfile
import unittest

import utils.data_utils as data_utils

class TestBase(unittest.TestCase):

    def working_dir(self):
        temp_folder = Path(tempfile.gettempdir())
        self.assertTrue(temp_folder.exists())
        working = temp_folder / "brepnet_test_working_dir"
        if not working.exists():
            working.mkdir()
        return working

    def data_dir(self):
        return Path(__file__).parent / "test_data"

    def create_dummy_options(self, dataset_file, dataset_dir, input_features):
        class DummyOptions: pass
        opts = DummyOptions()
        opts.kernel = self.kernel_file()
        opts.input_features = input_features
        opts.dataset_file =  dataset_file
        opts.dataset_dir =  dataset_dir
        opts.label_dir = self.label_dir()
        return opts


    def kernel_file(self):
        return self.parent_dir() / "kernels/simple_edge.json"

    def feature_list_file(self):
        return self.parent_dir() / "feature_lists/all.json"

    def parent_dir(self):
        """
        Find the main BRepNet folder from the path of this test file
        """
        return Path(__file__).parent.parent

    def label_dir(self):
        return self.working_dir()

    def remove_cache_folder(self, data_dir):
        cache_dir = data_dir / "cache"
        self.remove_folder(cache_dir)

    def remove_folder(self, dir):
        shutil.rmtree(dir, ignore_errors=True)