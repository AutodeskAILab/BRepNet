"""
This utility performs a basic check that the supplied segmentation file 
contains the same number of faces as the STEP data when this is loaded into
Open Cascade
"""
import numpy as np
from occwl.io import load_step

import utils.data_utils as data_utils


class SegmentationFileCrosschecker:
    def __init__(self, step_pathname, seg_pathname):
        """
        Initialize the class with the pathnames for the step file and
        corresponding segmentation file 
        """
        self.step_pathname = step_pathname
        self.seg_pathname = seg_pathname

    def check_data(self):
        if not self.step_pathname.exists():
            return False
        if not self.seg_pathname.exists():
            return False

        # Load the step file and find the number of 
        solids = load_step(self.step_pathname)
        assert len(solids) == 1
        solid = solids[0]
        faces = [ f for f in solid.faces()]
        num_faces = len(faces)

        # Read the seg file and find the number of 
        # face segment indices
        segment_indices = data_utils.load_labels(self.seg_pathname)
        num_face_segment_indices = segment_indices.size 

        # If the number of face segment indices is the same as the number
        # of segments then the check passes
        return num_faces == num_face_segment_indices
