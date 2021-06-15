
import numpy as np
import math 
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

from occwl.io import load_step

class ColorMap:
    def __init__(self):
        self.color_values = [
            [255, 0, 0],  # Blue
            [0, 255, 0],  # Green
            [0, 0,255]    # Red
        ]

    def interpolate_value(self, a, b, t):
        return (b-a)*t + a

    def interpolate_color(self, t):
        num_colors = len(self.color_values)
        tp = t*(num_colors-1)
        index_before = math.floor(tp)
        index_after = math.ceil(tp)
        tint = tp-index_before
        color = []
        for i in range(3):
            color.append(
                self.interpolate_value(
                    self.color_values[i][index_before], 
                    self.color_values[i][index_after], 
                    tint
                )
            )
        return color

class JupyterSegmentationViewer:
    def __init__(self, file_stem, step_folder, seg_folder=None, logit_folder=None):
        self.file_stem = file_stem
        self.step_folder = step_folder
        assert step_folder.exists()
    
        solids = self.load_step()
        assert len(solids) == 1, "Expect only 1 solid"
        self.solid = solids[0]

        self.seg_folder = seg_folder
        self.logit_folder = logit_folder

        self.bit8_colors = [
            [235, 85, 79],  # ExtrudeSide
            [220, 198, 73], # ExtrudeEnd
            [113, 227, 76], # CutSide
            [0, 226, 124],  # CutEnd
            [23, 213, 221], # Fillet
            [92, 99, 222],  # Chamfer
            [176, 57, 223], # RevolveSide
            [238, 61, 178]  # RevolveEnd
        ]

    def format_color(self, c):
        return '#%02x%02x%02x' % (c[0], c[1], c[2])

    def load_step(self):
        step_filename = self.step_folder / (self.file_stem + ".step")
        if not step_filename.exists():
            step_filename = self.step_folder / (self.file_stem + ".stp")
        assert step_filename.exists()
        return load_step(step_filename)

    def load_segmentation(self):
        """
        Load the seg file
        """
        assert not self.seg_folder is None,  "Must create this object specifying seg_folder"
        assert self.seg_folder.exists(), "The segmentation folder provided doesnt exist"

        seg_pathname = self.seg_folder / (self.file_stem + ".seg")
        return np.loadtxt(seg_pathname, dtype=np.uint64)


    def load_logits(self):
        """
        Load logits file
        """
        assert not self.logit_folder is None,  "Must create this object specifying logit_folder"
        assert self.logit_folder.exists(), "The logit folder provided doesnt exist"
        logit_pathname = self.logit_folder / (self.file_stem + ".logits")
        return np.loadtxt(logit_pathname)


    def view_solid(self):
        """
        Just show the solid.  No need to show any segmentation data
        """
        renderer = JupyterRenderer()
        renderer.DisplayShape(
            self.solid.topods_solid(), 
            topo_level="Face", 
            render_edges=True, 
            update=True
        )


    def view_segmentation(self):
        """
        View the initial segmentation of this file
        """
        face_segmentation = self.load_segmentation()
        self._view_segmentation(face_segmentation)


    def view_predicted_segmentation(self):
        """
        View the segmentation predicted by the network
        """
        logits = self.load_logits()
        face_segmentation = np.argmax(logits, axis=1)
        self._view_segmentation(face_segmentation)


    def view_errors_in_segmentation(self):
        """
        View faces which are correct in green and incorrect in red
        """
        face_segmentation = self.load_segmentation()
        logits = self.load_logits()
        predicted_segmentation = np.argmax(logits, axis=1)
        correct_faces = (face_segmentation == predicted_segmentation)
        correct_color = self.format_color([0, 255, 0])
        incorrect_color = self.format_color([255, 0, 0])
        colors = []
        for prediction in correct_faces:
            if prediction:
                colors.append(correct_color)
            else:
                colors.append(incorrect_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def view_faces_for_segment(self, segment_index, threshold):
        logits = self.load_logits()
        logits_for_segment = logits[:,segment_index]
        faces_of_segment = logits_for_segment > threshold
        highlighted_color = self.format_color([0, 255, 0])
        other_color = self.format_color([156, 152, 143])
        colors = []
        for prediction in faces_of_segment:
            if prediction:
                colors.append(highlighted_color)
            else:
                colors.append(other_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def _view_segmentation(self, face_segmentation):
        colors = []
        for segment in face_segmentation:
            color = self.format_color(self.bit8_colors[segment])
            colors.append(color)
        self._display_faces_with_colors(self.solid.faces(), colors)


    def _display_faces_with_colors(self, faces, colors):
        """
        Display the solid with each face colored
        with the given color
        """
        renderer = JupyterRenderer()
        output = []
        for face, face_color in zip(faces, colors):
            result = renderer.AddShapeToScene(
                face.topods_face(), 
                shape_color=face_color, 
                render_edges=True, 
                edge_color="#000000"
            )
            output.append(result)

        # Add the output data to the pickable objects or nothing get rendered
        for elem in output:
            renderer._displayed_pickable_objects.add(elem)                                         

        # Now display the scene
        renderer.Display()