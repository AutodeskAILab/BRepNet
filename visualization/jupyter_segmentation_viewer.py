
import numpy as np
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

from occwl.io import load_step

class JupyterSegmentationViewer:
    def __init__(self, step_folder, seg_folder):
        self.step_folder = step_folder
        assert step_folder.exists()

        self.seg_folder = seg_folder
        assert seg_folder.exists()
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

    def load_step(self, file_stem):
        step_filename = self.step_folder / (file_stem + ".step")
        if not step_filename.exists():
            step_filename = self.step_folder / (file_stem + ".stp")
        assert step_filename.exists()
        return load_step(step_filename)

    def load_segmentation(self, file_stem):
        """
        Load the seg file
        """
        seg_pathname = self.seg_folder / (file_stem + ".seg")
        return np.loadtxt(seg_pathname, dtype=np.uint64)


    def view_segmentation(self, file_stem):
        """
        View the initial segmentation of this file
        """
        solid = self.load_step(file_stem)[0]
        face_segmentation = self.load_segmentation(file_stem)
        renderer = JupyterRenderer()
        #renderer.DisplayShape(solid.topods_solid(), topo_level="Face", shape_color="#abdda4")
        #renderer

        output = []
        for face, segment_index in zip(solid.faces(), face_segmentation):
            face_color = self.format_color(self.bit8_colors[segment_index])
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
        
    