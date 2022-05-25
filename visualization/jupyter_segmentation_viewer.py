
import numpy as np
import math 
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

from pipeline.entity_mapper import EntityMapper

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

class MultiSelectJupyterRenderer(JupyterRenderer):
    def __init__(self, *args, **kwargs):
        super(MultiSelectJupyterRenderer, self).__init__(*args, **kwargs)
            
    def click(self, value):
        """ called whenever a shape  or edge is clicked
        """
        try:
            obj = value.owner.object
            self.clicked_obj = obj
            if self._current_mesh_selection != obj:
                if obj is not None:
                    self._shp_properties_button.disabled = False
                    self._toggle_shp_visibility_button.disabled = False
                    self._remove_shp_button.disabled = False
                    id_clicked = obj.name  # the mesh id clicked
                    self._current_mesh_selection = obj
                    self._current_selection_material_color = obj.material.color
                    obj.material.color = self._selection_color
                    # selected part becomes transparent
                    obj.material.transparent = True
                    obj.material.opacity = 0.5
                    # get the shape from this mesh id
                    selected_shape = self._shapes[id_clicked]
                    self._current_shape_selection = selected_shape
                # then execute calbacks
                for callback in self._select_callbacks:
                    callback(self._current_shape_selection)
        except Exception as e:
            self.html.value = f"{str(e)}"

class JupyterSegmentationViewer:
    def __init__(self, file_stem, step_folder, seg_folder=None, logit_folder=None):
        self.file_stem = file_stem
        self.step_folder = step_folder
        assert step_folder.exists()
    
        solids = self.load_step()
        assert len(solids) == 1, "Expect only 1 solid"
        self.solid = solids[0]
        self.entity_mapper = EntityMapper(self.solid.topods_shape())

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

        self.color_map = ColorMap()

        self.selection_list = []

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

    def select_face_callback(self, face):
        """
        Callback from the notebook when we select a face
        """
        face_index = self.entity_mapper.face_index(face)
        self.selection_list.append(face_index)

    def view_solid(self):
        """
        Just show the solid.  No need to show any segmentation data
        """
        renderer = MultiSelectJupyterRenderer()
        renderer.register_select_callback(self.select_face_callback)
        renderer.DisplayShape(
            self.solid.topods_shape(), 
            topo_level="Face", 
            render_edges=True, 
            update=True,
            quality=1.0
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

    
    def highlight_faces_with_indices(self, indices):
        indices = set(indices)

        highlighted_color = self.format_color([0, 255, 0])
        other_color = self.format_color([156, 152, 143])

        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            if face_index in indices:
                colors.append(highlighted_color)
            else:
                colors.append(other_color)
        self._display_faces_with_colors(self.solid.faces(), colors)

    def display_faces_with_heatmap(self, values, interval=None):
        if interval is None:
            norm_values = (values - np.min(values))/np.ptp(values)
        else:
            assert len(interval) == 2, "Interval must be length 1"
            interval_length = interval[1]-interval[0]
            assert interval_length > 0, "interval_length must be bigger than 0"
            norm_values = (values - interval[0])/(interval_length)
            norm_values = np.clip(norm_values, 0.0, 1.0)
        
        faces = self.solid.faces()
        colors = []

        for face in faces:
            face_index = self.entity_mapper.face_index(face.topods_shape())
            norm_value = norm_values[face_index]
            color_list = self.color_map.interpolate_color(norm_value)
            int_color_list = [int(v) for v in color_list]
            color = self.format_color(int_color_list)
            colors.append(color)

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
                face.topods_shape(), 
                shape_color=face_color, 
                render_edges=True, 
                edge_color="#000000",
                quality=1.0
            )
            output.append(result)

        # Add the output data to the pickable objects or nothing get rendered
        for elem in output:
            renderer._displayed_pickable_objects.add(elem)                                         

        # Now display the scene
        renderer.Display()