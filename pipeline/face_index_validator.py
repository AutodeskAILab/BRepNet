"""
This utility class can be used to cross check STEP data against
the triangle mesh and segmentation labels in the Fusion Gallery
dataset.  

The triangles for the model are loaded from the OBJ file and 
the face indices of each triangle are loaded from the fidx
file.  The checker expects the face index to be converted to
a color which is set on each face.

The algorithm performs the following checks.

  - Are the number of faces in the Fusion data the same as the
    number of faces in the solid read from step.  If not then
    the check fails

  - Are the faces in the same order as the colors suggest they should
    be.  If this check passes then we believe that the model is OK.

  - In the case where the colors and face order disagrees then we cross
    check against the box of the triangles.  If the boxes differ by a
    large factor then we reject the file as the labels are not 
    guaranteed to correspond to the correct face.
"""

import igl
import numpy as np
import math

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import (TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL)

from OCC.Core.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDF import TDF_LabelSequence

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.XCAFDoc import (XCAFDoc_DocumentTool, XCAFDoc_ColorGen)

from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)

class FaceIndexValidator():
    def __init__(self, step_file, mesh_dir):
        self.step_file = step_file
        self.mesh_dir = mesh_dir

    def validate(self):
        """
        Validate that the faces in the given STEP file map with the face
        indices as defined by the OBJ meshes extracted with the dataset.
        """
        face_boxes = self.find_face_boxes(self.step_file.stem)
        if face_boxes is None:
            print(f"{self.step_file.stem} missing face")
            return False
     
        parts, face_map = self.load_parts_and_fusion_indices_step_file(self.step_file)
        if len(parts) != 1:
            print(f"{self.step_file} has {len(parts)} parts")
            return False

        # To map between Fusion and STEP data we need the number of faces 
        # to be identical
        if not len(face_boxes) == len(face_map):
            print(f"In fusion {len(face_boxes)} faces.  In step {len(face_map)} faces")
            return False

        for part in parts:
            if not self.check_part(part, face_map, face_boxes):
                return False

        return True


    def check_part(self, part, face_map, face_boxes):
        """
        Check the part against the face boxes loaded which were
        found from the Fusion triangle file.
        """

        # We need to compute triangles for the part
        # or the box will be much bigger than the face
        mesh = BRepMesh_IncrementalMesh(
            part, 0.1, True
        )
        mesh.Perform()

        top_exp = TopologyExplorer(part)
        faces = top_exp.faces()
        face_index_ok = True
        for face_idx, face in enumerate(faces):
            # Find the box of the triangles, scaled to the same units 
            # as the mesh data.
            bscaled = self.get_box_from_tris(face)

            if not face in face_map:
                print("Face missing from face map")
                return False

            fusion_face_index = face_map[face]
            
            if fusion_face_index != face_idx:
                # This is the case where the color in the STEP face doesn't
                # agree with the index of the face in the STEP file.  We have to
                # trust one or the other and the index in the STEP is actually 
                # more reliable, but we cross check this against the bounding box of
                # the triangles from Fusion to make sure
                face_index_ok = False

            if not face_index_ok:
                fusion_face_box = face_boxes[face_idx]
                if fusion_face_box.IsVoid():
                    print("fusion_face_box is void")
                    return False
                
                if bscaled.IsVoid():
                    print("bscaled box is void")
                    return False

                diag = math.sqrt(bscaled.SquareExtent())
                box_check_ok = self.check_box(fusion_face_box, bscaled, diag/10, "Error exceeds 1/10 of face box")
                if not box_check_ok:
                    print(f"Face index and color do not agree and box check fails!")
                    return False
        return True


    def get_obj_pathname(self, basename):
        """
        Get the pathname of the OBJ file for the Fusion mesh
        """
        return  (self.mesh_dir / basename).with_suffix(".obj")

        
    def get_fidx_pathname(self, basename):
        """
        Get the pathname of the file which gives the face index
        of each triangle in the mesh
        """
        return (self.mesh_dir / basename).with_suffix(".fidx")


    def get_face_triangles(self, face):
        """
        Get the triangles from this face
        """
        tris = []
        verts = []

        face_orientation_wrt_surface_normal = (face.Orientation() == TopAbs_FORWARD)
        location = TopLoc_Location()
        brep_tool = BRep_Tool()
        mesh = brep_tool.Triangulation(face, location)
        if mesh != None:
            # Loop over the triangles
            num_tris = mesh.NbTriangles()
            for i in range(1, num_tris+1):
                index1, index2, index3 = mesh.Triangle(i).Get()

                if face_orientation_wrt_surface_normal:
                    # Same sense
                    tris.append([
                        index1 - 1, 
                        index2 - 1, 
                        index3 - 1
                    ])
                else:
                    # Opposite sense
                    tris.append([
                        index3 - 1, 
                        index2 - 1, 
                        index1 - 1
                    ])

            num_vertices = mesh.NbNodes()
            for i in range(1, num_vertices+1):
                vertex_stp = np.array(list(mesh.Node(i).Coord()))
                vertex_obj = 0.1*vertex_stp
                verts.append(vertex_obj)
        return np.array(verts), np.array(tris)

    
    def get_box_from_tris(self, face):
        """
        Get the box of the face.  We do this by using the triangles
        as other methods appear to give boxes which we too big.
        The box will be scaled from mm units to cm
        """
        verts, tris = self.get_face_triangles(face)
        box = Bnd_Box()
                
        # Scale from mm to cm
        transf = gp_Trsf()
        orig = gp_Pnt(0,0,0)
        transf.SetScale(orig, 0.1)

        for i in range(verts.shape[0]):
            vert = verts[i]
            pt = gp_Pnt(vert[0], vert[1], vert[2])
            box.Add(pt)
        return box


    def check_box(self, fusion_face_box, step_face_box, tol, msg):
        min_in_tol = fusion_face_box.CornerMin().Distance(step_face_box.CornerMin()) < tol
        max_in_tol = fusion_face_box.CornerMax().Distance(step_face_box.CornerMax()) < tol
        if not (min_in_tol or max_in_tol):
            print(msg)
            return False
        return True


    def find_face_boxes(self, basename):
        obj_pathname = self.get_obj_pathname(basename)
        fidx_pathname = self.get_fidx_pathname(basename)
        if not obj_pathname.exists():
            print(f"{obj_pathname} does not exist")
            return None

        if not fidx_pathname.exists():
            print(f"{fidx_pathname} does not exist")
            return None

        v, f = igl.read_triangle_mesh(str(obj_pathname))
        tris_to_faces = np.loadtxt(fidx_pathname, dtype=np.uint64)

        boxes = {}
        for tri_index in range(tris_to_faces.size):
            tri_box = Bnd_Box()
            for ptidx in f[tri_index]:
                point = v[ptidx]
                pt = gp_Pnt(point[0], point[1], point[2])
                tri_box.Add(pt)

            face_index = tris_to_faces[tri_index]
            if not face_index in boxes:
                boxes[face_index] = tri_box
            else:
                boxes[face_index].Add(tri_box.CornerMin())
                boxes[face_index].Add(tri_box.CornerMax())

        # Now turn the dictionary of boxes into an array
        box_arr = []
        for i in range(len(boxes)):
            if not i in boxes:
                return None
            box_arr.append(boxes[i])
        
        return box_arr

    def load_parts_and_fusion_indices_step_file(self, pathname):
        """
        Load a list of parts from a STEP file and also return a
        map from the hash value of the shape to the Fusion face index
        """
        # Code based on 
        # #https://github.com/tpaviot/pythonocc-core/blob/master/src/Extend/DataExchange.py
        assert pathname.exists()

        # Create an handle to a document
        doc = TDocStd_Document(TCollection_ExtendedString("FaceIndexValidator"))

        # Get root assembly
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
        step_reader = STEPCAFControl_Reader()

        # These are the attributes we want to transfer
        step_reader.SetColorMode(True)

        # Read the 
        status = step_reader.ReadFile(str(pathname))
        if status == IFSelect_RetDone:
            try:
                # Transfer the step data into the document
                step_reader.Transfer(doc)

                # Get the top level labels
                labels = TDF_LabelSequence()
                shape_tool.GetFreeShapes(labels)
                
                # Build up the shapes list and face map
                shapes = []
                face_map = {}

                # Loop over the top level items.  This
                # should be a list of labels containing 
                # parts
                for i in range(labels.Length()):
                    label = labels.Value(i+1)

                    # Get the shape.   It should be a solid,
                    # assuming we didn't load an assembly file
                    # which is not currently supported  
                    shape = shape_tool.GetShape(label)
                    if shape.ShapeType() == TopAbs_SOLID:
                        shapes.append(shape)
                    else:
                        print("Root shape is not a solid")

                    # Now loop over the sub-shapes one level down
                    sub_shapes_labels = TDF_LabelSequence()
                    shape_tool.GetSubShapes(label, sub_shapes_labels)
                    for j in range(sub_shapes_labels.Length()):
                        sub_label = sub_shapes_labels.Value(j+1)

                        # Get the shape.  It should be a face                
                        sub_shape = shape_tool.GetShape(sub_label)
                        if sub_shape.ShapeType() != TopAbs_FACE:
                            print("Sub shape is not a face")
                            continue

                        # Get the r, g, b values encoding the Fusion face index
                        c = Quantity_Color(0.5, 0.5, 0.5, Quantity_TOC_RGB) 
                        color_tool.GetColor(sub_label, 0, c)
                        color_tool.GetColor(sub_label, 1, c)
                        color_tool.GetColor(sub_label, 2, c)
                        r = int(c.Red()*256)
                        g = int(c.Green()*256*256)
                        b = int(c.Blue()*256*256*256)
                        recovered_index = r+g+b

                        # Get the hash of the face
                        face_map[sub_shape] = recovered_index
            except:
                print("Step transfer problem")
        else:
            print("Step reading problem.")
            raise AssertionError("Error: can't read file.")

        return shapes, face_map