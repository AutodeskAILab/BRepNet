"""
Extract feature data from a step file using Open Cascade
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)

from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid

# BRepNet
from pipeline.entity_mapper import EntityMapper
from pipeline.face_index_validator import FaceIndexValidator
from pipeline.segmentation_file_crosschecker import SegmentationFileCrosschecker

import utils.scale_utils as scale_utils 
from utils.create_occwl_from_occ import create_occwl

class BRepNetExtractor:
    def __init__(self, step_file, output_dir, feature_schema, scale_body=True):
        self.step_file = step_file
        self.output_dir = output_dir
        self.feature_schema = feature_schema
        self.scale_body = scale_body


    def process(self):
        """
        Process the file and extract the derivative data
        """
        # Load the body from the STEP file
        body = self.load_body_from_step()

        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            body = scale_utils.scale_solid_to_unit_box(body)

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)

        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return

        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return
                
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return

        entity_mapper = EntityMapper(body)

        face_features = self.extract_face_features_from_body(body, entity_mapper)
        edge_features = self.extract_edge_features_from_body(body, entity_mapper)
        coedge_features = self.extract_coedge_features_from_body(body, entity_mapper)

        face_point_grids = self.extract_face_point_grids(body, entity_mapper)
        assert face_point_grids.shape[1] == 7
        coedge_point_grids = self.extract_coedge_point_grids(body, entity_mapper)
        assert coedge_point_grids.shape[1] == 12

        coedge_lcs = self.extract_coedge_local_coordinate_systems(body, entity_mapper)
        coedge_reverse_flags = self.extract_coedge_reverse_flags(body, entity_mapper)

        next, mate, face, edge  = self.build_incidence_arrays(body, entity_mapper)

        coedge_scale_factors = self.extract_scale_factors(
            next, 
            mate, 
            face, 
            face_point_grids, 
            coedge_point_grids
        )

        output_pathname = self.output_dir / f"{self.step_file.stem}.npz"
        np.savez(
            output_pathname, 
            face_features=face_features,
            face_point_grids=face_point_grids,
            edge_features=edge_features,
            coedge_point_grids=coedge_point_grids,
            coedge_features=coedge_features,
            coedge_lcs=coedge_lcs,
            coedge_scale_factors=coedge_scale_factors,
            coedge_reverse_flags=coedge_reverse_flags,
            next=next, 
            mate=mate, 
            face=face, 
            edge=edge,
            savez_compressed = True
        )


    def load_body_from_step(self):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    def extract_face_features_from_body(self, body, entity_mapper):
        """
        Extract the face features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        face_features = []
        for face in top_exp.faces():
            assert len(face_features) == entity_mapper.face_index(face)
            face_features.append(self.extract_features_from_face(face))
        return np.stack(face_features)


    def extract_edge_features_from_body(self, body, entity_mapper):
        """
        Extract the edge features from each edge of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        edge_features = []
        for edge in top_exp.edges():
            assert len(edge_features) == entity_mapper.edge_index(edge)
            faces_of_edge = [ Face(f) for f in top_exp.faces_from_edge(edge)]
            edge_features.append(self.extract_features_from_edge(edge, faces_of_edge))
        return np.stack(edge_features)


    def extract_coedge_features_from_body(self, body, entity_mapper):
        """
        Extract the coedge features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        coedge_features = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_features) == entity_mapper.halfedge_index(coedge)
                coedge_features.append(self.extract_features_from_coedge(coedge))

        return np.stack(coedge_features)


    def extract_features_from_face(self, face):
        face_features = []
        for feature in self.feature_schema["face_features"]:
            if feature == "Plane":
                face_features.append(self.plane_feature(face))
            elif feature == "Cylinder":
                face_features.append(self.cylinder_feature(face))
            elif feature == "Cone":
                face_features.append(self.cone_feature(face))
            elif feature == "SphereFaceFeature":
                face_features.append(self.sphere_feature(face))
            elif feature == "TorusFaceFeature":
                face_features.append(self.torus_feature(face))
            elif feature == "FaceAreaFeature":
                face_features.append(self.area_feature(face))
            elif feature == "RationalNurbsFaceFeature":
                face_features.append(self.rational_nurbs_feature(face))
            else:
                assert False, "Unknown face feature"
        return np.array(face_features)
        

    def plane_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Plane:
            return 1.0
        return 0.0


    def cylinder_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Cylinder:
            return 1.0
        return 0.0


    def cone_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Cone:
            return 1.0
        return 0.0


    def sphere_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Sphere:
            return 1.0
        return 0.0


    def torus_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Torus:
            return 1.0
        return 0.0


    def area_feature(self, face):
        geometry_properties = GProp_GProps()
        brepgprop_SurfaceProperties(face, geometry_properties)
        return geometry_properties.Mass()


    def rational_nurbs_feature(self, face):
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_BSplineSurface:
            bspline = surf.BSpline()
        elif surf.GetType() == GeomAbs_BezierSurface:
            bspline = surf.Bezier()
        else:
            bspline = None
        
        if bspline is not None:
            if bspline.IsURational() or bspline.IsVRational():
                return 1.0
        return 0.0


    def extract_features_from_edge(self, edge, faces):
        feature_list = self.feature_schema["edge_features"]
        if "Concave edge" in feature_list or "Convex edge" in feature_list or "Smooth"  in feature_list:
            convexity = self.find_edge_convexity(edge, faces)
        edge_features = []
        for feature in feature_list:
            if feature == "Concave edge":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "Convex edge":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "Smooth":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "EdgeLengthFeature":
                edge_features.append(self.edge_length_feature(edge))
            elif feature == "CircularEdgeFeature":
                edge_features.append(self.circular_edge_feature(edge))
            elif feature == "ClosedEdgeFeature":
                edge_features.append(self.closed_edge_feature(edge))
            elif feature == "EllipticalEdgeFeature":
                edge_features.append(self.elliptical_edge_feature(edge))
            elif feature == "HelicalEdgeFeature":
                edge_features.append(self.helical_edge_feature(edge))
            elif feature == "IntcurveEdgeFeature":
                edge_features.append(self.int_curve_edge_feature(edge))
            elif feature == "StraightEdgeFeature":
                edge_features.append(self.straight_edge_feature(edge))
            elif feature == "HyperbolicEdgeFeature":
                edge_features.append(self.hyperbolic_edge_feature(edge))
            elif feature == "ParabolicEdgeFeature":
                edge_features.append(self.parabolic_edge_feature(edge))
            elif feature == "BezierEdgeFeature":
                edge_features.append(self.bezier_edge_feature(edge))
            elif feature == "NonRationalBSplineEdgeFeature":
                edge_features.append(self.non_rational_bspline_edge_feature(edge))
            elif feature == "RationalBSplineEdgeFeature":
                edge_features.append(self.rational_bspline_edge_feature(edge))
            elif feature == "OffsetEdgeFeature":
                edge_features.append(self.offset_edge_feature(edge))
            else:
                assert False, "Unknown face feature"
        return np.array(edge_features)

    def find_edge_convexity(self, edge, faces):
        edge_data = EdgeDataExtractor(Edge(edge), faces, use_arclength_params=False)
        if not edge_data.good:
            # This is the case where the edge is a pole of a sphere
            return 0.0
        angle_tol_rads = 0.0872664626 # 5 degrees 
        convexity = edge_data.edge_convexity(angle_tol_rads)
        return convexity

    def convexity_feature(self, convexity, feature):
        if feature == "Convex edge":
            return convexity == EdgeConvexity.CONVEX
        if feature == "Concave edge":
            return convexity == EdgeConvexity.CONCAVE
        if feature == "Smooth":
            return convexity == EdgeConvexity.SMOOTH
        assert False, "Unknown convexity"
        return 0.0

    def edge_length_feature(self, edge):
        geometry_properties = GProp_GProps()
        brepgprop_LinearProperties(edge, geometry_properties)
        return geometry_properties.Mass()

    def circular_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Circle:
            return 1.0
        return 0.0

    def closed_edge_feature(self, edge):
        if BRep_Tool().IsClosed(edge):
            return 1.0
        return 0.0

    def elliptical_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Ellipse:
            return 1.0
        return 0.0

    def helical_edge_feature(self, edge):
        # We don't have this feature in Open Cascade
        assert False, "Not implemented for the OCC pipeline"
        return 0.0

    def int_curve_edge_feature(self, edge):
        # We don't have this feature in Open Cascade
        assert False, "Not implemented for the OCC pipeline"
        return 0.0

    def straight_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Line:
            return 1.0
        return 0.0

    def hyperbolic_edge_feature(self, edge):
        if Edge(edge).curve_type() == "hyperbola":
            return 1.0
        return 0.0

    def parabolic_edge_feature(self, edge):
        if Edge(edge).curve_type() == "parabola":
            return 1.0
        return 0.0

    def bezier_edge_feature(self, edge):
        if Edge(edge).curve_type() == "bezier":
            return 1.0
        return 0.0

    def non_rational_bspline_edge_feature(self, edge):
        occwl_edge = Edge(edge)
        if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
            return 1.0
        return 0.0

    def rational_bspline_edge_feature(self, edge):
        occwl_edge = Edge(edge)
        if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
            return 1.0
        return 0.0

    def offset_edge_feature(self, edge):
        if Edge(edge).curve_type() == "offset":
            return 1.0
        return 0.0


    def extract_features_from_coedge(self, coedge):
        coedge_features = []
        for feature in self.feature_schema["coedge_features"]:
            if feature == "ReversedCoEdgeFeature":
                coedge_features.append(self.reversed_edge_feature(coedge))
            else:
                assert False, "Unknown coedge feature"
        return np.array(coedge_features)

    def reversed_edge_feature(self, edge):
        if edge.Orientation() == TopAbs_REVERSED:
            return 1.0
        return 0.0


    def extract_face_point_grids(self, body, entity_mapper):
        """
        Extract a UV-Net point grid for each face.

        Returns a tensor [ num_faces x 7 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast

        """
        face_grids = []
        solid = create_occwl(body)
        for face in solid.faces():
            assert len(face_grids) == entity_mapper.face_index(face.topods_shape())
            face_grids.append(self.extract_face_point_grid(face))
        return np.stack(face_grids)

    def extract_face_point_grid(self, face):
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 7 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast

        """
        num_u=10
        num_v=10

        points = uvgrid(face, num_u, num_v, method="point")
        normals = uvgrid(face, num_u, num_v, method="normal")
        mask = uvgrid(face, num_u, num_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 7 ]
        single_grid = np.concatenate([points, normals, mask], axis=2)
       
        return np.transpose(single_grid, (2, 0, 1))


    def extract_coedge_point_grids(self, body, entity_mapper):
        """
        Extract coedge grids (aligned with the coedge direction).

        The coedge grids will be of size

            [ num_coedges x 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """
        coedge_grids = []
        solid = create_occwl(body)
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_grids) == entity_mapper.halfedge_index(coedge)
                occwl_oriented_edge = Edge(coedge)
                faces = [ f for f in solid.faces_from_edge(occwl_oriented_edge) ]
                coedge_grids.append(self.extract_coedge_point_grid(occwl_oriented_edge, faces))
        return np.stack(coedge_grids)


    def extract_coedge_point_grid(self, coedge, faces):
        """
        Extract a coedge grid (aligned with the coedge direction).

        The coedge grids will be of size

            [ num_coedges x 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """
        num_u = 10
        coedge_data = EdgeDataExtractor(coedge, faces, num_samples=num_u, use_arclength_params=True)
        if not coedge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((12, num_u)) 

        single_grid = np.concatenate(
            [
                coedge_data.points, 
                coedge_data.tangents, 
                coedge_data.left_normals,
                coedge_data.right_normals
            ],
            axis = 1
        )
        return np.transpose(single_grid, (1,0))


    
    def extract_coedge_local_coordinate_systems(self, body, entity_mapper):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.  
        
            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face. 
            - The u_ref will be the coedge tangent at the midpoint.  We get the u_vec by projecting this normal
              to the w_vec
            - The v_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor
 
        Returns a tensor of size [ num_coedges x 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
        """
        solid = create_occwl(body)
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        coedge_lcs = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_lcs) == entity_mapper.halfedge_index(coedge)
                occwl_oriented_edge = Edge(coedge)
                faces = [ f for f in solid.faces_from_edge(occwl_oriented_edge) ]
                coedge_lcs.append(self.extract_coedge_local_coordinate_system(occwl_oriented_edge, faces))

        return np.stack(coedge_lcs)


    def extract_coedge_local_coordinate_system(self, oriented_edge, faces):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.  
        
            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face. 
            - The u_ref will be the coedge tangent at the midpoint.  We get the u_vec by projecting this normal
              to the w_vec
            - The v_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor
 
        Returns a tensor of size [ 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
            [[ u_vec.x  v_vec.x  v_vec.x  orig.x]
             [ u_vec.y  v_vec.y  v_vec.y  orig.y]
             [ u_vec.z  v_vec.z  v_vec.z  orig.z]
             [ 0        0        0        1     ]]
        """
        num_u = 3
        edge_data = EdgeDataExtractor(oriented_edge, faces, num_samples=num_u, use_arclength_params=True)
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # We want to return zeros in this case
            return np.zeros((4,4))
        origin = edge_data.points[1]
        w_vec = edge_data.left_normals[1]

        # Make sure w_vec is a unit vector
        w_vec = w_vec/np.linalg.norm(w_vec)

        # We need to project v_ref normal to w_vec
        v_ref =  edge_data.tangents[1]
        v_vec = self.try_to_project_normal(w_vec, v_ref)
        if v_vec is None:
            # This happens when v_ref is parallel to w_vec.
            # In this case we just pick any v_vec at random
            v_vec = self.any_orthogonal(v_vec)

        u_vec = np.cross(v_vec, w_vec)
        
        # The upper part of the matric should look like this
        # [[ u_vec.x  v_vec.x  v_vec.x  orig.x]
        #  [ u_vec.y  v_vec.y  v_vec.y  orig.y]
        #  [ u_vec.z  v_vec.z  v_vec.z  orig.z]]
        mat_upper = np.transpose(np.stack([u_vec, v_vec, w_vec, origin]))

        mat_lower = np.expand_dims(np.array([0, 0, 0, 1]), axis=0)
        mat = np.concatenate([mat_upper, mat_lower], axis=0)

        return mat


    def try_to_project_normal(self, vec, ref):
        """
        Try to project the vector `ref` normal to vec
        """
        dp = np.dot(vec, ref)
        delta = dp*vec
        normal_dir = ref - delta
        length = np.linalg.norm(normal_dir)
        eps = 1e-7
        if length < eps:
            # Failed to project this vector normal
            return None

        # Return a unit vector
        return normal_dir/length


    def any_orthogonal(self, vec):
        """
        Find any random vector orthogonal to the given vector
        """
        nx = self.try_to_project_normal(vec, np.array([1, 0, 0]))
        if nx is not None:
            return nx
                
        ny = self.try_to_project_normal(vec, np.array([0, 1, 0]))
        if ny is not None:
            return ny

        nz = self.try_to_project_normal(vec, np.array([0, 0, 1]))
        assert nz is not None, f"Something is wrong with vec {vec}.  No orthogonal vector found"
        return nz


    def bounding_box_point_cloud(self, pts):
        assert pts.shape[1] == 3
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return np.array(box)


    def scale_from_point_grids(self, grids):
        assert grids.shape[1] == 7
        face_pts = np.transpose(grids[:, :3, :, :].reshape((3, -1)))
        return self.scale_from_point_cloud(face_pts)


    def scale_from_point_cloud(self, pts):
        assert pts.shape[1] == 3
        bbox = self.bounding_box_point_cloud(pts)
        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        return scale


    def extract_scale_factors(self, next, mate, face, face_point_grids, coedge_point_grids):
        """
        The scale factors which need to be applied to the LCS for scale
        invariance
        """
        identity = np.arange(next.size, dtype=next.dtype)
        prev = np.zeros(next.size, dtype=next.dtype)
        prev[next] = identity

        num_coedges = mate.size

        scales = []
        
        scale_from_faces = True
        if scale_from_faces:
            # Probably very slow
            for i in range(num_coedges):
                left_index = face[i]
                right_index = face[mate[i]]
                left = face_point_grids[left_index]
                right = face_point_grids[right_index]
                scale = self.scale_from_point_grids(np.stack([left, right]))
                scales.append(scale)
        else:
            for i in range(num_coedges):
                # This is a bit like a brepnet kernel.  
                # We use the walks
                # c
                # c->next
                # c->prev
                # c->mate->next
                # c->mate->prev
                coedges = []
                coedges.append(i)
                coedges.append(next[i])
                coedges.append(prev[i])
                coedges.append(next[mate[i]])
                coedges.append(prev[mate[i]])
                points_from_coedges = []
                for coedge_index in coedges:
                    points = coedge_point_grids[coedge_index, :3]
                    num_u = 10
                    assert points_from_coedges.shape[0] == 3
                    assert points_from_coedges.shape[1] == num_u
                    points_from_coedges.append(points)
                points_from_coedges = np.concatenate(points_from_coedges, axis=1)
                points_from_coedges = points_from_coedges.transpose(points_from_coedges)
                scale = self.scale_from_point_cloud(points_from_coedges)
                scales.append(scale)
        return np.array(scales)
            


    
    def extract_coedge_reverse_flags(self, body, entity_mapper):
        """
        The flags for each coedge telling us if it is reversed wrt
        its parent edge.   Notice that when coedge features are 
        created, we need to reverse point ordering, flip tangent directions
        and swap left and right faces based on this flag.
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        reverse_flags = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(reverse_flags) == entity_mapper.halfedge_index(coedge)
                reverse_flags.append(self.reversed_edge_feature(coedge))
        return np.stack(reverse_flags)
    

    def build_incidence_arrays(self, body, entity_mapper):
        oriented_top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        num_coedges = len(entity_mapper.halfedge_map)

        next = np.zeros(num_coedges, dtype=np.uint32)
        mate = np.zeros(num_coedges, dtype=np.uint32)

        # Create the next, pervious and mate permutations
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            first_coedge_index = None
            previous_coedge_index = None
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)

                # Set up the mating coedge
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself.  This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                mate[coedge_index] = mating_coedge_index

                # Set up the next coedge
                if first_coedge_index == None:
                    first_coedge_index = coedge_index
                else:
                    next[previous_coedge_index] = coedge_index
                previous_coedge_index = coedge_index

            # Close the loop
            next[previous_coedge_index] = first_coedge_index

        # Create the arrays from coedge to face
        coedge_to_edge = np.zeros(num_coedges, dtype=np.uint32)
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)
                edge_index = entity_mapper.edge_index(coedge)
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself.  This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                coedge_to_edge[coedge_index] = edge_index
                coedge_to_edge[mating_coedge_index] = edge_index

        # Loop over the faces and make the back 
        # pointers back to the edges
        coedge_to_face = np.zeros(num_coedges, dtype=np.uint32)
        unoriented_top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        for face in unoriented_top_exp.faces():
            face_index = entity_mapper.face_index(face)
            for loop in unoriented_top_exp.wires_from_face(face):
                wire_exp =  TopologyUtils.WireExplorer(loop)
                for coedge in wire_exp.ordered_edges():
                    coedge_index = entity_mapper.halfedge_index(coedge)
                    coedge_to_face[coedge_index] = face_index

        return next, mate, coedge_to_face, coedge_to_edge


    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False

                coedge_set.add(tup)

        return True
        
    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0


    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set


    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set


    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True
            

def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)

def check_face_indices(step_file, mesh_dir):
    if mesh_dir is None:
        # Nothing to check
        return True
    # Check against the given meshes and Fusion labels    
    validator = FaceIndexValidator(step_file, mesh_dir)
    return validator.validate()

def crosscheck_faces_and_seg_file(file, seg_dir):
    seg_pathname = None
    if seg_dir is None:
        # Look to see if the seg file is in the step dir
        step_dir = file.parent
        trial_seg_pathname = step_dir / (file.stem + ".seg")
        if trial_seg_pathname.exists():
            seg_pathname = trial_seg_pathname
    else:
        # We expect to find the segmentation file in the 
        # seg dir
        seg_pathname = seg_dir / (file.stem + ".seg")
        if not seg_pathname.exists():
            print(f"Warning!! Segmentation file {seg_pathname} is missing")
            return False
    
    if seg_pathname is not None:
        checker = SegmentationFileCrosschecker(file, seg_pathname)
        data_ok = checker.check_data()
        if not data_ok:
            print(f"Warning!! Segmentation file {seg_pathname} and step file {file} have different numbers of faces")
        return data_ok
    
    # In the case where we don't know the seg pathname we don't do 
    # any extra checking
    return True

def extract_brepnet_features(file, output_path, feature_schema, mesh_dir, seg_dir):
    if not check_face_indices(file, mesh_dir):
        return
    if not crosscheck_faces_and_seg_file(file, seg_dir):
        return
    extractor = BRepNetExtractor(file, output_path, feature_schema)
    extractor.process()

def run_worker(worker_args):
    file = worker_args[0]
    output_path = worker_args[1]
    feature_schema = worker_args[2]
    mesh_dir = worker_args[3]
    seg_dir = worker_args[4]
    extract_brepnet_features(file, output_path, feature_schema, mesh_dir, seg_dir)

def filter_out_files_which_are_already_converted(files, output_path):
    files_to_convert = []
    for file in files:
        output_file = output_path / (file.stem + ".npz")
        if not output_file.exists():
            files_to_convert.append(file)
    return files_to_convert


def extract_brepnet_data_from_step(
        step_path, 
        output_path, 
        mesh_dir=None,
        seg_dir=None,
        feature_list_path=None,
        force_regeneration=True,
        num_workers=1
    ):
    parent_folder = Path(__file__).parent.parent
    if feature_list_path is None:
        feature_list_path = parent_folder / "feature_lists/all.json"
    feature_schema = load_json(feature_list_path)
    files = [ f for f in step_path.glob("**/*.stp")]
    step_files = [ f for f in step_path.glob("**/*.step")]
    files.extend(step_files)

    if not force_regeneration:
        files = filter_out_files_which_are_already_converted(files, output_path)

    use_many_threads = num_workers > 1
    if use_many_threads:
        worker_args = [(f, output_path, feature_schema, mesh_dir, seg_dir) for f in files]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(run_worker, worker_args), total=len(worker_args)))
    else:
        for file in tqdm(files):
            extract_brepnet_features(file, output_path, feature_schema, mesh_dir, seg_dir)

    gc.collect()
    print("Completed pipeline/extract_feature_data_from_step.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data")
    parser.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument(
        "--mesh_dir", 
        type=str,  
        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels"
    )
    parser.add_argument(
        "--seg_dir", 
        type=str,  
        help="Optionally provide a directory containing segmentation labels seg files."
    )
    args = parser.parse_args()

    step_path = Path(args.step_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    mesh_dir = None
    if args.mesh_dir is not None:
        mesh_dir = Path(args.mesh_dir)

    seg_dir = None
    if args.seg_dir is not None:
        seg_dir = Path(args.seg_dir)

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    extract_brepnet_data_from_step(step_path, output_path, mesh_dir, seg_dir, feature_list_path, args.num_workers)