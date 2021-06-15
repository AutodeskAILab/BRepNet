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

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopExp import topexp
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)

from OCC.Core.BRepBndLib import brepbndlib_AddOptimal

from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face

# BRepNet
from pipeline.entity_mapper import EntityMapper
from pipeline.face_index_validator import FaceIndexValidator
from pipeline.segmentation_file_crosschecker import SegmentationFileCrosschecker

class BRepNetExtractor:
    def __init__(self, step_file, output_dir, feature_schema):
        self.step_file = step_file
        self.output_dir = output_dir
        self.feature_schema = feature_schema


    def process(self):
        """
        Process the file and extract the derivative data
        """
        # Load the body from the STEP file
        body = self.load_body_from_step()

        body = self.scale_body_to_unit_box(body)

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

        next, mate, face, edge  = self.build_incidence_arrays(body, entity_mapper)

        output_pathname = self.output_dir / f"{self.step_file.stem}.npz"
        np.savez(
            output_pathname, 
            face_features,
            edge_features,
            coedge_features, 
            next, 
            mate, 
            face, 
            edge,
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

    def scale_body_to_unit_box(self, body):
        bbox = self.find_box(body)
        xmin = 0.0
        xmax = 0.0
        ymin = 0.0
        ymax = 0.0
        zmin = 0.0
        zmax = 0.0
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        longest_length = dx
        if longest_length < dy:
            longest_length = dy
        if longest_length < dz:
            longest_length = dz

        orig = gp_Pnt(0.0, 0.0, 0.0)
        center = gp_Pnt((xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0, )
        vec_center_to_orig = gp_Vec(center, orig)
        move_to_center = gp_Trsf()
        move_to_center.SetTranslation(vec_center_to_orig)

        scale_trsf = gp_Trsf()
        scale_trsf.SetScale(orig, 2.0/longest_length)
        trsf_to_apply = scale_trsf.Multiplied(move_to_center)
        
        apply_transform = BRepBuilderAPI_Transform(trsf_to_apply)
        apply_transform.Perform(body)

        transformed_body = apply_transform.ModifiedShape(body)

        check_bbox = self.find_box(transformed_body)
        xmin = 0.0
        xmax = 0.0
        ymin = 0.0
        ymax = 0.0
        zmin = 0.0
        zmax = 0.0
        xmin, ymin, zmin, xmax, ymax, zmax = check_bbox.Get()
        temp = 0
        return transformed_body


    def find_box(self, body):
        bbox = Bnd_Box()
        use_triangulation = True
        use_shapetolerance = False
        brepbndlib_AddOptimal(body, bbox, use_triangulation, use_shapetolerance)
        return bbox

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