# System
import json
import numpy as np
from pathlib import Path
import unittest
import sys

# Python OCC
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopExp import topexp
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.Extrema import Extrema_ExtFlag_MIN
from OCC.Core.gp import gp_Pnt
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)
                              

from pipeline.entity_mapper import EntityMapper
from pipeline.extract_brepnet_data_from_step import BRepNetExtractor
import utils.data_utils as data_utils

class TestBRepNetExtractor(unittest.TestCase):

    def load_solid_from_step(self, step_file):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    def run_test_on_all_files(self, step_folder, npz_folder):
        step_files = [ f for f in step_folder.glob("**/*.step")]
        stp_files = [ f for f in step_folder.glob("**/*.stp")]
        step_files.extend(stp_files)

        if len(step_files) == 0:
            self.fail("No files in directory")

        if not npz_folder.exists():
            npz_folder.mkdir()
        feature_schema = self.load_feature_schema()
        for file in step_files:
            self.run_tests_on_file(file, npz_folder, feature_schema)
                

    def load_feature_schema(self):
        parent_folder = Path(__file__).parent.parent
        feature_schema_pathname = parent_folder / "feature_lists/all.json"
        with open(feature_schema_pathname, "r") as fp:
            feature_schema = json.load(fp)
            return feature_schema


    def run_tests_on_single_file(self, step_file, npz_folder):
        if not npz_folder.exists():
            npz_folder.mkdir()
        feature_schema = self.load_feature_schema()
        solid = self.run_tests_on_file(step_file, npz_folder, feature_schema)
        return solid


    def run_tests_on_file(self, file, npz_folder, feature_schema):
        npz_pathname = npz_folder / f"{file.stem}.npz"
        if npz_pathname.exists():
            npz_pathname.unlink()

        brepnet_extractor = BRepNetExtractor(
            file, 
            npz_folder, 
            feature_schema,
            scale_body = False  # Don't scale the body for the tests
        )
        brepnet_extractor.process()

        print(f"Running tests for {file.stem}{file.suffix}")
        solid = self.load_solid_from_step(file)
        if not npz_pathname.exists():
            print("The npz file was not generated and cannot be tested")
            return

        data = data_utils.load_npz_data(npz_pathname)
        self.check_topology(solid, data)
        self.check_topology_and_feature_data_consistent(data)
        return solid


    def check_topology(self, solid, data):
        self.check_next_is_permutation(data["coedge_to_next"])
        self.check_mate_mate(data["coedge_to_mate"])

        entity_mapper = EntityMapper(solid)
        self.check_faces_consistent(solid, entity_mapper, data["coedge_to_face"])
        self.check_edges_consistent(solid, entity_mapper, data["coedge_to_edge"])
        self.check_loops_consistent(solid, entity_mapper, data["coedge_to_next"])
        self.check_face_adjacency_vs_num_shells(solid, entity_mapper, data)
        
    def check_topology_and_feature_data_consistent(self, data):
        num_faces_from_features = data["face_features"].shape[0]
        num_edges_from_features = data["edge_features"].shape[0]
        num_coedges_from_features = data["coedge_features"].shape[0]

        self.assertEqual(num_coedges_from_features, data["coedge_to_next"].size)
        self.assertEqual(num_coedges_from_features, data["coedge_to_mate"].size)
        self.assertEqual(num_coedges_from_features, data["coedge_to_face"].size)
        self.assertEqual(num_coedges_from_features, data["coedge_to_edge"].size)

        edge_set = set()
        for edge_index in data["coedge_to_edge"]:
            edge_set.add(edge_index)
        num_edges_from_top = len(edge_set)
        for index in range(num_edges_from_top):
            # Check the set contains 0, 1, 2 .. num_edges_from_top-1
            self.assertIn(index, edge_set)
        self.assertEqual(num_edges_from_top, num_edges_from_features)
        
        face_set = set()
        for face_index in data["coedge_to_face"]:
            face_set.add(face_index)
        num_faces_from_top = len(face_set)
        for index in range(num_faces_from_top):
            # Check the set contains 0, 1, 2 .. num_faces_from_top-1
            self.assertIn(index, face_set)
        self.assertEqual(num_faces_from_top, num_faces_from_features)


    def check_next_is_permutation(self, next):
        """
        We want to check that the array of indices in 
        next is actually a permutation.  i.e. there is
        a bijection (1:1) mapping between the elements
        """
        indices_mapped_onto = set()
        num_coedges = next.size
        for next_coedge_index in next:
            # No index should be out of range
            self.assertGreaterEqual(next_coedge_index, 0)
            self.assertLess(next_coedge_index, num_coedges)

            # Each index should be in the array only once
            self.assertNotIn(next_coedge_index, indices_mapped_onto)
            indices_mapped_onto.add(next_coedge_index)

        # Check we have the correct number of coedges in the end
        self.assertEqual(len(indices_mapped_onto), num_coedges)


    def check_mate_mate(self, mate):
        for coedge_index, mating_index in enumerate(mate):
            mate_mate = mate[mating_index]
            self.assertEqual(coedge_index, mate_mate)


    def check_loops_consistent(self, solid, entity_mapper, next):
        loops, coedges_to_loop = self.build_loops_data(next)
        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
        num_loops_in_solid = top_exp.number_of_wires()
        self.assertEqual(num_loops_in_solid, len(loops))

        num_faces_in_solid = top_exp.number_of_faces()
        self.assertLessEqual(num_faces_in_solid, len(loops))

        for loop_from_solid in top_exp.wires():
            is_first_coedge = True
            prev_coedge_index = None
            first_coedge_index = None
            wire_exp = TopologyUtils.WireExplorer(loop_from_solid)
            for coedge_from_solid in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge_from_solid)
                if is_first_coedge:
                    is_first_coedge = False
                    loop_index = coedges_to_loop[coedge_index]
                    first_coedge_index = coedge_index
                else:
                    # Check the "definition" of the next array
                    self.assertEqual(coedge_index,  next[prev_coedge_index])

                    # Check that every coedge in the loop is in the loop
                    # data recovered from the rings from the permutation
                    self.assertIn(coedge_index, loops[loop_index])
                prev_coedge_index =  coedge_index


    def build_loops_data(self, next):
        coedges_to_loop = {}
        next_loop_index = 0
        for coedge_index in range(next.size):
            if coedge_index in coedges_to_loop:
                continue
            
            # This is the start of a new loop
            loop_index = next_loop_index
            coedges_to_loop[coedge_index] = loop_index
            next_loop_index += 1

            # Now navigate around the loop until we get back to 
            # the coedge with coedge_index
            around_loop = next[coedge_index]
            while around_loop != coedge_index:
                self.assertNotIn(around_loop, coedges_to_loop)
                coedges_to_loop[around_loop] = loop_index
                around_loop = next[around_loop]

        self.assertEqual(len(coedges_to_loop), next.size)

        loops = [ set() for i in range(next_loop_index)]
        for coedge_index in coedges_to_loop:
            loop_index = coedges_to_loop[coedge_index]

            # A given coedge can't be in a loop more than once
            self.assertNotIn(coedge_index, loops[loop_index])
            loops[loop_index].add(coedge_index)

        # Check for empty loops
        for loop in loops:
            self.assertGreater(len(loop), 0)

        return loops, coedges_to_loop


    def check_faces_consistent(self, solid, entity_mapper, coedges_to_faces):
        """
        Check the mapping from coedges to faces is consistent
        """
        face_to_coedges = self.build_face_to_coedges_map(coedges_to_faces)

        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
        self.assertEqual(len(face_to_coedges), top_exp.number_of_faces())

        for face in top_exp.faces():
            face_index = entity_mapper.face_index(face)
            for loop in top_exp.wires_from_face(face):
                wire_exp = TopologyUtils.WireExplorer(loop)
                for coedge in wire_exp.ordered_edges():
                    coedge_index = entity_mapper.halfedge_index(coedge)
                    self.assertEqual(face_index, coedges_to_faces[coedge_index])
                    self.assertIn(coedge_index, face_to_coedges[face_index])


    def check_edges_consistent(self, solid, entity_mapper, coedges_to_edges):
        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=False)
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_index = entity_mapper.edge_index(edge)
                coedge_index = entity_mapper.halfedge_index(edge)
                mate = edge.Reversed()
                if entity_mapper.halfedge_exists(mate):
                    mate_index = entity_mapper.halfedge_index(mate)
                else:
                    mate_index = coedge_index
                self.assertEqual(edge_index, coedges_to_edges[coedge_index])
                self.assertEqual(edge_index, coedges_to_edges[mate_index])


    def build_face_to_coedges_map(self, coedges_to_faces):
        faces_to_coedges = {}
        for coedge_index, face_index in enumerate(coedges_to_faces):
            if not face_index in faces_to_coedges:
                faces_to_coedges[face_index] = set()
            faces_to_coedges[face_index].add(coedge_index)
        return faces_to_coedges


    def check_face_adjacency_vs_num_shells(self, solid, entity_mapper, data):
        """
        Basically we want to check that the mating coedges are set up
        correctly.  It's quite hard to get the mate information
        out of Open Cascade, so as a cross check we can ensure that the 
        face adjacency graph is connected.  There should be as many
        connected components as we have shells in the solid.
        """
        coedge_to_face = data["coedge_to_face"]
        coedge_to_mate = data["coedge_to_mate"]

        visited_faces = set()
        face_to_coedges = self.build_face_to_coedges_map(coedge_to_face)
        num_faces = len(face_to_coedges)

        faces_in_shells = []

        num_shells = 0

        last_face = 0
        while len(visited_faces) < num_faces:
            num_shells += 1

            # Find the next face to use.  This will be a 
            # face in a shell which we haven't yet flooded
            # over
            for next_face in range(last_face, num_faces):
                if not next_face in visited_faces:
                    break

            faces_in_shell = []
            last_face = next_face
            faces_to_visit = [ next_face ]
            while len(faces_to_visit) > 0:
                current_face = faces_to_visit.pop()
                if current_face in visited_faces:
                    continue
                faces_in_shell.append(current_face)
                visited_faces.add(current_face)
                for coedge in face_to_coedges[current_face]:
                    mate = coedge_to_mate[coedge]
                    mate_face = coedge_to_face[mate]
                    if not mate_face in visited_faces:
                        faces_to_visit.append(mate_face)
            faces_in_shells.append(faces_in_shell)

        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
        self.assertEqual(num_shells, top_exp.number_of_shells())



    def find_distance_to_entity(self, ent, point):
        # Folowing https://dev.opencascade.org/content/how-retrieve-nearest-face-shape-given-gppnt
        # Create a vertex from the point
        occ_point = gp_Pnt(point[0], point[1], point[2])
        vertex_maker = BRepBuilderAPI_MakeVertex(occ_point)
        vertex = vertex_maker.Shape()
        dist_shape_shape = BRepExtrema_DistShapeShape(
            vertex, 
            ent,
            Extrema_ExtFlag_MIN
        )
        ok = dist_shape_shape.Perform()
        self.assertTrue(ok)
        return dist_shape_shape.Value()

         
    def find_closest_ent_in_list(self, ents, point):
        closest_dist_yet = sys.float_info.max
        closest_ent = None
        for ent in ents:
            dist_to_point = self.find_distance_to_entity(ent, point)
            if dist_to_point < closest_dist_yet:
                closest_ent = ent
                closest_dist_yet = dist_to_point
        return closest_ent


    def find_closest_face(self, solid, point):
        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
        return self.find_closest_ent_in_list(top_exp.faces(), point)


    def find_closest_edge(self, solid, point):
        top_exp = TopologyUtils.TopologyExplorer(solid, ignore_orientation=True)
        return self.find_closest_ent_in_list(top_exp.edges(), point)


    def run_tests_on_simple_solid(self, step_filename):
        step_folder = Path(__file__).parent / "test_data/simple_solids"
        npz_folder = Path(__file__).parent / "test_working_dir"
        step_file = step_folder / step_filename
        solid  = self.run_tests_on_single_file(step_file, npz_folder)
        return (npz_folder / step_filename).with_suffix(".npz"), solid


    def check_num_faces(self, data, expected_num_faces):
        self.assertEqual(data["face_features"].shape[0], expected_num_faces)
        face_index_set = set()
        for index in data["coedge_to_face"]:
            face_index_set.add(index)
        for index in range(expected_num_faces):
            self.assertIn(index, face_index_set)
        self.assertEqual(len(face_index_set), expected_num_faces)

        
    def check_num_edges(self, data, expected_num_edges):
        self.assertEqual(data["edge_features"].shape[0], expected_num_edges)
        edge_index_set = set()
        for index in data["coedge_to_edge"]:
            edge_index_set.add(index)
        for index in range(expected_num_edges):
            self.assertIn(index, edge_index_set)
        self.assertEqual(len(edge_index_set), expected_num_edges)

    def check_num_coedges(self, data, expected_num_coedges):
        self.assertEqual(data["coedge_features"].shape[0], expected_num_coedges)
        self.assertEqual(data["coedge_to_next"].size, expected_num_coedges)
        self.assertEqual(data["coedge_to_mate"].size, expected_num_coedges)
        self.assertEqual(data["coedge_to_face"].size, expected_num_coedges)
        self.assertEqual(data["coedge_to_edge"].size, expected_num_coedges)


    def check_loop_hist(self, data, expected_loop_hist):
        loops, coedges_to_loop = self. build_loops_data(data["coedge_to_next"])
        found_loop_hist = {}
        for loop in loops:
            num_coedges = len(loop)
            if not num_coedges in found_loop_hist:
                found_loop_hist[num_coedges] = 0
            found_loop_hist[num_coedges] += 1
        
        self.assertEqual(len(expected_loop_hist), len(found_loop_hist))
        for key in expected_loop_hist:
            self.assertIn(key, found_loop_hist)
            self.assertEqual(expected_loop_hist[key], found_loop_hist[key])
            self.assertEqual(len(expected_loop_hist), len(found_loop_hist))

    def find_index_in_feature_schema(self, key, feature_schema):
        return feature_schema.index(key)

    def check_num_planes(self, data, feature_schema, expected_num_planes):
        index = self.find_index_in_feature_schema(
            "Plane", 
            feature_schema["face_features"]
        )
        num_planes = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_planes, expected_num_planes)


    def check_num_cylinders(self, data, feature_schema, expected_num_cylinders):
        index = self.find_index_in_feature_schema(
            "Cylinder", 
            feature_schema["face_features"]
        )
        num_cylinders = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_cylinders, expected_num_cylinders)

    def check_num_cones(self, data, feature_schema, expected_num_cones):
        index = self.find_index_in_feature_schema(
            "Cone", 
            feature_schema["face_features"]
        )
        num_cones = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_cones, expected_num_cones)

    def check_num_torus(self, data, feature_schema, expected_num_torus):
        index = self.find_index_in_feature_schema(
            "TorusFaceFeature", 
            feature_schema["face_features"]
        )
        num_torus = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_torus, expected_num_torus)

    def check_num_sphere(self, data, feature_schema, expected_num_sphere):
        index = self.find_index_in_feature_schema(
            "SphereFaceFeature", 
            feature_schema["face_features"]
        )
        num_sphere = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_sphere, expected_num_sphere)

    def check_num_rational_nurbs(self, data, feature_schema, expected_num_rational_nurbs):
        index = self.find_index_in_feature_schema(
            "RationalNurbsFaceFeature", 
            feature_schema["face_features"]
        )
        num_rational_nurbs = data["face_features"].sum(axis=0)[index]
        self.assertEqual(num_rational_nurbs, expected_num_rational_nurbs)

    def check_expect_face_area(self, data, feature_schema, expected_face_area):
        feature_index = self.find_index_in_feature_schema(
            "FaceAreaFeature", 
            feature_schema["face_features"]
        )
        found_it = False
        area_tol = 1.0
        face_features = data["face_features"]
        for face_index in range(face_features.shape[0]):
            face_area = face_features[face_index, feature_index]
            if abs(face_area-expected_face_area) < area_tol:
                found_it = True
        self.assertTrue(found_it)


    def check_num_concave(self, data, feature_schema, expected_num_concave):
        index = self.find_index_in_feature_schema(
            "Concave edge", 
            feature_schema["edge_features"]
        )
        num_concave = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_concave, expected_num_concave)


    def check_num_convex(self, data, feature_schema, expected_num_convex):
        index = self.find_index_in_feature_schema(
            "Convex edge", 
            feature_schema["edge_features"]
        )
        num_convex = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_convex, expected_num_convex)

    def check_num_smooth(self, data, feature_schema, expected_num_smooth):
        index = self.find_index_in_feature_schema(
            "Smooth", 
            feature_schema["edge_features"]
        )
        num_smooth = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_smooth, expected_num_smooth)


    def check_expect_edge_length(self, data, feature_schema, expected_edge_length):
        feature_index = self.find_index_in_feature_schema(
            "EdgeLengthFeature", 
            feature_schema["edge_features"]
        )
        found_it = False
        area_tol = 1.0
        edge_features = data["edge_features"]
        for edge_index in range(edge_features.shape[0]):
            edge_length = edge_features[edge_index, feature_index]
            if abs(edge_length-expected_edge_length) < area_tol:
                found_it = True
        self.assertTrue(found_it)


    def check_num_circular_edge(self, data, feature_schema, expected_num_circular_edge):
        index = self.find_index_in_feature_schema(
            "CircularEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_circular_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_circular_edge, expected_num_circular_edge)


    def check_num_closed_edge(self, data, feature_schema, expected_num_closed_edge):
        index = self.find_index_in_feature_schema(
            "ClosedEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_closed_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_closed_edge, expected_num_closed_edge)


    def check_num_elliptical_edge(self, data, feature_schema, expected_num_elliptical_edge):
        index = self.find_index_in_feature_schema(
            "EllipticalEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_elliptical_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_elliptical_edge, expected_num_elliptical_edge)


    def check_num_helical_edge(self, data, feature_schema, expected_num_helical_edge):
        index = self.find_index_in_feature_schema(
            "HelicalEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_helical_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_helical_edge, expected_num_helical_edge)

    def check_num_intcurve_edge(self, data, feature_schema, expected_num_intcurve_edge):
        index = self.find_index_in_feature_schema(
            "IntcurveEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_intcurve_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_intcurve_edge, expected_num_intcurve_edge)

    def check_num_straight_edge(self, data, feature_schema, expected_num_straight_edge):
        index = self.find_index_in_feature_schema(
            "StraightEdgeFeature", 
            feature_schema["edge_features"]
        )
        num_straight_edge = data["edge_features"].sum(axis=0)[index]
        self.assertEqual(num_straight_edge, expected_num_straight_edge)


    def check_closest_features(self, data, solid, feature_schema, expected_closest_features):
        """
        Check a specific feature value for the closest face or edge 
        to a given point
        """
        entity_mapper = EntityMapper(solid)
        for expected_feature in expected_closest_features:
            entity_type = expected_feature[0]
            point = expected_feature[1]
            feature_type = expected_feature[2]
            expected_feature_value = expected_feature[3]

            if entity_type == "face_features":
                face = self.find_closest_face(solid, point)
                ent_index = entity_mapper.face_index(face)
            elif entity_type == "edge_features":
                edge = self.find_closest_edge(solid, point)
                ent_index = entity_mapper.edge_index(edge)
            else:
                assert False, "Unsupported feature type"
            feature_index = self.find_index_in_feature_schema(
                feature_type, 
                feature_schema[entity_type]
            )
            feature_value = data[entity_type][ent_index, feature_index]
            self.assertEqual(feature_value, expected_feature_value)
            

    def check_feature_stats(
            self, 
            npz_file,
            solid,
            expected_data
        ):
        data = data_utils.load_npz_data(npz_file)
        feature_schema = self.load_feature_schema()
        if "num_faces" in expected_data:
            self.check_num_faces(data, expected_data["num_faces"])
        if "num_edges" in expected_data:
            self.check_num_edges(data, expected_data["num_edges"])
        if "num_coedges" in expected_data:
            self.check_num_coedges(data, expected_data["num_coedges"])
        if "loop_hist" in expected_data:
            self.check_loop_hist(data, expected_data["loop_hist"])
        if "num_planes" in expected_data:
            self.check_num_planes(data, feature_schema, expected_data["num_planes"])
        if "num_cylinders" in expected_data:
            self.check_num_cylinders(data, feature_schema, expected_data["num_cylinders"])
        if "num_cones" in expected_data:
            self.check_num_cones(data, feature_schema, expected_data["num_cones"])
        if "num_torus" in expected_data:
            self.check_num_torus(data, feature_schema, expected_data["num_torus"])
        if "num_sphere" in expected_data:
            self.check_num_sphere(data, feature_schema, expected_data["num_sphere"])
        if "num_rational_nurbs" in expected_data:
            self.check_num_rational_nurbs(data, feature_schema, expected_data["num_rational_nurbs"])
        if "expect_face_area" in expected_data:
            self.check_expect_face_area(data, feature_schema, expected_data["expect_face_area"])
        if "num_concave" in expected_data:
            self.check_num_concave(data, feature_schema, expected_data["num_concave"])
        if "num_convex" in expected_data:
            self.check_num_convex(data, feature_schema, expected_data["num_convex"])
        if "num_smooth" in expected_data:
            self.check_num_smooth(data, feature_schema, expected_data["num_smooth"])
        if "expect_edge_length" in expected_data:
            self.check_expect_edge_length(data, feature_schema, expected_data["expect_edge_length"])
        if "num_circular_edge" in expected_data:
            self.check_num_circular_edge(data, feature_schema, expected_data["num_circular_edge"])
        if "num_closed_edge" in expected_data:
            self.check_num_closed_edge(data, feature_schema, expected_data["num_closed_edge"])
        if "num_elliptical_edge" in expected_data:
            self.check_num_elliptical_edge(data, feature_schema, expected_data["num_elliptical_edge"])
        if "num_helical_edge" in expected_data:
            self.check_num_helical_edge(data, feature_schema, expected_data["num_helical_edge"])
        if "num_intcurve_edge" in expected_data:
            self.check_num_intcurve_edge(data, feature_schema, expected_data["num_intcurve_edge"])
        if "num_straight_edge" in expected_data:
            self.check_num_straight_edge(data, feature_schema, expected_data["num_straight_edge"])
        if "closest_features" in expected_data:
            self.check_closest_features(data, solid, feature_schema, expected_data["closest_features"])

        

    def test_block(self):
        npz_file, solid = self.run_tests_on_simple_solid("block.step")
        expected_data = {
            "num_faces": 6,
            "num_edges": 12,
            "num_coedges": 24,
            "loop_hist": {
                4: 6
            },
            "num_planes": 6,
            "num_cylinders": 0,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "expect_face_area": 2500,
            "num_concave": 0,
            "num_convex": 12,
            "num_smooth": 0,
            "expect_edge_length": 50,
            "num_circular_edge": 0,
            "num_closed_edge": 0,
            "num_elliptical_edge": 0,
            "num_straight_edge": 12,
        }
        self.check_feature_stats(npz_file, solid, expected_data)
        
    def test_block_fillet1(self):
        npz_file, solid = self.run_tests_on_simple_solid("block_fillet1.step")
        expected_data = {
            "num_faces": 7,
            "num_edges": 15,
            "num_coedges": 30,
            "loop_hist": {
                4: 5,
                5: 2
            },
            "num_planes": 6,
            "num_cylinders": 1,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "expect_face_area": 1750,
            "num_concave": 0,
            "num_convex": 13,
            "num_smooth": 2,
            "expect_edge_length": 35,
            "num_circular_edge": 2,
            "num_closed_edge": 0,
            "num_elliptical_edge": 0,
            "num_straight_edge": 13,
            "closest_features": [
                [ "face_features", [23.49, 8.62, 38.58], "Plane", 0],
                [ "face_features", [23.49, 8.62, 38.58], "Cylinder", 1]
            ]
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_block_fillet3(self):
        npz_file, solid = self.run_tests_on_simple_solid("block_fillet3.step")
        expected_data = {
            "num_faces": 10,
            "num_edges": 21,
            "num_coedges": 42,
            "loop_hist": {
                3: 1,
                4: 6,
                5: 3
            },
            "num_planes": 6,
            "num_cylinders": 3,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 1,
            "expect_face_area": 1225,
            "num_concave": 0,
            "num_convex": 12,
            "num_smooth": 9,
            "expect_edge_length": 35,
            "num_circular_edge": 6,
            "num_closed_edge": 0,
            "num_elliptical_edge": 0,
            "num_straight_edge": 15,
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_block_hole(self):
        npz_file, solid = self.run_tests_on_simple_solid("block_hole.step")
        expected_data = {
            "num_faces": 8,
            "num_edges": 15,
            "num_coedges": 30,
            "loop_hist": {
                1: 2,
                4: 7   # 6 loops on the cube.  The cylinder is split
            },
            "num_planes": 7,
            "num_cylinders": 1,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "expect_face_area": 2000,
            "num_concave": 1,
            "num_convex": 13,
            "num_smooth": 1,   # Seam of the cylinder
            "expect_edge_length": 50,
            "num_circular_edge": 2,
            "num_closed_edge": 2,
            "num_elliptical_edge": 0,
            "num_straight_edge": 13, # One extra straight edge for the cylinder
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_block_through_hole(self):
        npz_file, solid = self.run_tests_on_simple_solid("block_through_hole.step")
        expected_data = {
            "num_faces": 7,
            "num_edges": 15,
            "num_coedges": 30,
            "loop_hist": {
                1: 2,
                4: 7   # 6 loops on the cube.  The cylinder is split
            },
            "num_planes": 6,
            "num_cylinders": 1,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "expect_face_area": 2000,
            "num_concave": 0,
            "num_convex": 14,
            "num_smooth": 1,   # Seam of the cylindrical hole
            "expect_edge_length": 50,
            "num_circular_edge": 2,
            "num_closed_edge": 2,
            "num_elliptical_edge": 0,
            "num_straight_edge": 13, # One extra straight edge for the cylinder
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_block_torus_fillet3(self):
        npz_file, solid = self.run_tests_on_simple_solid("block_torus_fillet3.step")
        expected_data = {
            "num_faces": 10,
            "num_edges": 22,
            "num_coedges": 44,
            "loop_hist": {
                4: 6,
                5: 4
            },
            "num_planes": 6,
            "num_cylinders": 3,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "num_rational_nurbs": 1,
            "expect_face_area": 1200,
            "num_concave": 0,
            "num_convex": 12,
            "num_smooth": 10,
            "expect_edge_length": 30,
            "num_circular_edge": 7,
            "num_closed_edge": 0,
            "num_elliptical_edge": 0,
            "num_straight_edge": 15,
            "closest_features": [
                [ "face_features", [47.066851, 8.558856, 32.675388], "RationalNurbsFaceFeature", 1],
                [ "face_features", [48.183980, 35.791371, 35.746519], "Cylinder", 1],
                [ "edge_features", [44.091955, 0.875533, 20.000000], "CircularEdgeFeature", 1]
            ]
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_cylinder(self):
        npz_file, solid = self.run_tests_on_simple_solid("cylinder.step")
        expected_data = {
            "num_faces": 3,
            "num_edges": 3,
            "num_coedges": 6,
            "loop_hist": {
                1: 2,
                4: 1
            },
            "num_planes": 2,
            "num_cylinders": 1,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "num_concave": 0,
            "num_convex": 2,
            "num_smooth": 1,  # The seam
            "num_circular_edge": 2,
            "num_closed_edge": 2,
            "num_elliptical_edge": 0,
            "num_straight_edge": 1,
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_face_3_loops(self):
        npz_file, solid = self.run_tests_on_simple_solid("face_3_loops.step")
        expected_data = {
            "loop_hist": {
                1: 2,
                4: 19,
                6: 2
            },
            "num_cylinders": 1,
            "num_circular_edge": 2
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_four_cone(self):
        npz_file, solid = self.run_tests_on_simple_solid("four_cone.step")
        expected_data = {
            "num_planes": 6,
            "num_cylinders": 0,
            "num_cones": 4
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_sphere(self):
        npz_file, solid = self.run_tests_on_simple_solid("sphere.step")
        expected_data = {
            "num_faces": 1,
            "num_edges": 3,  # Two "polar" edges and 1 seam edge 
            "num_coedges": 4, # The polar edges mate to themselves
            "loop_hist": {
                4: 1
            },
            "num_planes": 0,
            "num_cylinders": 0,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 1,
            "num_concave": 0,
            "num_convex": 0,
            "num_smooth": 1,
            "num_circular_edge": 1,
            "num_straight_edge": 0,
        }
        self.check_feature_stats(npz_file, solid, expected_data)

    def test_three_concave_edge(self):
        npz_file, solid = self.run_tests_on_simple_solid("three_concave_edge.step")
        expected_data = {
            "num_faces": 10,
            "num_edges": 24,
            "num_coedges": 48,
            "loop_hist": {
                4: 6,
                5: 1,
                6: 2,
                7: 1
            },
            "num_planes": 9,
            "num_cylinders": 1,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "expect_face_area": 1500,
            "num_concave": 3,
            "num_convex": 19,
            "num_smooth": 2,
            "expect_edge_length": 50,
            "num_circular_edge": 2,
            "num_closed_edge": 0,
            "num_elliptical_edge": 0,
            "num_straight_edge": 22,
        }
        self.check_feature_stats(npz_file, solid, expected_data)


    def test_two_concave_closed_edge(self):
        npz_file, solid = self.run_tests_on_simple_solid("two_concave_closed_edge.step")
        expected_data = {
            "num_faces": 10,
            "num_edges": 18, # 12 from block 4 circular 2 seam
            "num_coedges": 36,
            "loop_hist": {
                1: 4,
                4: 8
            },
            "num_planes": 8,
            "num_cylinders": 2,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "num_concave": 2,
            "num_convex": 14,
            "num_smooth": 2,
            "num_circular_edge": 4,
            "num_closed_edge": 4,
            "num_elliptical_edge": 0,
            "num_straight_edge": 14,
        }
        self.check_feature_stats(npz_file, solid, expected_data)


    def test_two_elliptical_edges(self):
        npz_file, solid = self.run_tests_on_simple_solid("two_elliptical_edges.step")
        expected_data = {
            "num_faces": 8,
            "num_edges": 15,
            "num_coedges": 30,
            "loop_hist": {
                1: 2,
                4: 7
            },
            "num_planes": 7,
            "num_cylinders": 0,
            "num_cones": 0,
            "num_torus": 0,
            "num_sphere": 0,
            "num_circular_edge": 0,
            "num_closed_edge": 2,
            "num_elliptical_edge": 2,
            "num_straight_edge": 13, # 12 from block.  1 from seam
        }
        self.check_feature_stats(npz_file, solid, expected_data)





    def test_feature_data(self):
        # step_folder = Path("E:/Autodesk/FusionGallery/SegmentationDataset/converted_stp")
        # npz_folder = Path("E:/Autodesk/FusionGallery/SegmentationDataset/feature_data")
        # self.run_test_on_all_files(step_folder, npz_folder)

        step_folder = Path(__file__).parent / "test_data"
        npz_folder = Path(__file__).parent / "test_working_dir"
        self.run_test_on_all_files(step_folder, npz_folder)
        self.run_tests_on_single_file(step_folder / "118539_1dff9cf9_6.stp", npz_folder)
        self.run_tests_on_single_file(step_folder / "119129_8f04623b_0.stp", npz_folder)

if __name__ == '__main__':
    unittest.main()