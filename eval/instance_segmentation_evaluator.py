import argparse
from pathlib import Path
import numpy as np

from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomProjLib import geomprojlib
from OCC.Core.Geom import Geom_Plane, Geom_TrimmedCurve
from OCC.Core.Geom2dAPI import Geom2dAPI_InterCurveCurve
from OCC.Core.Geom2dAPI import Geom2dAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.Geom2d import Geom2d_TrimmedCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

import utils.data_utils as data_utils

from occwl.solid import Solid 
from occwl.entity_mapper import EntityMapper
import occwl.geometry.geom_utils as geom_utils
from occwl.viewer import Viewer

def debug_show_solid(solid, faces_to_highlight=None, faces_to_highlight2=[]):
    viewer = Viewer()
    if faces_to_highlight is None:
        viewer.display(solid)
    else:
        for face_index, face in enumerate(solid.faces()):
            if face_index in faces_to_highlight:
                viewer.display(face, color="yellow")
            elif face_index in faces_to_highlight2:
                viewer.display(face, color="red")
            else:
                viewer.display(face)
    viewer.show()

def debug_show_trimmed_curve(curve):
    ts = curve.FirstParameter()
    te = curve.LastParameter()
    edge = BRepBuilderAPI_MakeEdge(curve, ts, te).Edge()
    viewer = Viewer()
    viewer.display(edge)
    viewer.show()


def display_2d_curves(curves, curves_to_highlight=set(), point=None):
    import logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, ax = plt.subplots()

    for curve in curves:
        if curve in curves_to_highlight:
            color = "red"
        else:
            color = "black"

        num_points = 10
        tmin = curve.FirstParameter()
        tmax = curve.LastParameter()
        dt = (tmax-tmin)/(num_points-1)
        pts = []
        for i in range(num_points):
            t = tmin + dt*i
            pt2d = curve.Value(t)
            pts.append((pt2d.X(), pt2d.Y()))
        pts = np.asarray(pts)
        l = mlines.Line2D(pts[:,0], pts[:,1], color=color)
        ax.add_line(l)
        if point is not None:
            plt.plot(point.X(), point.Y(), 'o', color='black')
            
    ax.axis('equal')
    plt.axis('off')
    plt.show()


class InstanceSegmentationEvaluator:
    def __init__(self, solid, face_to_instance, face_to_seg, debug_mode=False):
        self.solid = solid
        self.faces = list(solid.faces())
        self.face_to_instance = face_to_instance
        self.instance_to_faces = self.find_faces_in_each_instance(face_to_instance)
        self.face_to_seg = face_to_seg
        self.debug_mode = debug_mode

        # Could be loaded from the segmentation dataset
        self.segment_index_to_name = [
            "ExtrudeSide",
            "ExtrudeEnd",
            "CutSide",
            "CutEnd",
            "Fillet",
            "Chamfer",
            "RevolveSide",
            "RevolveEnd"
        ]
        self.face_vertices, self.face_normals = self.find_vertices_and_normals_for_faces()
        self.base_face_indices_for_instances = self.find_base_face_indices_for_instances()
        self.base_face_planes = self.find_base_face_planes()
        self.extrusion_directions_for_instances = self.find_extrusion_directions_for_instances()
        self.base_face_groups_for_instances = {}

        angle_degrees = 5.0
        self.angle_tol = np.pi*angle_degrees/180.0
        self.length_tol = self.find_length_tol()

        self.barrel_edges_projected_onto_plane = self.project_barrel_edges_to_extrude_plane()

    def is_base_segment(self, segment):
        return segment == "ExtrudeEnd" or segment == "CutEnd"
        
    def get_op_type(self, segment):
        if segment == "ExtrudeSide" or segment == "ExtrudeEnd":
            op_type = "Additive"
        elif segment == "CutSide" or segment == "CutEnd":
            op_type = "Subtractive"
        else:
            op_type = "Undefined"
        return op_type

    def find_faces_in_each_instance(self, face_to_instance):
        instance_to_faces = {}
        for index, instance in enumerate(face_to_instance):
            if not instance in instance_to_faces:
                instance_to_faces[instance] = []
            instance_to_faces[instance].append(index)
        return instance_to_faces

    def find_base_face_indices_for_instances(self):
        base_face_indices_for_instances = {}
        for instance_index in self.instance_to_faces:
            instance_index = int(instance_index)
            instance_faces = self.instance_to_faces[instance_index]
            base_face_indices = []
            for face_index in instance_faces:
                segment = self.get_segment_type_for_face_index(face_index)
                is_base_face = self.is_base_segment(segment)
                if is_base_face:
                    base_face_indices.append(face_index)
            base_face_indices_for_instances[instance_index] = base_face_indices
        return base_face_indices_for_instances


    def get_segment_type_for_face_index(self, face_index):
        segment_index = self.face_to_seg[face_index]
        return self.segment_index_to_name[segment_index]


    def find_vertices_and_normals_for_face(self, face):
        location = TopLoc_Location()
        bt = BRep_Tool()
        
        facing = bt.Triangulation(face.topods_shape(), location)
        if facing == None:
            return [], []
        assert facing.HasUVNodes()
        assert location.IsIdentity()

        tab = facing.Nodes()
        vertices = []
        normals = []
        for i in range(1, facing.NbNodes() + 1):
            vertex = geom_utils.gp_to_numpy(facing.Node(i))
            vertices.append(vertex)
            uv = facing.UVNode(i)
            normal = face.normal(np.array([uv.X(), uv.Y()]))
            normals.append(normal)
        return vertices, normals

    
        
    def find_vertices_and_normals_for_faces(self):
        self.solid.triangulate_all_faces()
        face_vertices = []
        face_normals = []
        for face in self.solid.faces():
            vertices, normals = self.find_vertices_and_normals_for_face(face)
            face_vertices.append(vertices)
            face_normals.append(normals)
        return face_vertices, face_normals

    def find_extrusion_directions_for_instances(self):
        extrusion_directions = {}
        for instance in self.base_face_indices_for_instances:
            base_face_indices = self.base_face_indices_for_instances[instance]
            if len(base_face_indices) > 0:
                first_face = base_face_indices[0]
                if first_face in self.base_face_planes:
                    plane = self.base_face_planes[first_face]
                    extrusion_directions[instance] = plane["normal"]
        return extrusion_directions

    def find_base_face_planes(self):
        base_face_planes = {}
        for face_index, segment_index in enumerate(self.face_to_seg):
            segment = self.get_segment_type_for_face_index(face_index)
            if self.is_base_segment(segment):
                vertices = self.face_vertices[face_index]
                normals = self.face_normals[face_index]
                if len(vertices)>0 and len(normals)>0:
                    base_face_planes[face_index] = {
                        "point": vertices[0],
                        "normal": normals[0]
                    }
        return base_face_planes


    def find_length_tol(self):
        box = self.solid.exact_box()
        return box.max_box_length()/100.0

    def edge_is_line_normal_to_plane(self, edge, plane):
        if edge.curve_type() != "line":
            return False
        line = edge.specific_curve()
        line_axis = line.Position()
        pln = plane.Pln()
        plane_axis = pln.Axis()
        if line_axis.IsParallel(plane_axis, self.angle_tol):
            return True
        return False

        
    def project_barrel_edges_to_extrude_plane_for_face(self, plane, face_index):
        gpl = geomprojlib()
        face = self.faces[face_index]
        curves_2d = []
        for edge in face.edges():
            if not edge.has_curve():
                continue
            if self.edge_is_line_normal_to_plane(edge, plane):
                continue
            assert edge.topods_shape().Location().IsIdentity()
            curve = edge.curve()
            u_bound = edge.u_bounds()
            curve_trimmed = Geom_TrimmedCurve(curve,  u_bound.a, u_bound.b)
            #debug_show_trimmed_curve(curve_trimmed)
            try:
                curve2 = gpl.Curve2d(curve_trimmed, u_bound.a, u_bound.b, plane)
            except:
                # This problem is expected when a line
                # is projected to a plane resulting in a point
                curve2 = None
            if curve2 is not None:
                ts = curve2.FirstParameter()
                te = curve2.LastParameter()
                assert ts > -1e10
                assert te < 1e10
                trimmed_2d = Geom2d_TrimmedCurve(curve2, ts, te)
                curves_2d.append(trimmed_2d)
                #display_2d_curves(curves_2d)


        return curves_2d


    def project_barrel_edges_to_extrude_plane_for_instance(self, instance):
        projected_barrel_edges = {}

        base_faces = self.base_face_indices_for_instances[instance]
        if len(base_faces) == 0:
            return projected_barrel_edges

        plane = self.base_face_planes[base_faces[0]]
        geom_plane = Geom_Plane(
            geom_utils.numpy_to_gp(plane["point"]),
            geom_utils.numpy_to_gp_dir(plane["normal"])
        )

        for face_index in self.instance_to_faces[instance]:
            segment = self.get_segment_type_for_face_index(face_index)
            if not self.is_base_segment(segment):
                projected_barrel_edges_from_face = self.project_barrel_edges_to_extrude_plane_for_face(geom_plane, face_index)
                projected_barrel_edges[face_index] = projected_barrel_edges_from_face
        return projected_barrel_edges

    def project_barrel_edges_to_extrude_plane(self):
        projected_barrel_edges = {}
        for instance in self.base_face_indices_for_instances:
            projected_barrel_edges[instance] = self.project_barrel_edges_to_extrude_plane_for_instance(instance)
        return projected_barrel_edges

    def evaluate(self):
        """
        The following checks need to be run

            - All faces in an instance must have the same operation type.  i.e. all Extrude or all Cut
            - Base faces must be planes
            - If we have two base faces the normals must be anti-parallel
            - If we have two base faces in an extrude the normals must faces outwards.  i.e. away from the other base face
            - If we have two base faces in a cut the normals must point inwards.  i.e. towards the other base face
            - All barrel faces must have normals which are perpendicular to the normals of the base faces.  This assumes no draft angle
            - If we have two base faces then all barrel faces must lie in between them assuming merges have been removed
            - When the barrel faces are projected onto the plane of the base face, they form curves which may touch but not intersect each other

        """
        problems = self.check_operation_type_consistent_with_instance()
        problems.extend(self.check_base_faces_planar())
        problems.extend(self.check_base_face_normals())
        problems.extend(self.check_barrel_face_normals())
        problems.extend(self.check_barrel_faces_between_base_faces())
        if len(problems) == 0:
            # If any problems were detected then the intersection check will
            # fail for sure
            problems.extend(self.check_intersections())
        return problems

    def check_operation_type_consistent_with_instance(self):
        problems = []
        self.instance_op_types = {}
        for instance in self.instance_to_faces:
            faces_indices_in_instance = self.instance_to_faces[instance]
            op_type, problems_for_instance = self.check_operation_type_consistent_within_an_instance(faces_indices_in_instance)
            self.instance_op_types[instance] = op_type
            problems.extend(problems_for_instance)
        return problems

    def check_operation_type_consistent_within_an_instance(self, faces_indices_in_instance):
        problems = []
        num_additive = 0
        num_subtractive = 0
        for face_index in faces_indices_in_instance:
            segment = self.get_segment_type_for_face_index(face_index)
            op_type = self.get_op_type(segment)
            if op_type == "Additive":
                num_additive += 1
            else:
                assert op_type == "Subtractive", "Unsupported operation type"
                num_subtractive += 1
        if num_additive > 0 and num_subtractive > 0:
            problems.append(
                {
                    "check": "operation_type_consistent_within_an_instance",
                    "error": f"Found {num_additive} additive faces and {num_subtractive} subtractive faces"
                }
            )
        

        return op_type, problems


    def check_base_faces_planar(self):
        problems = []
        for instance in self.base_face_indices_for_instances:
            base_face_indices_for_instance = self.base_face_indices_for_instances[instance]
            for base_face_index in base_face_indices_for_instance:
                face = self.faces[base_face_index]
                if face.surface_type() != "plane":
                    problems.append(
                        {
                            "check": "base_faces_planar",
                            "error": f"Face {base_face_index} is not planar"
                        }
                    )

        return problems

    def angle_between_vectors(self, v1, v2):
        """
        Assume we don't have tiny vectors here
        """
        eps = 1e-7
        v1_norm = np.linalg.norm(v1)
        assert v1_norm > eps
        v1 /= v1_norm
                
        v2_norm = np.linalg.norm(v2)
        assert v2_norm > eps
        v2 /= v2_norm
        d = np.dot(v1, v2)
        d = np.clip(d, 0, 1)
        return np.arccos(d)

    def vectors_parallel_or_anti_parallel(self, v1, v2):
        return self.vectors_parallel(v1, v2) or self.vectors_anti_parallel(v1, v2)

    def vectors_parallel(self, v1, v2):
        angle_rads = self.angle_between_vectors(v1, v2)
        if angle_rads < self.angle_tol:
            return True
        return False

    def vectors_anti_parallel(self, v1, v2):
        return self.vectors_parallel(v1, -v2)

    def vectors_perpendicular(self, v1, v2):
        eps = 1e-7
        v1_norm = np.linalg.norm(v1)
        assert v1_norm > eps
        v1 /= v1_norm
                
        v2_norm = np.linalg.norm(v2)
        assert v2_norm > eps
        v2 /= v2_norm
        d = np.dot(v1, v2)
        return abs(d) < np.sin(self.angle_tol)


    def check_face_normals_align_with_vector(self, face_index, direction):
        normals = self.face_normals[face_index]
        for normal in normals:
            if not self.vectors_parallel_or_anti_parallel(normal, direction):
                return False
        return True
        
    def check_face_normals_parallel_to_vector(self, faces, direction):
        for face_index in faces:
            normals = self.face_normals[face_index]
            for normal in normals:
                if not self.vectors_parallel(normal, direction):
                    return False
        return True

    def check_face_normals_antiparallel_to_vector(self, faces, direction):
        for face_index in faces:
            normals = self.face_normals[face_index]
            for normal in normals:
                if not self.vectors_anti_parallel(normal, direction):
                    return False
        return True

    def check_face_normals_perpendicular_to_vector(self, face_index, extrusion_direction):
        normals = self.face_normals[face_index]
        for normal in normals:
            if not self.vectors_perpendicular(normal, extrusion_direction):
                return False
        return True

    def find_coplanar_face_groups(self, base_face_indices_for_instance, extrusion_direction):
        positions_along_extrusion_axis = {}
        for base_face_index in base_face_indices_for_instance:
            plane = self.base_face_planes[base_face_index]
            positions_along_extrusion_axis[base_face_index] = np.dot(extrusion_direction, plane["point"])

        groups = []
        for base_face_index in base_face_indices_for_instance:
            face_pos = positions_along_extrusion_axis[base_face_index]
            added_to_group = False
            for group in groups:
                grouped_pos = group["position_along_extrusion_axis"]
                if abs(face_pos-grouped_pos) < self.length_tol:
                    added_to_group = True
                    group["faces"].append(base_face_index)
            if not added_to_group:
                # Make a new group
                groups.append({
                    "position_along_extrusion_axis": face_pos,
                    "faces": [ base_face_index ]
                })

        # Now we want to sort the groups by position along the axis
        groups = sorted(groups, key=lambda k: k["position_along_extrusion_axis"])
        return groups


    def check_base_face_normals_for_instance(self, instance):
        problems = []
        op_type = self.instance_op_types[instance]
        if not instance in self.extrusion_directions_for_instances:
            assert len(self.base_face_indices_for_instances[instance]) == 0
            # This instance has not base faces any more
            return problems

        extrusion_direction = self.extrusion_directions_for_instances[instance]
        base_face_indices_for_instance = self.base_face_indices_for_instances[instance]
        for base_face_index in base_face_indices_for_instance:
            if not self.check_face_normals_align_with_vector(base_face_index, extrusion_direction):
                #debug_show_solid(self.solid, faces_to_highlight=[base_face_index])
                problems.append(
                    {
                        "check": "check_base_face_normals_for_instance",
                        "error": f"Face {base_face_index} does not align with extrusion direction"
                    }
                )
        if len(problems) > 0:
            # Can't continue checking if this condition isn't met
            #debug_show_solid(self.solid, faces_to_highlight=base_face_indices_for_instance, faces_to_highlight2=[base_face_index])
            return problems
        base_face_groups = self.find_coplanar_face_groups(base_face_indices_for_instance, extrusion_direction)
        self.base_face_groups_for_instances[instance] = base_face_groups

        if len(base_face_groups) > 2:
            problems.append(
                {
                    "check": "check_base_face_normals_for_instance",
                    "error": f"Base faces found in {len(base_face_groups)} groups.  Only 2 groups allowed"
                }
            )
            return problems
        
        if len(base_face_groups) < 2:
            # The remaining checks require two groups
            return problems

        # Now we check the normals point outwards for Additive and inwards for subtractive 
        # extrudes
        if op_type == "Additive":
            #debug_show_solid(self.solid, faces_to_highlight=base_face_groups[0]["faces"])
            if not self.check_face_normals_antiparallel_to_vector(base_face_groups[0]["faces"], extrusion_direction):
                problems.append(
                    {
                        "check": "check_base_face_normals_for_instance",
                        "error": f"For Additive extrude the lower plane must point away from the extrusion_direction"
                    }
                )
            if not self.check_face_normals_parallel_to_vector(base_face_groups[1]["faces"], extrusion_direction):
                problems.append(
                    {
                        "check": "check_base_face_normals_for_instance",
                        "error": f"For Additive extrude the upper plane must point along from the extrusion_direction"
                    }
                )
        else:
            if not self.check_face_normals_parallel_to_vector(base_face_groups[0]["faces"], extrusion_direction):
                #debug_show_solid(self.solid, faces_to_highlight=base_face_groups[0]["faces"])
                problems.append(
                    {
                        "check": "check_base_face_normals_for_instance",
                        "error": f"For Subtractive extrude the lower plane must point along the extrusion_direction"
                    }
                )
            if not self.check_face_normals_antiparallel_to_vector(base_face_groups[1]["faces"], extrusion_direction):
                problems.append(
                    {
                        "check": "check_base_face_normals_for_instance",
                        "error": f"For Subtractive extrude the upper plane must point away from the extrusion_direction"
                    }
                )
        return problems

        
    def check_base_face_normals(self):
        problems = []
        for instance in self.base_face_indices_for_instances:
            base_faces = self.base_face_indices_for_instances[instance]
            problems_for_instance = self.check_base_face_normals_for_instance(instance)
            problems.extend(problems_for_instance)
        return problems

        
    def check_barrel_face_normals_for_instance(self, instance):
        problems = []
        if not instance in self.extrusion_directions_for_instances:
            if instance in self.base_face_indices_for_instances:
                assert len(self.base_face_indices_for_instances[instance]) == 0
            # This instance has not base faces any more
            return problems

        extrusion_direction = self.extrusion_directions_for_instances[instance]
        for face_index in self.instance_to_faces[instance]:
            segment = self.get_segment_type_for_face_index(face_index)
            is_base_face = self.is_base_segment(segment)
            if not is_base_face:
                if not self.check_face_normals_perpendicular_to_vector(face_index, extrusion_direction):
                    if len(problems) == 0:
                        # debug_show_solid(
                        #     self.solid, 
                        #     faces_to_highlight=self.base_face_indices_for_instances[instance],
                        #     faces_to_highlight2 = [ face_index ]
                        # )
                        pass
                    problems.append(
                        {
                            "check": "check_barrel_face_normals_for_instance",
                            "error": f"Barrel face {face_index} with semantic label {segment} normal not perpendicular to extrusion direction"
                        }
                    )

        return problems



    def check_barrel_face_normals(self):
        problems = []
        for instance, base_faces in enumerate(self.base_face_indices_for_instances):
            problems_for_instance = self.check_barrel_face_normals_for_instance(instance)
            problems.extend(problems_for_instance)
        return problems

    def check_points_between_planes(self, face_index, base_face_groups, extrusion_direction):
        assert len(base_face_groups) == 2
        points = self.face_vertices[face_index]
        for point in points:
            pos = np.dot(point, extrusion_direction)
            if pos < base_face_groups[0]["position_along_extrusion_axis"] + self.length_eps:
                return False
            if pos > base_face_groups[1]["position_along_extrusion_axis"] - self.length_eps:
                return False
        return True


    def check_barrel_faces_between_base_faces_for_instance(instance):
        problems = []
        if not instance in self.extrusion_directions_for_instances:
            assert not instance in self.base_face_indices_for_instances
            # This instance has not base faces any more
            return problems

        extrusion_direction = self.extrusion_directions_for_instances[instance]

        if not instance in self.base_face_groups_for_instances:
            # This check requires 2 base face groups
            return problems
        
        base_face_groups = self.base_face_groups_for_instances[instance]
        if len(base_face_groups) != 2:
            # This check requires 2 base face groups
            return problems

        for face_index in self.instance_to_faces[instance]:
            segment = self.get_segment_type_for_face_index(face_index)
            is_base_face = self.is_base_segment(segment)
            if not is_base_face:
                if not self.check_points_between_planes(face_index, base_face_groups, extrusion_direction):
                    problems.append(
                        {
                            "check": "check_barrel_faces_between_base_faces_for_instance",
                            "error": f"Barrel face {face_index} is not between base faces"
                        }
                    )
        return problems
        
    def check_barrel_faces_between_base_faces(self):
        problems = []
        for instance, base_faces in enumerate(self.base_face_indices_for_instances):
            problems_for_instance = self.check_barrel_face_normals_for_instance(instance)
            problems.extend(problems_for_instance)
        return problems

        
    def check_intersections(self):
        problems = []
        for instance in self.base_face_indices_for_instances:
            problems_for_instance = self.check_intersections_for_instance(instance)
            problems.extend(problems_for_instance)
        return problems

    def is_end_point(self, point, curve):
        tol_2d = 0.01
        start = curve.StartPoint()
        end = curve.EndPoint()
        if start.Distance(point) < tol_2d:
            return True
        if end.Distance(point) < tol_2d:
            return True
        return False


    def check_intersections_for_instance(self, instance):
        problems = []
        projected_barrel_edges_for_faces = self.barrel_edges_projected_onto_plane[instance]
        for face_index1 in projected_barrel_edges_for_faces:
            barrel_edges_for_face1 = projected_barrel_edges_for_faces[face_index1]
            for face_index2 in projected_barrel_edges_for_faces:
                barrel_edges_for_face2 = projected_barrel_edges_for_faces[face_index2]
                if face_index1 < face_index2:
                    for barrel_edge1 in barrel_edges_for_face1:
                        for barrel_edge2 in barrel_edges_for_face2:
                            intersector = Geom2dAPI_InterCurveCurve(barrel_edge1, barrel_edge2)
                            for i in range(1, intersector.NbPoints()+1):
                                point = intersector.Point(i)
                                if self.is_end_point(point, barrel_edge1) or self.is_end_point(point, barrel_edge2):
                                    continue
                                if self.debug_mode:
                                    self.debug_show_intersections(
                                        instance,
                                        face_index1,
                                        face_index2,
                                        barrel_edge1,
                                        barrel_edge2,
                                        point
                                    )
                                problems.append({
                                    "check": "check_intersections_for_instance",
                                    "error": f"Intersection found between projections of face {face_index1} and {face_index2}"
                                })
        return problems

        

    def debug_show_intersections(
            self,
            instance,
            face1,
            face2,
            barrel_edge1,
            barrel_edge2,
            point
        ):
        base_faces = self.base_face_indices_for_instances[instance]
        barrel_faces = [ face1, face2 ]
        debug_show_solid(self.solid, faces_to_highlight=base_faces, faces_to_highlight2=barrel_faces)
        
        projected_barrel_edges_for_faces = self.barrel_edges_projected_onto_plane[instance]

        curves = [ c for c in projected_barrel_edges_for_faces[face1] ]
        curves.extend([ c for c in projected_barrel_edges_for_faces[face2] ])
        curves_to_highlight = set([barrel_edge1, barrel_edge2])
        display_2d_curves(curves, curves_to_highlight, point)

        base_faces = self.base_face_indices_for_instances[instance]
        plane = self.base_face_planes[base_faces[0]]
        geom_plane = Geom_Plane(
            geom_utils.numpy_to_gp(plane["point"]),
            geom_utils.numpy_to_gp_dir(plane["normal"])
        )

        self.project_barrel_edges_to_extrude_plane_for_face(geom_plane, face1)
        self.project_barrel_edges_to_extrude_plane_for_face(geom_plane, face2)



def evaluate_example(solid, face_to_instance, face_to_seg, debug_mode=False):
    eval = InstanceSegmentationEvaluator(
        solid, 
        face_to_instance, 
        face_to_seg, 
        debug_mode=debug_mode
    )
    problems = eval.evaluate()
    return problems


def load_compsolid(step_file):
    """
    Load the body from the step file.  
    We expect only one body in each file
    """
    step_filename_str = str(step_file)
    reader = STEPControl_Reader()
    reader.ReadFile(step_filename_str)
    reader.TransferRoots()
    shape = reader.OneShape()
    return Solid(shape)

def face_to_instance_from_timeline(timeline_info_file):
    timeline_info = data_utils.load_json_data(timeline_info_file)
    faces = timeline_info["body"]["faces"]
    face_to_instance = []
    uuid_to_instance_index = {}
    for face in faces:
        uuid = face["feature"]
        if not uuid in uuid_to_instance_index:
            uuid_to_instance_index[uuid] = len(uuid_to_instance_index)
        face_to_instance.append(uuid_to_instance_index[uuid])
    return face_to_instance

def only_extrude_and_cut(file):
    timeline_info = data_utils.load_json_data(file)
    for feature in timeline_info["features"].values():
        if feature["type"] != "ExtrudeFeature":
            return False
    return True


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction_dataset_folder", type=str, help="Dataset folder with the construction layout")
    parser.add_argument("--construction_dataset_file", type=str, help="Dataset file")
    parser.add_argument("--dataset_folder", type=str, help="folder with full dataset")
    parser.add_argument("--step_file", type=str, help="A step file containing the data")
    parser.add_argument("--step_folder", type=str, help="Folder containing step data")
    parser.add_argument("--instance_file", type=str, help="For each face we have an integer representing the instance")
    parser.add_argument("--timeline_info", type=str, help="The timeline_info file for the solid")
    parser.add_argument("--semantic_file", type=str, help="For each face we have an integer representing the segment")
    return parser

def evaluate_single_file(args):
    if args.instance_file is not None:
        face_to_instance = np.loadtxt(args.instance_file, dtype=np.int64)
    else:
        assert args.timeline_info is not None
        face_to_instance = face_to_instance_from_timeline(args.timeline_info)
    face_to_seg = np.loadtxt(args.semantic_file, dtype=np.int64)
    solid = load_compsolid(args.step_file)

    problems = evaluate_example(solid, face_to_instance, face_to_seg)
    for problem in problems:
        print(problem)

    print("Completed evaluate_single_file()")

def evaluate_folder(args):
    folder = Path(args.dataset_folder)
    step_folder = folder / "breps/step"
    seg_folder = folder / "breps/seg"
    timeline_folder = folder / "timeline_info"
    timeline_files = timeline_folder.glob("*.json")
    for file in timeline_files:
        if not only_extrude_and_cut(file):
            continue
        step_file = step_folder / (file.stem + ".stp")
        if not step_file.exists():
            continue
        seg_file = seg_folder / (file.stem + ".seg")
        if not seg_file.exists():
            continue
        print(f"Checking {step_file}")
        face_to_instance = face_to_instance_from_timeline(file)
        face_to_seg = np.loadtxt(seg_file, dtype=np.int64)
        solid = load_compsolid(step_file)
        problems = evaluate_example(solid, face_to_instance, face_to_seg)
        if len(problems) >0:
            print(f"Problem with {step_file}")
            for problem in problems:
                print(problem)

    print("Completed evaluate_folder()")

def load_npz(file):
    return np.load(file)


def evaluate_construction_dataset_folder(args):
    dataset_file = Path(args.construction_dataset_file)
    dataset_info = data_utils.load_json_data(dataset_file)
    folder = Path(args.construction_dataset_folder)
    step_folder = Path(args.step_folder)
    seg_folder = folder / "semantics"
    instance_folder = folder / "instance"
    for model_name in dataset_info["modelname_to_count"]:
        count = dataset_info["modelname_to_count"][model_name]
        index = -1
        c = 0
        while c < count:
            index = index + 1
            if index > 1000 and index > count:
                # A file is missing
                break
            model = f"{model_name}_{index:04d}"
            step_file = step_folder / (model + ".step")
            if not step_file.exists():
                continue
            seg_file = seg_folder / (f"{model_name}_{c}_semantic.npz")
            if not seg_file.exists():
                continue
            instance_file = instance_folder / (f"{model_name}_{c}_instance.npz")
            if not instance_file.exists():
                print(f"seg_file {seg_file} exists but {instance_file} is missing")
                continue
            #print(f"Checking {step_file}")
            c = c + 1
            face_to_instance = load_npz(instance_file)
            face_to_seg = load_npz(seg_file)
            solid = load_compsolid(step_file)
            problems = evaluate_example(solid, face_to_instance, face_to_seg)
            if len(problems) >0:
                print(f"Problem with {step_file}")
                print(f"\"--step_file\", \"{step_file}\",")
                print(f"\"--semantic_file\", \"{seg_file}\",")
                print(f"\"--instance_file\",\"{instance_file}\"")
                print(problems[0])
                # for problem in problems:
                #     print(problem)
            

    print("Completed evaluate_construction_dataset_folder()")

def evaluate_construction_dataset_file(args):
    face_to_instance = load_npz(args.instance_file)
    face_to_seg = load_npz(args.semantic_file)
    solid = load_compsolid(args.step_file)
    problems = evaluate_example(solid, face_to_instance, face_to_seg, debug_mode=True)
    if len(problems) >0:
        print(f"Problem with {args.step_file}")
        for problem in problems:
            print(problem)


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.construction_dataset_folder is not None:
        evaluate_construction_dataset_folder(args)
    elif args.dataset_folder is not None:
        evaluate_folder(args)
    else:
        if args.timeline_info is not None:
            evaluate_single_file(args)
        else:
            evaluate_construction_dataset_file(args)


