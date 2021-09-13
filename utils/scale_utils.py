"""
Utility function to scale a solid body into a box 
[-1, 1]^3
"""
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_AddOptimal
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

# occwl
from occwl.solid import Solid

def find_box(solid):
    bbox = Bnd_Box()
    use_triangulation = True
    use_shapetolerance = False
    brepbndlib_AddOptimal(solid, bbox, use_triangulation, use_shapetolerance)
    return bbox

def scale_solid_to_unit_box(solid):
    is_occwl = False
    if isinstance(solid, Solid):
        is_occwl = True
        topods_solid = solid.topods_solid()
    else:
        topods_solid = solid
    bbox = find_box(topods_solid)
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
    apply_transform.Perform(topods_solid)
    transformed_solid = apply_transform.ModifiedShape(topods_solid)

    if is_occwl:
        print("Switch back to occwl solid")
        return Solid(transformed_solid)
    return transformed_solid



