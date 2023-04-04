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
from occwl.compound import Compound

from utils.create_occwl_from_occ import create_occwl

def find_box(solid):
    bbox = Bnd_Box()
    use_triangulation = True
    use_shapetolerance = False
    brepbndlib_AddOptimal(solid, bbox, use_triangulation, use_shapetolerance)
    return bbox

def scale_solid_to_unit_box(solid):
    if isinstance(solid, Solid):
        return solid.scale_to_unit_box(copy=True)
    solid = create_occwl(solid)
    solid = solid.scale_to_unit_box(copy=True)
    return solid.topods_shape()



