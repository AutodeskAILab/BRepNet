"""
Create an occwl version 1.0.0 entity from a TopoDS entity 
"""
from OCC.Core.TopoDS import (
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shell,
    TopoDS_Solid,
    TopoDS_Vertex,
    TopoDS_Wire,
    TopoDS_Compound,
    TopoDS_CompSolid,
)

# occwl
from occwl.solid import Solid
from occwl.compound import Compound
from occwl.shell import Shell
from occwl.face import Face
from occwl.edge import Edge
from occwl.wire import Wire
from occwl.vertex import Vertex

def create_occwl(topo_ds_shape):
    if isinstance(topo_ds_shape, TopoDS_Edge):
        occwl_ent = Edge(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_Face):
        occwl_ent = Face(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_Shell):
        occwl_ent = Shell(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_Solid):
        occwl_ent = Solid(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_Vertex):
        occwl_ent = Vertex(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_Wire):
        occwl_ent = Wire(topo_ds_shape)
    elif isinstance(topo_ds_shape, TopoDS_CompSolid) or isinstance(topo_ds_shape, TopoDS_Compound):
        occwl_ent = Compound(topo_ds_shape)
    else:
        assert False, f"Unsupported entity {type(topo_ds_shape)}. Cant convert to occwl"
    
    return occwl_ent