from graph_tool import load_graph
import os

from pycurv import TriangleGraph, calculate_curvedness
from pycurv import pycurv_io as io

# Read in graph GT file
tg_file = '/fs/gpfs03/lv04/pool/pool-ruben/Maria/4Javier/new_curvature/TCB/' \
          '180830_TITAN_l2_t2half/TCB_180830_l2_t2half.cER.AVV_rh10.gt'
tg = TriangleGraph()
tg.graph = load_graph(tg_file)
# Read in VTK principle curvatures for each vertex,
# calculate curvedness and add as a new property
tg.graph.vp.vtk_curvedness = tg.graph.new_vertex_property("float")
for v in tg.graph.vertices():
    v_vtk_kappa_1 = tg.graph.vp.max_curvature[v]
    v_vtk_kappa_2 = tg.graph.vp.min_curvature[v]
    v_vtk_curvedness = calculate_curvedness(v_vtk_kappa_1, v_vtk_kappa_2)
    tg.graph.vp.vtk_curvedness[v] = v_vtk_curvedness
# Transform the graph to surface and write to a VTP file
surf = tg.graph_to_triangle_poly(verbose=False)
surf_file = os.path.splitext(tg_file)[0] + "_VTKcurvedness.vtp"
io.save_vtp(surf, surf_file)
