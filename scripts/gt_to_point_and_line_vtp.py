from graph_tool import load_graph
import os

from curvaturia import TriangleGraph
from curvaturia import curvaturia_io as io

tg_file = '/fs/gpfs03/lv04/pool/pool-ruben/Maria/4Javier/new_curvature/TCB/' \
          '180830_TITAN_l2_t2half/TCB_180830_l2_t2half.cER.AVV_rh10.gt'
tg = TriangleGraph()
tg.graph = load_graph(tg_file)
poly_verts, poly_lines = tg.graph_to_points_and_lines_polys()
vtp_files_base = os.path.splitext(tg_file)[0]
io.save_vtp(poly_verts, vtp_files_base + '.vertices.vtp')
io.save_vtp(poly_lines, vtp_files_base + '.edges.vtp')
