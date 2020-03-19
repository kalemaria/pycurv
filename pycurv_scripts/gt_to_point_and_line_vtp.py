from graph_tool import load_graph
import os

from pycurv import TriangleGraph
from pycurv import pycurv_io as io

"""
Contains a script getting triangle vertices and edges from a TriangleGraph and
saving them in to separate VTP file s for visualization with ParaView.

Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""

__author__ = 'Maria Salfer'

tg_file = '/fs/pool/pool-ruben/Maria/workspace/github/pycurv/' \
          'experimental_data_sets/ER/AVV/TCB_180830_l2_t2half.ER.AVV_rh10.gt'
tg = TriangleGraph()
tg.graph = load_graph(tg_file)
poly_verts, poly_lines = tg.graph_to_points_and_lines_polys()
vtp_files_base = os.path.splitext(tg_file)[0]
io.save_vtp(poly_verts, vtp_files_base + '.vertices.vtp')
io.save_vtp(poly_lines, vtp_files_base + '.edges.vtp')
