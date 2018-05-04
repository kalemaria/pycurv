import numpy as np
import math
import vtk
from graph_tool import Graph, load_graph
import time
import pandas as pd

from surface_graphs import TriangleGraph
import pysurf_io as io


def calculate_distances(tg_PM, tg_cER, maxdist, minth, verbose=False):
    # TODO docstring
    # tg_PM and tg_cER ate TriangleGraph objects, including PM or cER surface
    # tg_PM has corrected normals (in cER surface direction)

    # initialize the vertex properties if cER graph and distances lists:
    tg_cER.graph.vp.cERmembrane = tg_cER.graph.new_vertex_property("int")
    tg_cER.graph.vp.PMdistance = tg_cER.graph.new_vertex_property("float")
    tg_cER.graph.vp.cERthickness = tg_cER.graph.new_vertex_property("float")
    d1s = []  # distances from PM to cER
    d2s = []  # distances between both cER membranes (cER thickness)

    # Define a cellLocator to be able to compute intersections between lines
    # and the cER surface:
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(tg_cER.surface)
    locator.BuildLocator()
    tolerance = 0.001

    # functions used in the loop
    sqrt = math.sqrt
    dot = np.dot

    # for each vertex v in PM graph (represents triangle on PM surface):
    for v in tg_PM.graph.vertices():
        # get its center coordinate as p0 and corrected normal vector N_v
        p0 = np.array(tg_PM.graph.vp.xyz[v])
        N_v = np.array(tg_PM.graph.vp.N_v[v]) * -1  # TODO check both directions

        # find a point pmax at distance maxdist from p0 in the normal direction
        pmax = p0 + N_v * maxdist

        # find 1st intersection p1 on cER surface with the line from p0 to pmax
        # and the triangle containing it
        # Outputs (we need only p1 and cell1_id):
        t = vtk.mutable(0)
        p1 = [0.0, 0.0, 0.0]  # x, y, z position of the first intersection
        pcoords = [0.0, 0.0, 0.0]
        sub_id = vtk.mutable(0)
        cell1_id = vtk.mutable(0)  # the triangle id containing p1
        locator.IntersectWithLine(p0, pmax, tolerance, t, p1, pcoords, sub_id,
                                  cell1_id)
        # If there is no intersection, p1 stays like initialized:
        if p1 == [0.0, 0.0, 0.0]:
            # if verbose:
            #     print("No intersection point found")
            continue  # with next PM triangle

        if verbose:
            print("First intersection point found: ({}, {}, {})".format(
                p1[0], p1[1], p1[2]))
        p1 = np.array(p1)

        # calculate the distance d1 from PM to cER:
        d1 = sqrt(dot(p1 - p0, p1 - p0))
        if verbose:
            print("distance from PM to cER d1 = {}".format(d1))

        # find two new points: at distance minth and maxdist from p1 in the
        # normal direction
        p0_new = p1 + N_v * minth
        pmax_new = p1 + N_v * maxdist

        # find 2nd intersection p2 on cER surface with the line from p0_new to
        # pmax_new and the triangle containing it
        t = vtk.mutable(0)
        p2 = [0.0, 0.0, 0.0]  # x, y, z position of the next intersection
        pcoords = [0.0, 0.0, 0.0]
        sub_id = vtk.mutable(0)
        cell2_id = vtk.mutable(0)  # the triangle id containing p2
        locator.IntersectWithLine(p0_new, pmax_new, tolerance, t, p2, pcoords,
                                  sub_id, cell2_id)
        # If there is no intersection, p2 stays like initialized:
        if p2 == [0.0, 0.0, 0.0]:
            if verbose:
                print("No second intersection point found - discard the first")
            continue  # with next PM triangle

        if verbose:
            print("Second intersection point found: ({}, {}, {})".format(
                p2[0], p2[1], p2[2]))
        p2 = np.array(p2)

        # calculate the cER thickness d2:
        d2 = sqrt(dot(p2 - p1, p2 - p1))
        if verbose:
            print("cER thickness d2 = {}".format(d2))

        # add d1 and d2 to lists:
        d1s.append(d1)
        d2s.append(d2)

        # get the two vertices (of the triangle centers) from the cell ids:
        v1 = tg_cER.graph.vertex(cell1_id)
        v2 = tg_cER.graph.vertex(cell2_id)

        # fill out the vertex property of cER graph "cERmembrane":
        # 1 for the 1st intersected triangle and 2 for the 2nd one
        tg_cER.graph.vp.cERmembrane[v1] = 1
        tg_cER.graph.vp.cERmembrane[v2] = 2

        # fill out the vertex property of cER graph "PMdistance":
        # d1 for the 1st intersected triangle and d1+d2 for the 2nd one
        tg_cER.graph.vp.PMdistance[v1] = d1
        tg_cER.graph.vp.PMdistance[v2] = d1 + d2

        # fill out the vertex property of cER graph  "cERthickness":
        # d2 for both intersected triangles
        tg_cER.graph.vp.cERthickness[v1] = d2
        tg_cER.graph.vp.cERthickness[v2] = d2

    return d1s, d2s


def run_calculate_distances(PM_graph_file, cER_surf_file, cER_graph_file,
                            cER_surf_outfile, cER_graph_outfile,
                            distances_outfile, maxdist, minth, verbose):
    # Load the input files:
    tg_PM = TriangleGraph(vtk.vtkPolyData())  # PM surface is not required
    tg_PM.graph = load_graph(PM_graph_file)
    cER_surf = io.load_poly(cER_surf_file)
    tg_cER = TriangleGraph(cER_surf)
    tg_cER.graph = load_graph(cER_graph_file)

    # Calculate distances:
    d1s, d2s = calculate_distances(tg_PM, tg_cER, maxdist, minth, verbose)
    print("{} d1s".format(len(d1s)))
    print("{} d2s".format(len(d2s)))
    # Save the distances into distances_outfile:
    df = pd.DataFrame()
    df["d1"] = d1s
    df["d2"] = d2s
    df.to_csv(distances_outfile, sep=';')

    # Transform the resulting graph to a surface with triangles:
    cER_surf_dist = tg_cER.graph_to_triangle_poly()
    # Save the resulting graph and surface into files:
    tg_cER.graph.save(cER_graph_outfile)
    io.save_vtp(cER_surf_dist, cER_surf_outfile)

if __name__ == "__main__":
    t_begin = time.time()

    base_fold = "/fs/pool/pool-ruben/Maria/curvature/Javier/"
    # The "famous" tcb (done cER RH=6 and 10 + PM RH=6):
    fold = "{}tcb_t3_ny01/new_workflow/".format(base_fold)
    base_filename = "t3_ny01_cropped_"

    # Input files:
    # File with scaled PM graph with corrected normals:
    PM_graph_file = "{}{}PM_holes3.NVV_rh6_epsilon0_eta0.gt".format(
        fold, base_filename)
    # Files with scaled cER surface and graph, after curvature calculation:
    cER_surf_file = "{}{}cER_holes3.VV_area2_rh6_epsilon0_eta0.vtp".format(
        fold, base_filename)
    cER_graph_file = "{}{}cER_holes3.VV_area2_rh6_epsilon0_eta0.gt".format(
        fold, base_filename)

    # Input parameters:
    maxdist = 50
    for minth in [5]:

        # Output files:
        cER_surf_outfile = "{}.PMdist_maxdist{}_minth{}.vtp".format(
            cER_surf_file[0:-4], maxdist, minth)
        cER_graph_outfile = "{}.PMdist_maxdist{}_minth{}.gt".format(
            cER_graph_file[0:-3], maxdist, minth)
        distances_outfile = "{}.PMdist_maxdist{}_minth{}.csv".format(
            cER_surf_file[0:-4], maxdist, minth)

        run_calculate_distances(PM_graph_file, cER_surf_file, cER_graph_file,
                                cER_surf_outfile, cER_graph_outfile,
                                distances_outfile, maxdist, minth,
                                verbose=False)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)
