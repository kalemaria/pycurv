import numpy as np
import math
import vtk
from graph_tool import Graph, load_graph
import time
import pandas as pd

from surface_graphs import TriangleGraph
import pysurf_io as io


def calculate_distances(tg_PM, tg_cER, maxdist, maxth, verbose=False):
    """

    Args:
        tg_PM (TriangleGraph): TriangleGraph object of PM surface with corrected
            normals (now should be pointing away from cER surface direction)
        tg_cER (TriangleGraph): TriangleGraph object of cER surface
        maxdist (int): maximal distance (nm) from PM to first cER membrane
        maxth (int): maximal cER thickness (nm)
        verbose:

    Returns:
        two lists of distances:
        d1: distances (nm) from PM to first cER membrane)
        d2: cER thicknesses (nm)
    """
    print("maxdist = {} nm".format(maxdist))
    print("maxth = {} nm".format(maxth))

    # initialize the vertex properties if cER graph and distances lists:
    tg_cER.graph.vp.cERmembrane = tg_cER.graph.new_vertex_property("int")
    tg_cER.graph.vp.PMdistance = tg_cER.graph.new_vertex_property(
        "float", vals=-1)
    tg_cER.graph.vp.cERthickness = tg_cER.graph.new_vertex_property(
        "float", vals=-1)
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
        N_v = np.array(tg_PM.graph.vp.N_v[v]) * -1  # TODO check both directions?

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
            continue  # with next PM triangle

        # get the vertex (at the triangle center) of cell1 from its id:
        v1 = tg_cER.graph.vertex(cell1_id)
        # check if p1 is on first cER membrane:
        # is the angle between the normal from v1 and N_v from v > pi/2?
        # then it's on the second membrane - don't continue looking
        N1 = tg_cER.graph.vp.N_v[v1]
        cos_angle1 = np.dot(N_v, N1)
        if cos_angle1 < 0:  # angle1 > pi/2
            if verbose:
                print("First intersection point found on second membrane - "
                      "discard it")
            continue  # with next PM triangle

        # if verbose:
        #     print("First intersection point found: ({}, {}, {})".format(
        #         p1[0], p1[1], p1[2]))
        p1 = np.array(p1)

        # calculate the distance d1 from PM to cER:
        d1 = sqrt(dot(p1 - p0, p1 - p0))
        # if verbose:
        #     print("distance from PM to cER d1 = {}".format(d1))
        if d1 > maxdist:
            if verbose:
                print("d1 = {} > maxdist - discard the point".format(d1))
            continue  # with next PM triangle

        t = vtk.mutable(0)
        p2 = [0.0, 0.0, 0.0]  # x, y, z position of the next intersection
        pcoords = [0.0, 0.0, 0.0]
        sub_id = vtk.mutable(0)
        cell2_id = vtk.mutable(0)  # the triangle id containing p2
        for minth in range(1, maxth):
            # find two new points: at distance minth and maxth from p1 in the
            # normal direction
            p0_new = p1 + N_v * minth
            pmax_new = p1 + N_v * maxth

            # find 2nd intersection p2 on cER surface with the line from p0_new
            # to pmax_new and the triangle containing it
            locator.IntersectWithLine(p0_new, pmax_new, tolerance, t, p2,
                                      pcoords, sub_id, cell2_id)
            if p2 == [0.0, 0.0, 0.0]:  # no 2nd intersection - stop looking
                break
            else:
                # get the vertex (at the triangle center) of cell2 from its id:
                v2 = tg_cER.graph.vertex(cell2_id)
                # check if p2 is on second cER membrane:
                # is the angle between the normals from v1 and from v2 < pi/2?
                # then we are still on the first membrane - continue looking
                N2 = tg_cER.graph.vp.N_v[v2]
                cos_angle2 = np.dot(N1, N2)
                if cos_angle2 > 0:  # angle2 < pi/2
                    # if verbose:
                    #     print("Second intersection point found on the first "
                    #           "membrane - continue looking")
                    minth += 1
                else:  # otherwise we are on the second membrane - stop looking
                    break
        # If there is no intersection, p2 stays like initialized:
        if p2 == [0.0, 0.0, 0.0]:
            if verbose:
                print("No second intersection point found - discard the first")
            continue  # with next PM triangle

        # if verbose:
        #     print("Second intersection point found: ({}, {}, {})".format(
        #         p2[0], p2[1], p2[2]))
        p2 = np.array(p2)

        # calculate the cER thickness d2:
        d2 = sqrt(dot(p2 - p1, p2 - p1))
        # if verbose:
        #     print("cER thickness d2 = {}".format(d2))
        if d2 > maxth:
            if verbose:
                print("d2 = {} > maxth - discard the pair".format(d2))
            continue  # with next PM triangle

        # add d1 and d2 to lists:
        d1s.append(d1)
        d2s.append(d2)

        # fill out the vertex property of cER graph "cERmembrane":
        # 1 for the 1st intersected triangle and 2 for the 2nd one
        tg_cER.graph.vp.cERmembrane[v1] = 1
        tg_cER.graph.vp.cERmembrane[v2] = 2

        # fill out the vertex property of cER graph "PMdistance":
        # d1 for the 1st intersected triangle and d1+d2 for the 2nd one
        tg_cER.graph.vp.PMdistance[v1] = d1
        # tg_cER.graph.vp.PMdistance[v2] = d1 + d2

        # fill out the vertex property of cER graph  "cERthickness":
        # d2 for both intersected triangles
        # tg_cER.graph.vp.cERthickness[v1] = d2
        tg_cER.graph.vp.cERthickness[v2] = d2

    return d1s, d2s


def run_calculate_distances(PM_graph_file, cER_surf_file, cER_graph_file,
                            cER_surf_outfile, cER_graph_outfile,
                            distances_outfile, maxdist, maxth, verbose):
    # Load the input files:
    tg_PM = TriangleGraph(vtk.vtkPolyData())  # PM surface is not required
    tg_PM.graph = load_graph(PM_graph_file)
    cER_surf = io.load_poly(cER_surf_file)
    tg_cER = TriangleGraph(cER_surf)
    tg_cER.graph = load_graph(cER_graph_file)

    # Calculate distances:
    d1s, d2s = calculate_distances(tg_PM, tg_cER, maxdist, maxth, verbose)
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
    # fold = "{}tcb_t3_ny01/new_workflow/".format(base_fold)
    # base_filename = "t3_ny01_cropped_"
    # rh = 6
    # The new tcb (done cropped cER and PM with RH=15):
    # tomo = "tcb_170924_l2_t2_ny01"
    # fold = "{}{}/".format(base_fold, tomo)
    # base_filename = "{}_cropped_".format(tomo)
    # The good scs (done cER and PM with RH=15):
    tomo = "scs_171108_l1_t2_ny01"
    fold = "{}{}/".format(base_fold, tomo)
    base_filename = "{}_".format(tomo)
    rh = 15
    pixel_size = 1.368

    # Input files:
    # File with scaled PM graph with corrected normals:
    PM_graph_file = "{}{}PM_holes3.NVV_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)
    # Files with scaled cER surface and graph, after curvature calculation:
    cER_surf_file = "{}{}cER_holes3.VV_area2_rh{}_epsilon0_eta0.vtp".format(
        fold, base_filename, rh)
    cER_graph_file = "{}{}cER_holes3.VV_area2_rh{}_epsilon0_eta0.gt".format(
        fold, base_filename, rh)

    # Input parameters:
    maxdist_voxels = 60
    maxth_voxels = 60
    maxdist_nm = int(maxdist_voxels * pixel_size)
    maxth_nm = int(maxth_voxels * pixel_size)

    # Output files:
    cER_surf_outfile = "{}.PMdist_maxdist{}_maxth{}.vtp".format(
        cER_surf_file[0:-4], maxdist_nm, maxth_nm)
    cER_graph_outfile = "{}.PMdist_maxdist{}_maxth{}.gt".format(
        cER_graph_file[0:-3], maxdist_nm, maxth_nm)
    distances_outfile = "{}.PMdist_maxdist{}_maxth{}.csv".format(
        cER_surf_file[0:-4], maxdist_nm, maxth_nm)

    run_calculate_distances(PM_graph_file, cER_surf_file, cER_graph_file,
                            cER_surf_outfile, cER_graph_outfile,
                            distances_outfile, maxdist_nm, maxth_nm,
                            verbose=False)

    t_end = time.time()
    duration = t_end - t_begin
    print '\nTotal elapsed time: %s min %s s' % divmod(duration, 60)
