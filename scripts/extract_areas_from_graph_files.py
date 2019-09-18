import pandas as pd
from graph_tool import load_graph
from pathlib import Path
from os import path

from pycurv import TriangleGraph
from pycurv import pycurv_io as io


def extract_areas_from_graph(
        tg, csv_file, exclude_borders, gt_file=None, vtp_file=None):
    # If don't want to include triangles near borders, filter out those
    if exclude_borders > 0:
        tg.find_vertices_near_border(exclude_borders, purge=True)

    # Saving the changes into graph and surface files, if specified:
    if gt_file is not None:
        tg.graph.save(gt_file)
    if vtp_file is not None:
        # Transforming the resulting graph to a surface with triangles:
        surf = tg.graph_to_triangle_poly()
        io.save_vtp(surf, vtp_file)

    # Getting areas from the graph:
    triangle_areas = tg.get_vertex_property_array("area")

    # Writing all the curvature values and errors into a csv file:
    df = pd.DataFrame()
    df["triangleAreas"] = triangle_areas
    df.to_csv(csv_file, sep=';')


if __name__ == "__main__":
    folders = ["ER-Mito", "Vac-Nuc"]
    membranes = ["ER", "Mito", "Vac", "Nuc"]
    folders = ["ER-Mito", "Vac-Nuc"]
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/"

    for folder in folders:
        print(folder)
        fold = path.join(base_fold, folder)
        fold_p = Path(fold)
        # iterate over all subfolders
        subfolds_p = [x for x in fold_p.iterdir() if x.is_dir()]
        for subfold_p in subfolds_p:
            for membrane in membranes:
                graph_files_p = list(subfold_p.glob('*.{}.gt'.format(membrane)))
                if len(graph_files_p) == 1:
                    print("\t{}".format(membrane))
                    graph_file_p = graph_files_p[0]
                    graph_file = str(graph_file_p)
                    print("\t\t{}".format(graph_file))
                    csv_file = graph_file.replace(
                        ".gt", ".triangle_areas_excluding1borders.csv")
                    tg = TriangleGraph()
                    tg.graph = load_graph(graph_file)
                    extract_areas_from_graph(tg, csv_file, exclude_borders=1)