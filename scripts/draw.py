#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python standard library imports
import collections
import itertools
import os
import subprocess
import tempfile

# Third-party imports
import gmpy2    # pylint: disable=import-error
import networkx # pylint: disable=import-error

# Project-specific imports
from mpfr_vector import MPFRVector


POVRAY_PATH = 'povray'
QCONVEX_PATH = 'qconvex'

PCreoRunHeader = collections.namedtuple('PCreoRunHeader',
                                        ('dim', 's', 'num_points'))

PCreoRunResult = collections.namedtuple('PCreoRunResult',
                                        ('energy', 'rms_gradient'))

PCreoRunRecord = collections.namedtuple('PCreoRunRecord',
                                        ('header', 'results', 'points'))


def read_pcreo_file(pathlike):
    with open(pathlike) as pcreo_file:
        header, result, *points = [
            line.strip().replace(',', ' ').split()
            for line in pcreo_file.readlines()]
    assert len(header) == 3
    header = PCreoRunHeader(int(header[0]), header[1], int(header[2]))
    embedded_dim = header.dim + 1
    result = PCreoRunResult(*tuple(map(gmpy2.mpfr, result)))
    points = tuple(MPFRVector(map(gmpy2.mpfr, point))
                   for point in points)
    assert all(len(point) == embedded_dim for point in points)
    return PCreoRunRecord(header, result, points)


def convex_hull_facets(points):
    dim_set = set(map(len, points))
    assert len(dim_set) == 1
    dim, = dim_set
    num_points = len(points)
    points_str = '\n'.join(' '.join(map('{0:+.16e}'.format, point))
                           for point in points)
    with tempfile.TemporaryFile(mode='w+') as qconvex_input:
        qconvex_input.write(str(dim) + '\n')
        qconvex_input.write(str(num_points) + '\n')
        qconvex_input.write(points_str + '\n')
        qconvex_input.flush()
        qconvex_input.seek(0)
        qconvex_info = subprocess.run((QCONVEX_PATH, 'Fv'),
                                      stdin=qconvex_input,
                                      stdout=subprocess.PIPE)
    assert qconvex_info.returncode == 0
    qconvex_lines = qconvex_info.stdout.decode('utf-8').splitlines()[1:]
    qconvex_facets = tuple(tuple(map(int, line.split()[1:]))
                           for line in qconvex_lines
                           if line.strip())
    return qconvex_facets


def nearest_neighbor_graph(run_record):
    facets = convex_hull_facets(run_record.points)
    nngraph = networkx.Graph()
    nngraph.add_nodes_from(range(len(run_record.points)))
    for facet in facets:
        assert len(facet) == 3
        nngraph.add_edges_from(itertools.combinations(facet, 2))
    networkx.set_node_attributes(nngraph, 'weight',
                                 dict(nngraph.degree_iter()))
    return nngraph


def remove_nodes_of_degree(n, graph):
    target_nodes = [node
                    for node, degree in graph.degree_iter()
                    if degree == n]
    graph.remove_nodes_from(target_nodes)
    return None


def equivalence_classes(items, eq_func):
    classes = []
    for item in items:
        for eq_class in classes:
            if eq_func(item, eq_class[0]):
                eq_class.append(item)
                break
        else:
            classes.append([item])
    return tuple(map(tuple, classes))


def defect_classes(run_record):
    nngraph = nearest_neighbor_graph(run_record)
    remove_nodes_of_degree(6, nngraph)
    match_func = networkx.algorithms.isomorphism.categorical_node_match(
        'weight', None)
    equiv_func = lambda g, h: networkx.is_isomorphic(g, h,
                                                     node_match=match_func)
    return equivalence_classes(networkx.connected_component_subgraphs(nngraph),
                               equiv_func)


def orthonormal_basis(v):
    if len(v) == 0: return []
    sign = -1 if v[0] < 0 else +1
    w = MPFRVector([v[0] + sign * v.norm(), *v[1:]])
    r = 2 / w.norm_squared()
    return [MPFRVector((i == j) - r * w[i] * w[j]
                       for j in range(len(v)))
            for i in range(len(v))]


def centroid(points):
    dim_set = set(map(len, points))
    assert len(dim_set) == 1
    dim, = dim_set
    cent = [gmpy2.zero() for _ in range(dim)]
    for point in points:
        for i in range(dim):
            cent[i] += point[i]
    num_points = len(points)
    for i in range(dim):
        cent[i] /= num_points
    return tuple(cent)


def circumcenter(a, b, c):
    v = b - a
    w = c - a
    d = v @ w
    q = w - (d / v.norm_squared()) * v
    t = (w.norm_squared() - d) / (2 * (q @ w))
    return a + gmpy2.mpfr(0.5) * v + t * q


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# TODO: Can this iterator be written without a cast?
def circular_pairwise(items):
    return pairwise(items + type(items)([items[0]]))


def povray_scalar(c):
    return '{0:+.16e}'.format(c)


def povray_vector(v):
    assert len(v) == 3
    return '<' + ', '.join(map(povray_scalar, v)) + '>'


def povray_smooth_triangle(a, b, c, color):
    return ("smooth_triangle {{\n    {0},\n    {0},\n    {1},\n    {1},\n" +
            "    {2},\n    {2}\n    pigment {{ color rgb {3} }}\n}}").format(
                povray_vector(a), povray_vector(b), povray_vector(c),
                povray_vector(color))


def povray_cylinder(a, b, radius, color):
    return ("cylinder {{\n    {0},\n    {1},\n    {2}\n" +
            "    pigment {{ color rgb {3} }}\n}}").format(
                povray_vector(a), povray_vector(b), povray_scalar(radius),
                povray_vector(color))


def povray_disc(center, radius, color):
    return ("disc {{\n    {0},\n    {0},\n    {1}\n" +
            "    pigment {{ color rgb {2} }}\n}}").format(
                povray_vector(center), povray_scalar(radius),
                povray_vector(color))


def povray_primitives(run_record):
    nngraph = nearest_neighbor_graph(run_record)
    result = ["#version 3.7;", "global_settings { assumed_gamma 1 }"]
    for node_index, adjacency_list in nngraph.adjacency_iter():
        center = run_record.points[node_index]
        result.append(povray_disc(center, 0.004, [0, 0, 0]))
        center_basis = orthonormal_basis(center)
        def projected_angle(i):
            displacement = run_record.points[i] - center
            x = displacement @ center_basis[1]
            y = displacement @ center_basis[2]
            return gmpy2.atan2(y, x)
        sorted_neighbor_indices = sorted(adjacency_list, key=projected_angle)
        voronoi_vertices = tuple(
            circumcenter(center,
                         run_record.points[i],
                         run_record.points[j]).normalized()
            for i, j in circular_pairwise(sorted_neighbor_indices))
        if len(voronoi_vertices) == 4:
            color = [1, 1, 0]
        elif len(voronoi_vertices) == 5:
            color = [1, 0, 0]
        elif len(voronoi_vertices) == 6:
            color = [0, 1, 0]
        elif len(voronoi_vertices) == 7:
            color = [0, 0, 1]
        else:
            color = [1, 1, 1]
        for v, w in circular_pairwise(voronoi_vertices):
            result.append(povray_smooth_triangle(center, v, w, color))
            result.append(povray_cylinder(v, w, 0.002, [0, 0, 0]))
    result.append("light_source { <5, 30, -30> color <1, 1, 1> }")
    result.append("camera {\n    up <0, 1, 0>\n    right <1, 0, 0>\n"
                  "    location <+0.7, +0.7, -2.7>\n"
                  "    look_at  <-0.7, -0.7, +2.7>\n}")
    return result


def povray_draw(run_record, resolution, filename):
    with tempfile.NamedTemporaryFile(mode='w+') as povray_input:
        for line in povray_primitives(run_record):
            povray_input.write(line)
            povray_input.write('\n')
        povray_input.flush()
        povray_input.seek(0)
        subprocess.check_call(
            (POVRAY_PATH, '-D', '+FN8', '+UA', # '+A0.1', '+AM2', '+R2',
             '+H' + str(resolution),
             '+W' + str(resolution),
             '+I' + povray_input.name,
             '+O' + filename + '.png'),
            stderr=subprocess.DEVNULL)


def main():
    gmpy2.get_context().precision = 100
    for dir_entry in os.scandir():
        try:
            run_record = read_pcreo_file(dir_entry)
            print("Successfully read PCreo output file:", dir_entry.name)
        except:
            print("File", dir_entry.name, "is not a PCreo output file.")
            continue
        print(len(defect_classes(run_record)))
        povray_draw(run_record, 800, dir_entry.name)


if __name__ == "__main__": main()
