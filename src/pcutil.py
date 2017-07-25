import collections
import gmpy2
import os
import subprocess
import tempfile


def dot_product(v, w):
    assert len(v) == len(w)
    if len(v) == 0:
        return gmpy2.zero()
    else:
        dot = v[0] * w[0]
        for i in range(1, len(v)):
            dot += v[i] * w[i]
        return dot


def squared_distance(v, w):
    assert len(v) == len(w)
    if len(v) == 0:
        return gmpy2.zero()
    else:
        dist = gmpy2.square(v[0] - w[0])
        for i in range(1, len(v)):
            dist += gmpy2.square(v[i] - w[i])
        return dist


def squared_difference(v, w):
    assert len(v) == len(w)
    if len(v) == 0:
        return gmpy2.zero(), ()
    else:
        diff = tuple(v[i] - w[i] for i in range(len(v)))
        dist = gmpy2.square(diff[0])
        for i in range(1, len(v)):
            dist += gmpy2.square(diff[i])
        return dist, diff


def rec_norm(v):
    if len(v) == 0:
        return gmpy2.inf()
    else:
        norm_sq = gmpy2.square(v[0])
        for i in range(1, len(v)):
            norm_sq += gmpy2.square(v[i])
        return gmpy2.rec_sqrt(norm_sq)


def normalized(v):
    r = rec_norm(v)
    return tuple(r * c for c in v)


################################################################################


def riesz_energy(points, s):
    num_points = len(points)
    if num_points < 2:
        return gmpy2.zero()
    if s == 1:
        energy = gmpy2.rec_sqrt(squared_distance(points[0], points[1]))
        for j in range(2, num_points):
            energy += gmpy2.rec_sqrt(squared_distance(points[0], points[j]))
        for i in range(1, num_points):
            for j in range(i + 1, num_points):
                energy += gmpy2.rec_sqrt(squared_distance(points[i], points[j]))
    elif s == 2:
        energy = 1 / squared_distance(points[0], points[1])
        for j in range(2, num_points):
            energy += 1 / squared_distance(points[0], points[j])
        for i in range(1, num_points):
            for j in range(i + 1, num_points):
                energy += 1 / squared_distance(points[i], points[j])
    else:
        energy = squared_distance(points[0], points[1]) ** (-s/2)
        for j in range(2, num_points):
            energy += squared_distance(points[0], points[j]) ** (-s/2)
        for i in range(1, num_points):
            for j in range(i + 1, num_points):
                energy += squared_distance(points[i], points[j]) ** (-s/2)
    return energy


################################################################################


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


def mvmul(mat, vec):
    m = len(mat)
    n_set = set(map(len, mat))
    assert len(n_set) == 1
    n, = n_set
    assert len(vec) == n
    ans = [gmpy2.zero() for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ans[i] += mat[i][j] * vec[j]
    return tuple(ans)


def determinant_3(mat):
    assert len(mat) == 3
    assert all(len(row) == 3 for row in mat)
    return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) +
            mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) +
            mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]))


################################################################################


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
    points = tuple(tuple(map(gmpy2.mpfr, point))
                   for point in points)
    assert all(len(point) == embedded_dim for point in points)
    return PCreoRunRecord(header, result, points)


################################################################################


def icosahedron_vertices():
    zero = gmpy2.zero()
    one = gmpy2.mpfr(1)
    phi = (gmpy2.sqrt(5) + 1) / 2
    return (
        (+one, +phi, zero),
        (+one, -phi, zero),
        (-one, +phi, zero),
        (-one, -phi, zero),
        (zero, +one, +phi),
        (zero, +one, -phi),
        (zero, -one, +phi),
        (zero, -one, -phi),
        (+phi, zero, +one),
        (+phi, zero, -one),
        (-phi, zero, +one),
        (-phi, zero, -one))


def icosahedron_edge_centers():
    edge_indices = (
        (1, 3), (1, 5), (1, 7), (1, 9), (1, 11),
        (2, 4), (2, 5), (2, 7), (2, 10), (2, 12),
        (3, 6), (3, 8), (3, 9), (3, 11),
        (4, 6), (4, 8), (4, 10), (4, 12),
        (5, 7), (5, 9), (5, 10),
        (6, 8), (6, 9), (6, 10),
        (7, 11), (7, 12),
        (8, 11), (8, 12),
        (9, 10),
        (11, 12))
    vertices = icosahedron_vertices()
    return tuple(
        centroid([vertices[i-1], vertices[j-1]])
        for i, j in edge_indices)


def icosahedron_face_centers():
    face_indices = (
        (1, 3, 9), (1, 9, 5), (1, 5, 7), (1, 7, 11), (1, 11, 3),
        (3, 6, 9), (9, 6, 10), (9, 10, 5), (5, 10, 2), (5, 2, 7),
        (7, 2, 12), (7, 12, 11), (11, 12, 8), (11, 8, 3), (3, 8, 6),
        (4, 2, 10), (4, 10, 6), (4, 6, 8), (4, 8, 12), (4, 12, 2))
    vertices = icosahedron_vertices()
    return tuple(
        centroid([vertices[i-1], vertices[j-1], vertices[k-1]])
        for i, j, k in face_indices)


def icosahedral_points_of_high_symmetry(num_points):
    if num_points % 60 == 0:
        return ()
    elif num_points % 60 == 2:
        return tuple(map(normalized, icosahedron_vertices())) + \
            tuple(map(normalized, icosahedron_edge_centers())) + \
            tuple(map(normalized, icosahedron_face_centers()))
    elif num_points % 60 == 12:
        return tuple(map(normalized, icosahedron_vertices()))
    elif num_points % 60 == 20:
        return tuple(map(normalized, icosahedron_face_centers()))
    elif num_points % 60 == 30:
        return tuple(map(normalized, icosahedron_edge_centers()))
    elif num_points % 60 == 32:
        return tuple(map(normalized, icosahedron_vertices())) + \
            tuple(map(normalized, icosahedron_face_centers()))
    elif num_points % 60 == 42:
        return tuple(map(normalized, icosahedron_vertices())) + \
            tuple(map(normalized, icosahedron_edge_centers()))
    elif num_points % 60 == 50:
        return tuple(map(normalized, icosahedron_edge_centers())) + \
            tuple(map(normalized, icosahedron_face_centers()))
    else:
        raise ValueError("An icosahedrally symmetric point configuration "
                         "with the requested number of points does not exist")


def full_icosahedral_configuration(run_record):
    high_sym_points = icosahedral_points_of_high_symmetry(
        run_record.header.num_points)
    if len(high_sym_points) == 0:
        vertex_indices = range(0, 0)
        edge_indices = range(0, 0)
        face_indices = range(0, 0)
    elif len(high_sym_points) == 12:
        vertex_indices = range(0, 12)
        edge_indices = range(0, 0)
        face_indices = range(0, 0)
    elif len(high_sym_points) == 20:
        vertex_indices = range(0, 0)
        edge_indices = range(0, 0)
        face_indices = range(0, 20)
    elif len(high_sym_points) == 30:
        vertex_indices = range(0, 0)
        edge_indices = range(0, 30)
        face_indices = range(0, 0)
    elif len(high_sym_points) == 32:
        vertex_indices = range(0, 12)
        edge_indices = range(0, 0)
        face_indices = range(12, 32)
    elif len(high_sym_points) == 42:
        vertex_indices = range(0, 12)
        edge_indices = range(12, 42)
        face_indices = range(0, 0)
    elif len(high_sym_points) == 50:
        vertex_indices = range(0, 0)
        edge_indices = range(0, 30)
        face_indices = range(30, 50)
    elif len(high_sym_points) == 62:
        vertex_indices = range(0, 12)
        edge_indices = range(12, 42)
        face_indices = range(42, 62)
    else:
        raise ValueError("An icosahedrally symmetric point configuration "
                         "with the requested number of points does not exist")
    full_points = high_sym_points + tuple(mvmul(g, p)
                                          for p in run_record.points
                                          for g in chiral_icosahedral_group())
    return full_points, (vertex_indices, edge_indices, face_indices)


def chiral_icosahedral_group():
    zero = gmpy2.zero()
    one = gmpy2.mpfr(1)
    half = one / 2
    psi = (gmpy2.sqrt(5) - 1) / 4
    phi = (gmpy2.sqrt(5) + 1) / 4
    return (
        ((+one, zero, zero), (zero, +one, zero), (zero, zero, +one)),
        ((+one, zero, zero), (zero, -one, zero), (zero, zero, -one)),
        ((-one, zero, zero), (zero, +one, zero), (zero, zero, -one)),
        ((-one, zero, zero), (zero, -one, zero), (zero, zero, +one)),
        ((zero, +one, zero), (zero, zero, +one), (+one, zero, zero)),
        ((zero, +one, zero), (zero, zero, -one), (-one, zero, zero)),
        ((zero, -one, zero), (zero, zero, +one), (-one, zero, zero)),
        ((zero, -one, zero), (zero, zero, -one), (+one, zero, zero)),
        ((zero, zero, +one), (+one, zero, zero), (zero, +one, zero)),
        ((zero, zero, +one), (-one, zero, zero), (zero, -one, zero)),
        ((zero, zero, -one), (+one, zero, zero), (zero, -one, zero)),
        ((zero, zero, -one), (-one, zero, zero), (zero, +one, zero)),
        ((+half, +psi, +phi), (+psi, +phi, -half), (-phi, +half, +psi)),
        ((+half, +psi, +phi), (-psi, -phi, +half), (+phi, -half, -psi)),
        ((+half, +psi, -phi), (+psi, +phi, +half), (+phi, -half, +psi)),
        ((+half, +psi, -phi), (-psi, -phi, -half), (-phi, +half, -psi)),
        ((+half, -psi, +phi), (+psi, -phi, -half), (+phi, +half, -psi)),
        ((+half, -psi, +phi), (-psi, +phi, +half), (-phi, -half, +psi)),
        ((+half, -psi, -phi), (+psi, -phi, +half), (-phi, -half, -psi)),
        ((+half, -psi, -phi), (-psi, +phi, -half), (+phi, +half, +psi)),
        ((-half, +psi, +phi), (+psi, -phi, +half), (+phi, +half, +psi)),
        ((-half, +psi, +phi), (-psi, +phi, -half), (-phi, -half, -psi)),
        ((-half, +psi, -phi), (+psi, -phi, -half), (-phi, -half, +psi)),
        ((-half, +psi, -phi), (-psi, +phi, +half), (+phi, +half, -psi)),
        ((-half, -psi, +phi), (+psi, +phi, +half), (-phi, +half, -psi)),
        ((-half, -psi, +phi), (-psi, -phi, -half), (+phi, -half, +psi)),
        ((-half, -psi, -phi), (+psi, +phi, -half), (+phi, -half, -psi)),
        ((-half, -psi, -phi), (-psi, -phi, +half), (-phi, +half, +psi)),
        ((+phi, +half, +psi), (+half, -psi, -phi), (-psi, +phi, -half)),
        ((+phi, +half, +psi), (-half, +psi, +phi), (+psi, -phi, +half)),
        ((+phi, +half, -psi), (+half, -psi, +phi), (+psi, -phi, -half)),
        ((+phi, +half, -psi), (-half, +psi, -phi), (-psi, +phi, +half)),
        ((+phi, -half, +psi), (+half, +psi, -phi), (+psi, +phi, +half)),
        ((+phi, -half, +psi), (-half, -psi, +phi), (-psi, -phi, -half)),
        ((+phi, -half, -psi), (+half, +psi, +phi), (-psi, -phi, +half)),
        ((+phi, -half, -psi), (-half, -psi, -phi), (+psi, +phi, -half)),
        ((-phi, +half, +psi), (+half, +psi, +phi), (+psi, +phi, -half)),
        ((-phi, +half, +psi), (-half, -psi, -phi), (-psi, -phi, +half)),
        ((-phi, +half, -psi), (+half, +psi, -phi), (-psi, -phi, -half)),
        ((-phi, +half, -psi), (-half, -psi, +phi), (+psi, +phi, +half)),
        ((-phi, -half, +psi), (+half, -psi, +phi), (-psi, +phi, +half)),
        ((-phi, -half, +psi), (-half, +psi, -phi), (+psi, -phi, -half)),
        ((-phi, -half, -psi), (+half, -psi, -phi), (+psi, -phi, +half)),
        ((-phi, -half, -psi), (-half, +psi, +phi), (-psi, +phi, -half)),
        ((+psi, +phi, +half), (+phi, -half, +psi), (+half, +psi, -phi)),
        ((+psi, +phi, +half), (-phi, +half, -psi), (-half, -psi, +phi)),
        ((+psi, +phi, -half), (+phi, -half, -psi), (-half, -psi, -phi)),
        ((+psi, +phi, -half), (-phi, +half, +psi), (+half, +psi, +phi)),
        ((+psi, -phi, +half), (+phi, +half, +psi), (-half, +psi, +phi)),
        ((+psi, -phi, +half), (-phi, -half, -psi), (+half, -psi, -phi)),
        ((+psi, -phi, -half), (+phi, +half, -psi), (+half, -psi, +phi)),
        ((+psi, -phi, -half), (-phi, -half, +psi), (-half, +psi, -phi)),
        ((-psi, +phi, +half), (+phi, +half, -psi), (-half, +psi, -phi)),
        ((-psi, +phi, +half), (-phi, -half, +psi), (+half, -psi, +phi)),
        ((-psi, +phi, -half), (+phi, +half, +psi), (+half, -psi, -phi)),
        ((-psi, +phi, -half), (-phi, -half, -psi), (-half, +psi, +phi)),
        ((-psi, -phi, +half), (+phi, -half, -psi), (+half, +psi, +phi)),
        ((-psi, -phi, +half), (-phi, +half, +psi), (-half, -psi, -phi)),
        ((-psi, -phi, -half), (+phi, -half, +psi), (-half, -psi, +phi)),
        ((-psi, -phi, -half), (-phi, +half, -psi), (+half, +psi, -phi)))


def full_icosahedral_group():
    zero = gmpy2.zero()
    one = gmpy2.mpfr(1)
    half = one / 2
    psi = (gmpy2.sqrt(5) - 1) / 4
    phi = (gmpy2.sqrt(5) + 1) / 4
    return (
        ((+one, zero, zero), (zero, +one, zero), (zero, zero, +one)),
        ((+one, zero, zero), (zero, +one, zero), (zero, zero, -one)),
        ((+one, zero, zero), (zero, -one, zero), (zero, zero, +one)),
        ((+one, zero, zero), (zero, -one, zero), (zero, zero, -one)),
        ((-one, zero, zero), (zero, +one, zero), (zero, zero, +one)),
        ((-one, zero, zero), (zero, +one, zero), (zero, zero, -one)),
        ((-one, zero, zero), (zero, -one, zero), (zero, zero, +one)),
        ((-one, zero, zero), (zero, -one, zero), (zero, zero, -one)),
        ((zero, +one, zero), (zero, zero, +one), (+one, zero, zero)),
        ((zero, +one, zero), (zero, zero, +one), (-one, zero, zero)),
        ((zero, +one, zero), (zero, zero, -one), (+one, zero, zero)),
        ((zero, +one, zero), (zero, zero, -one), (-one, zero, zero)),
        ((zero, -one, zero), (zero, zero, +one), (+one, zero, zero)),
        ((zero, -one, zero), (zero, zero, +one), (-one, zero, zero)),
        ((zero, -one, zero), (zero, zero, -one), (+one, zero, zero)),
        ((zero, -one, zero), (zero, zero, -one), (-one, zero, zero)),
        ((zero, zero, +one), (+one, zero, zero), (zero, +one, zero)),
        ((zero, zero, +one), (+one, zero, zero), (zero, -one, zero)),
        ((zero, zero, +one), (-one, zero, zero), (zero, +one, zero)),
        ((zero, zero, +one), (-one, zero, zero), (zero, -one, zero)),
        ((zero, zero, -one), (+one, zero, zero), (zero, +one, zero)),
        ((zero, zero, -one), (+one, zero, zero), (zero, -one, zero)),
        ((zero, zero, -one), (-one, zero, zero), (zero, +one, zero)),
        ((zero, zero, -one), (-one, zero, zero), (zero, -one, zero)),
        ((+half, +psi, +phi), (+psi, +phi, -half), (+phi, -half, -psi)),
        ((+half, +psi, +phi), (+psi, +phi, -half), (-phi, +half, +psi)),
        ((+half, +psi, +phi), (-psi, -phi, +half), (+phi, -half, -psi)),
        ((+half, +psi, +phi), (-psi, -phi, +half), (-phi, +half, +psi)),
        ((+half, +psi, -phi), (+psi, +phi, +half), (+phi, -half, +psi)),
        ((+half, +psi, -phi), (+psi, +phi, +half), (-phi, +half, -psi)),
        ((+half, +psi, -phi), (-psi, -phi, -half), (+phi, -half, +psi)),
        ((+half, +psi, -phi), (-psi, -phi, -half), (-phi, +half, -psi)),
        ((+half, -psi, +phi), (+psi, -phi, -half), (+phi, +half, -psi)),
        ((+half, -psi, +phi), (+psi, -phi, -half), (-phi, -half, +psi)),
        ((+half, -psi, +phi), (-psi, +phi, +half), (+phi, +half, -psi)),
        ((+half, -psi, +phi), (-psi, +phi, +half), (-phi, -half, +psi)),
        ((+half, -psi, -phi), (+psi, -phi, +half), (+phi, +half, +psi)),
        ((+half, -psi, -phi), (+psi, -phi, +half), (-phi, -half, -psi)),
        ((+half, -psi, -phi), (-psi, +phi, -half), (+phi, +half, +psi)),
        ((+half, -psi, -phi), (-psi, +phi, -half), (-phi, -half, -psi)),
        ((-half, +psi, +phi), (+psi, -phi, +half), (+phi, +half, +psi)),
        ((-half, +psi, +phi), (+psi, -phi, +half), (-phi, -half, -psi)),
        ((-half, +psi, +phi), (-psi, +phi, -half), (+phi, +half, +psi)),
        ((-half, +psi, +phi), (-psi, +phi, -half), (-phi, -half, -psi)),
        ((-half, +psi, -phi), (+psi, -phi, -half), (+phi, +half, -psi)),
        ((-half, +psi, -phi), (+psi, -phi, -half), (-phi, -half, +psi)),
        ((-half, +psi, -phi), (-psi, +phi, +half), (+phi, +half, -psi)),
        ((-half, +psi, -phi), (-psi, +phi, +half), (-phi, -half, +psi)),
        ((-half, -psi, +phi), (+psi, +phi, +half), (+phi, -half, +psi)),
        ((-half, -psi, +phi), (+psi, +phi, +half), (-phi, +half, -psi)),
        ((-half, -psi, +phi), (-psi, -phi, -half), (+phi, -half, +psi)),
        ((-half, -psi, +phi), (-psi, -phi, -half), (-phi, +half, -psi)),
        ((-half, -psi, -phi), (+psi, +phi, -half), (+phi, -half, -psi)),
        ((-half, -psi, -phi), (+psi, +phi, -half), (-phi, +half, +psi)),
        ((-half, -psi, -phi), (-psi, -phi, +half), (+phi, -half, -psi)),
        ((-half, -psi, -phi), (-psi, -phi, +half), (-phi, +half, +psi)),
        ((+phi, +half, +psi), (+half, -psi, -phi), (+psi, -phi, +half)),
        ((+phi, +half, +psi), (+half, -psi, -phi), (-psi, +phi, -half)),
        ((+phi, +half, +psi), (-half, +psi, +phi), (+psi, -phi, +half)),
        ((+phi, +half, +psi), (-half, +psi, +phi), (-psi, +phi, -half)),
        ((+phi, +half, -psi), (+half, -psi, +phi), (+psi, -phi, -half)),
        ((+phi, +half, -psi), (+half, -psi, +phi), (-psi, +phi, +half)),
        ((+phi, +half, -psi), (-half, +psi, -phi), (+psi, -phi, -half)),
        ((+phi, +half, -psi), (-half, +psi, -phi), (-psi, +phi, +half)),
        ((+phi, -half, +psi), (+half, +psi, -phi), (+psi, +phi, +half)),
        ((+phi, -half, +psi), (+half, +psi, -phi), (-psi, -phi, -half)),
        ((+phi, -half, +psi), (-half, -psi, +phi), (+psi, +phi, +half)),
        ((+phi, -half, +psi), (-half, -psi, +phi), (-psi, -phi, -half)),
        ((+phi, -half, -psi), (+half, +psi, +phi), (+psi, +phi, -half)),
        ((+phi, -half, -psi), (+half, +psi, +phi), (-psi, -phi, +half)),
        ((+phi, -half, -psi), (-half, -psi, -phi), (+psi, +phi, -half)),
        ((+phi, -half, -psi), (-half, -psi, -phi), (-psi, -phi, +half)),
        ((-phi, +half, +psi), (+half, +psi, +phi), (+psi, +phi, -half)),
        ((-phi, +half, +psi), (+half, +psi, +phi), (-psi, -phi, +half)),
        ((-phi, +half, +psi), (-half, -psi, -phi), (+psi, +phi, -half)),
        ((-phi, +half, +psi), (-half, -psi, -phi), (-psi, -phi, +half)),
        ((-phi, +half, -psi), (+half, +psi, -phi), (+psi, +phi, +half)),
        ((-phi, +half, -psi), (+half, +psi, -phi), (-psi, -phi, -half)),
        ((-phi, +half, -psi), (-half, -psi, +phi), (+psi, +phi, +half)),
        ((-phi, +half, -psi), (-half, -psi, +phi), (-psi, -phi, -half)),
        ((-phi, -half, +psi), (+half, -psi, +phi), (+psi, -phi, -half)),
        ((-phi, -half, +psi), (+half, -psi, +phi), (-psi, +phi, +half)),
        ((-phi, -half, +psi), (-half, +psi, -phi), (+psi, -phi, -half)),
        ((-phi, -half, +psi), (-half, +psi, -phi), (-psi, +phi, +half)),
        ((-phi, -half, -psi), (+half, -psi, -phi), (+psi, -phi, +half)),
        ((-phi, -half, -psi), (+half, -psi, -phi), (-psi, +phi, -half)),
        ((-phi, -half, -psi), (-half, +psi, +phi), (+psi, -phi, +half)),
        ((-phi, -half, -psi), (-half, +psi, +phi), (-psi, +phi, -half)),
        ((+psi, +phi, +half), (+phi, -half, +psi), (+half, +psi, -phi)),
        ((+psi, +phi, +half), (+phi, -half, +psi), (-half, -psi, +phi)),
        ((+psi, +phi, +half), (-phi, +half, -psi), (+half, +psi, -phi)),
        ((+psi, +phi, +half), (-phi, +half, -psi), (-half, -psi, +phi)),
        ((+psi, +phi, -half), (+phi, -half, -psi), (+half, +psi, +phi)),
        ((+psi, +phi, -half), (+phi, -half, -psi), (-half, -psi, -phi)),
        ((+psi, +phi, -half), (-phi, +half, +psi), (+half, +psi, +phi)),
        ((+psi, +phi, -half), (-phi, +half, +psi), (-half, -psi, -phi)),
        ((+psi, -phi, +half), (+phi, +half, +psi), (+half, -psi, -phi)),
        ((+psi, -phi, +half), (+phi, +half, +psi), (-half, +psi, +phi)),
        ((+psi, -phi, +half), (-phi, -half, -psi), (+half, -psi, -phi)),
        ((+psi, -phi, +half), (-phi, -half, -psi), (-half, +psi, +phi)),
        ((+psi, -phi, -half), (+phi, +half, -psi), (+half, -psi, +phi)),
        ((+psi, -phi, -half), (+phi, +half, -psi), (-half, +psi, -phi)),
        ((+psi, -phi, -half), (-phi, -half, +psi), (+half, -psi, +phi)),
        ((+psi, -phi, -half), (-phi, -half, +psi), (-half, +psi, -phi)),
        ((-psi, +phi, +half), (+phi, +half, -psi), (+half, -psi, +phi)),
        ((-psi, +phi, +half), (+phi, +half, -psi), (-half, +psi, -phi)),
        ((-psi, +phi, +half), (-phi, -half, +psi), (+half, -psi, +phi)),
        ((-psi, +phi, +half), (-phi, -half, +psi), (-half, +psi, -phi)),
        ((-psi, +phi, -half), (+phi, +half, +psi), (+half, -psi, -phi)),
        ((-psi, +phi, -half), (+phi, +half, +psi), (-half, +psi, +phi)),
        ((-psi, +phi, -half), (-phi, -half, -psi), (+half, -psi, -phi)),
        ((-psi, +phi, -half), (-phi, -half, -psi), (-half, +psi, +phi)),
        ((-psi, -phi, +half), (+phi, -half, -psi), (+half, +psi, +phi)),
        ((-psi, -phi, +half), (+phi, -half, -psi), (-half, -psi, -phi)),
        ((-psi, -phi, +half), (-phi, +half, +psi), (+half, +psi, +phi)),
        ((-psi, -phi, +half), (-phi, +half, +psi), (-half, -psi, -phi)),
        ((-psi, -phi, -half), (+phi, -half, +psi), (+half, +psi, -phi)),
        ((-psi, -phi, -half), (+phi, -half, +psi), (-half, -psi, +phi)),
        ((-psi, -phi, -half), (-phi, +half, -psi), (+half, +psi, -phi)),
        ((-psi, -phi, -half), (-phi, +half, -psi), (-half, -psi, +phi)))


################################################################################


def squared_reduced_distance(v, w, symmetry_group):
    return min(squared_distance(mvmul(g, v), w)
               for g in symmetry_group)


def reduced_nearest_neighbor_indices(p, q, symmetry_group):
    return (min(enumerate(squared_reduced_distance(v, w, symmetry_group)
                          for w in q),
                key=lambda t: tuple(reversed(t)))
            for v in p)


def reduced_isometric(p, q, symmetry_group, epsilon=1e-10):
    found = [False for _ in range(len(q))]
    for j, dist in reduced_nearest_neighbor_indices(p, q, symmetry_group):
        if dist > gmpy2.square(epsilon):
            return False
        if found[j]:
            return False
        found[j] = True
    return all(found)


################################################################################


def convex_hull_facets(points, qconvex_path='qconvex'):
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
        qconvex_input.seek(0)
        qconvex_info = subprocess.run((qconvex_path, 'Fv'),
                                      stdin=qconvex_input,
                                      stdout=subprocess.PIPE)
    assert qconvex_info.returncode == 0
    qconvex_lines = qconvex_info.stdout.decode('utf-8')\
                                       .split(os.linesep)[1:]
    qconvex_facets = tuple(tuple(map(int, line.split()[1:]))
                           for line in qconvex_lines)
    return qconvex_facets


def icosahedral_nearest_neighbor_graph(run_record, qconvex_path='qconvex'):
    full_points, (vertex_indices, edge_indices, face_indices) = \
        full_icosahedral_configuration(run_record)
    facets = convex_hull_facets(full_points, qconvex_path=qconvex_path)
    nngraph = networkx.Graph()
    for i in range(len(full_points)):
        if i in vertex_indices:
            nngraph.add_node(i, color=1)
        elif i in edge_indices:
            nngraph.add_node(i, color=2)
        elif i in face_indices:
            nngraph.add_node(i, color=3)
        else:
            nngraph.add_node(i, color=0)
    for facet in facets:
        assert len(facet) == 3
        nngraph.add_edges_from(
            itertools.combinations(facet, 2))
    return nngraph
