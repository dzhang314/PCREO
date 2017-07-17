import itertools
import gmpy2
import random
import sys


################################################################################


def dot_product(v, w):
    dot = v[0] * w[0]
    for i in range(1, len(v)):
        dot += v[i] * w[i]
    return dot


def squared_distance(v, w):
    dist = gmpy2.square(v[0] - w[0])
    for i in range(1, len(v)):
        dist += gmpy2.square(v[i] - w[i])
    return dist


def squared_difference(v, w):
    diff = [v[i] - w[i] for i in range(len(v))]
    dist = gmpy2.square(diff[0])
    for i in range(1, len(v)):
        dist += gmpy2.square(diff[i])
    return dist, diff


def rec_norm(v):
    norm_sq = gmpy2.square(v[0])
    for i in range(1, len(v)):
        norm_sq += gmpy2.square(v[i])
    return gmpy2.rec_sqrt(norm_sq)


def normalize_each_in_place(points):
    for point in points:
        norm_factor = rec_norm(point)
        for k in range(len(point)):
            point[k] *= norm_factor
    return None


################################################################################


def double_dot_product(v, w):
    dot = dot_product(v[0], w[0])
    for i in range(1, len(v)):
        dot += dot_product(v[i], w[i])
    return dot


def double_subtract(v, w):
    m = len(v)
    n = len(v[0])
    return [[v[i][j] - w[i][j]
             for j in range(n)]
            for i in range(m)]


def double_subtract_in_place(v, a, w):
    m = len(v)
    n = len(v[0])
    for i in range(m):
        for j in range(n):
            v[i][j] -= a * w[i][j]
    return None


def double_symmetric_rank_2_update_in_place(a, c, x, y):
    m = len(a)
    n = len(a[0])
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    a[i][j][k][l] += c * (
                        x[i][j] * y[k][l] + y[i][j] * x[k][l])
    return None


def double_matrix_multiply(a, b):
    m = len(a)
    n = len(a[0])
    c = [[gmpy2.zero()
          for _ in range(n)]
         for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    c[i][j] += a[i][j][k][l] * b[k][l]
    return c


################################################################################


def riesz_energy(points, s):
    if s == 1:
        energy = gmpy2.rec_sqrt(squared_distance(points[0], points[1]))
        for j in range(2, len(points)):
            energy += gmpy2.rec_sqrt(squared_distance(points[0], points[j]))
        for i in range(1, len(points)):
            for j in range(i + 1, len(points)):
                energy += gmpy2.rec_sqrt(squared_distance(points[i], points[j]))
    elif s == 2:
        energy = 1 / squared_distance(points[0], points[1])
        for j in range(2, len(points)):
            energy += 1 / squared_distance(points[0], points[j])
        for i in range(1, len(points)):
            for j in range(i + 1, len(points)):
                energy += 1 / squared_distance(points[i], points[j])
    else:
        energy = squared_distance(points[0], points[1]) ** (-s/2)
        for j in range(2, len(points)):
            energy += squared_distance(points[0], points[j]) ** (-s/2)
        for i in range(1, len(points)):
            for j in range(i + 1, len(points)):
                energy += squared_distance(points[i], points[j]) ** (-s/2)
    return energy


def riesz_force(points, s):
    point_indices = range(len(points))
    dim_indices = range(len(points[0]))
    force = [[gmpy2.zero()
              for _ in dim_indices]
             for _ in point_indices]
    if s == 2:
        for i in point_indices:
            for j in point_indices:
                if i != j:
                    dist, disp = squared_difference(points[i], points[j])
                    dist = 2 / gmpy2.square(dist)
                    for k in dim_indices:
                        force[i][k] += dist * disp[k]
    elif s == 1:
        for i in point_indices:
            for j in point_indices:
                if i != j:
                    dist, disp = squared_difference(points[i], points[j])
                    dist = gmpy2.rec_sqrt(dist) / dist
                    for k in dim_indices:
                        force[i][k] += dist * disp[k]
    else:
        for i in point_indices:
            for j in point_indices:
                if i != j:
                    dist, disp = squared_difference(points[i], points[j])
                    dist = s * dist ** (-s/2 - 1)
                    for k in dim_indices:
                        force[i][k] += dist * disp[k]
    return force


################################################################################


def constrained_step(points, step_size, step_direction):
    point_indices = range(len(points))
    dim_indices = range(len(points[0]))
    new_points = [[points[i][k] + step_size * step_direction[i][k]
                   for k in dim_indices]
                  for i in point_indices]
    normalize_each_in_place(new_points)
    return new_points


def constrain_force_in_place(points, force):
    point_indices = range(len(points))
    dim_indices = range(len(points[0]))
    for i in point_indices:
        dot = dot_product(force[i], points[i])
        for k in dim_indices:
            force[i][k] -= dot * points[i][k]
    return None


################################################################################


def bfgs_update_inverse_hessian_in_place(
        inv_hess, delta_points, delta_gradient):
    gamma = double_dot_product(delta_gradient, delta_points)
    kappa = double_matrix_multiply(inv_hess, delta_gradient)
    theta = double_dot_product(delta_gradient, kappa)
    sigma = (gamma + theta) / (gamma * gamma)
    double_subtract_in_place(kappa, gamma * sigma / 2, delta_points)
    double_symmetric_rank_2_update_in_place(
        inv_hess, -1 / gamma, kappa, delta_points)
    return None


def quadratic_line_search(points, s, initial_energy,
                          step_direction, initial_step_size):
    step_size = initial_step_size
    new_points = constrained_step(points, step_size, step_direction)
    new_energy = riesz_energy(new_points, s)
    if new_energy < initial_energy:
        num_increases = 0
        while True:
            newer_points = constrained_step(
                points, 2 * step_size, step_direction)
            newer_energy = riesz_energy(newer_points, s)
            if newer_energy >= new_energy:
                break
            else:
                step_size = 2 * step_size
                new_points = newer_points
                new_energy = newer_energy
                num_increases += 1
                if num_increases >= 4:
                    return step_size
        optimal_step_size = (step_size / 2) * (
            4 * new_energy - newer_energy - 3 * initial_energy) / (
            2 * new_energy - newer_energy - initial_energy)
        if 0 < optimal_step_size < step_size:
            return optimal_step_size
        else:
            return step_size / 2
    else:
        while True:
            newer_points = constrained_step(
                points, step_size / 2, step_direction)
            newer_energy = riesz_energy(newer_points, s)
            if newer_energy <= initial_energy:
                break
            else:
                step_size = step_size / 2
                if step_size == 0:
                    return 0
                new_points = newer_points
                new_energy = newer_energy
        optimal_step_size = (step_size / 4) * (
            new_energy - 4 * newer_energy + 3 * initial_energy) / (
            new_energy - 2 * newer_energy + initial_energy)
        if 0 < optimal_step_size < step_size:
            return optimal_step_size
        else:
            return step_size / 2


################################################################################


if __name__ == '__main__':
    random.seed()
    gmpy2.get_context().precision = 100
    epsilon = gmpy2.exp2(-gmpy2.get_context().precision + 20)

    num_points = 100
    num_dims = 3

    point_indices = range(num_points)
    dim_indices = range(num_dims)
    all_indices = range(num_points * num_dims)

    points = [[gmpy2.mpfr(random.gauss(0, 1))
               for _ in dim_indices]
              for _ in point_indices]
    normalize_each_in_place(points)

    energy = riesz_energy(points, 1)
    force = riesz_force(points, 1)
    constrain_force_in_place(points, force)
    inv_hess = [[[[gmpy2.mpfr(i == k and j == l)
                   for l in dim_indices]
                  for k in point_indices]
                 for j in dim_indices]
                for i in point_indices]
    step_direction = double_matrix_multiply(inv_hess, force)
    step_size = gmpy2.exp2(-20)

    while True:
        step_size = quadratic_line_search(
            points, 1, energy, step_direction, step_size)
        new_points = constrained_step(points, step_size, step_direction)
        delta_points = double_subtract(new_points, points)
        new_energy = riesz_energy(new_points, 1)
        print(new_energy)
        if (energy - new_energy) / (energy + new_energy) < epsilon:
            gmpy2.get_context().precision += 100
            epsilon = gmpy2.exp2(-gmpy2.get_context().precision + 20)
        new_force = riesz_force(new_points, 1)
        constrain_force_in_place(new_points, new_force)
        delta_gradient = double_subtract(force, new_force)
        bfgs_update_inverse_hessian_in_place(
            inv_hess, delta_points, delta_gradient)
        step_direction = double_matrix_multiply(inv_hess, new_force)
        points = new_points
        energy = new_energy
        force = new_force
