def riesz_hessian_entry_s(points, s, k, l, m, n):
    if k == m and l == n:
        ans = gmpy2.zero()
        for j in range(k):
            dist_sq = squared_distance(points[j], points[k])
            a = dist_sq ** (-s/2 - 2)
            b = dist_sq - ((s + 2) * (points[j][l] - points[k][l])
                                   * (points[j][n] - points[k][n]))
            ans += a * b
        for j in range(k + 1, len(points)):
            dist_sq = squared_distance(points[j], points[k])
            a = dist_sq ** (-s/2 - 2)
            b = dist_sq - ((s + 2) * (points[j][l] - points[k][l])
                                   * (points[j][n] - points[k][n]))
            ans += a * b
        return -s * ans
    elif k == m and l != n:
        ans = gmpy2.zero()
        for j in range(k):
            dist_sq = squared_distance(points[j], points[k])
            a = dist_sq ** (-s/2 - 2)
            b = -((s + 2) * (points[j][l] - points[k][l])
                          * (points[j][n] - points[k][n]))
            ans += a * b
        for j in range(k + 1, len(points)):
            dist_sq = squared_distance(points[j], points[k])
            a = dist_sq ** (-s/2 - 2)
            b = -((s + 2) * (points[j][l] - points[k][l])
                          * (points[j][n] - points[k][n]))
            ans += a * b
        return -s * ans
    elif k != m and l == n:
        dist_sq = squared_distance(points[k], points[m])
        a = dist_sq ** (-s/2 - 2)
        b = dist_sq - ((s + 2) * (points[k][l] - points[m][l])
                               * (points[k][n] - points[m][n]))
        return s * a * b
    elif k != m and l != n:
        dist_sq = squared_distance(points[k], points[m])
        a = dist_sq ** (-s/2 - 2)
        b = -((s + 2) * (points[k][l] - points[m][l])
                      * (points[k][n] - points[m][n]))
        return s * a * b


def riesz_hessian_entry_1(points, k, l, m, n):
    if k == m and l == n:
        ans = gmpy2.zero()
        for j in range(k):
            dist_sq = squared_distance(points[j], points[k])
            a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
            b = dist_sq - (3 * (points[j][l] - points[k][l])
                             * (points[j][n] - points[k][n]))
            ans -= b / a
        for j in range(k + 1, len(points)):
            dist_sq = squared_distance(points[j], points[k])
            a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
            b = dist_sq - (3 * (points[j][l] - points[k][l])
                             * (points[j][n] - points[k][n]))
            ans -= b / a
        return ans
    elif k == m and l != n:
        ans = gmpy2.zero()
        for j in range(k):
            dist_sq = squared_distance(points[j], points[k])
            a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
            b = (3 * (points[j][l] - points[k][l])
                   * (points[j][n] - points[k][n]))
            ans += b / a
        for j in range(k + 1, len(points)):
            dist_sq = squared_distance(points[j], points[k])
            a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
            b = (3 * (points[j][l] - points[k][l])
                   * (points[j][n] - points[k][n]))
            ans += b / a
        return ans
    elif k != m and l == n:
        dist_sq = squared_distance(points[k], points[m])
        a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
        b = dist_sq - (3 * (points[k][l] - points[m][l])
                         * (points[k][n] - points[m][n]))
        return b / a
    elif k != m and l != n:
        dist_sq = squared_distance(points[k], points[m])
        a = gmpy2.square(dist_sq) * gmpy2.sqrt(dist_sq)
        b = -(3 * (points[k][l] - points[m][l])
                * (points[k][n] - points[m][n]))
        return b / a


################################################################################


def squared_norm(v):
    norm_sq = gmpy2.square(v[0])
    for i in range(1, len(v)):
        norm_sq += gmpy2.square(v[i])
    return norm_sq


def norm(v):
    norm_sq = gmpy2.square(v[0])
    for i in range(1, len(v)):
        norm_sq += gmpy2.square(v[i])
    return gmpy2.sqrt(norm_sq)


def ldlt_decomposition(mat):
    n = len(mat)
    diag = []
    low = [[] for _ in range(n)]
    for i in range(n):
        temp = gmpy2.zero()
        for j in range(i):
            temp += gmpy2.square(low[i][j]) * diag[j]
        diag.append(mat[i][i] - temp)
        for j in range(i + 1, n):
            temp = gmpy2.zero()
            for k in range(i):
                temp += low[j][k] * low[i][k] * diag[k]
            low[j].append((mat[j][i] - temp) / diag[i])
    return low, diag


def ult_solve_in_place(low, vec):
    for i in range(len(vec)):
        for j in range(i):
            vec[i] -= low[i][j] * vec[j]
    return None


def uut_solve_in_place(up, vec):
    n = len(vec)
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            vec[i] -= up[i][j] * vec[j]
    return None


def ultt_solve_in_place(low, vec):
    n = len(vec)
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            vec[i] -= low[j][i] * vec[j]
    return None


################################################################################


def double_add_in_place(v, a, w):
    m = len(v)
    n = len(v[0])
    for i in range(m):
        for j in range(n):
            v[i][j] += a * w[i][j]
    return None


def lbfgs_correction_in_place(grad, delta_x, delta_g, rho):
    hist_len = len(delta_x)
    hist_indices = range(hist_len)
    alpha = [None for _ in hist_indices]
    for i in reversed(hist_indices):
        alpha[i] = rho[i] * double_dot_product(delta_x[i], grad)
        double_subtract_in_place(grad, alpha[i], delta_g[i])
    for i in hist_indices:
        beta = rho[i] * double_dot_product(delta_g[i], grad)
        double_add_in_place(grad, alpha[hist_len - i - 1] - beta, delta_x[i])
    return None


def constrained_step_in_place(points, step_size, step_direction):
    point_indices = range(len(points))
    dim_indices = range(len(points[0]))
    for i in point_indices:
        for k in dim_indices:
            points[i][k] += step_size * step_direction[i][k]
    normalize_each_in_place(points)
    return None
