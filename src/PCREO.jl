module PCREO

using GenericLinearAlgebra
using LinearAlgebra: cross, det, eigvals!, svd
using NearestNeighbors: KDTree, knn

using DZOptimization: dot, half, norm, normalize!, normalize_columns!,
    unsafe_sqrt
using DZOptimization.ExampleFunctions:
    riesz_energy, riesz_gradient!, riesz_hessian!,
    constrain_riesz_gradient_sphere!, constrain_riesz_hessian_sphere!

export PCREO_DIRNAME_REGEX, PCREO_FILENAME_REGEX,
    PCREO_OUTPUT_DIRECTORY, PCREO_DATABASE_DIRECTORY,
    PCREO_GRAPH_DIRECTORY, PCREO_FACET_ERROR_DIRECTORY,
    riesz_energy, constrain_sphere!,
    spherical_riesz_gradient!, spherical_riesz_gradient,
    spherical_riesz_hessian, spherical_riesz_gradient_norm,
    spherical_riesz_hessian_spectral_gap,
    convex_hull_facets, facet_normal_vector, parallel_facet_distance,
    PCREORecord,
    distances, labeled_distances, bucket_by_first, middle,
    positive_transformation_matrix, negative_transformation_matrix,
    candidate_isometries, matching_distance,
    dict_push!, dict_incr!,
    adjacency_structure, incidence_degrees, connected_components,
    defect_graph, defect_classes, unicode_defect_string, html_defect_string,
    automorphism_group, point_group


########################################################### FILE NAMES AND PATHS


const PCREO_DIRNAME_REGEX = Regex(
    "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
    "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\$")

const PCREO_FILENAME_REGEX = Regex(
    "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
    "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\\.csv\$")

const PCREO_OUTPUT_DIRECTORY = "D:\\Data\\PCREO"

const PCREO_DATABASE_DIRECTORY = "D:\\Data\\PCREODatabase"

const PCREO_GRAPH_DIRECTORY = "D:\\Data\\PCREOGraphs"

const PCREO_FACET_ERROR_DIRECTORY = "D:\\Data\\PCREOFacetErrors"

const QCONVEX_PATH = "C:\\Programs\\qhull-2020.2\\bin\\qconvex.exe"


######################################################## RIESZ ENERGY ON SPHERES


function constrain_sphere!(points)
    normalize_columns!(points)
    return true
end


function spherical_riesz_gradient!(grad, points)
    riesz_gradient!(grad, points)
    constrain_riesz_gradient_sphere!(grad, points)
    return grad
end


spherical_riesz_gradient(points) =
    spherical_riesz_gradient!(similar(points), points)


function spherical_riesz_hessian(points::Matrix{T}) where {T}
    unconstrained_grad = riesz_gradient!(similar(points), points)
    hess = Array{T,4}(undef, size(points)..., size(points)...)
    riesz_hessian!(hess, points)
    constrain_riesz_hessian_sphere!(hess, points, unconstrained_grad)
    return reshape(hess, length(points), length(points))
end


############################################################ CONVERGENCE TESTING


spherical_riesz_gradient_norm(points) =
    maximum(abs.(spherical_riesz_gradient(points)))


function symmetrize!(mat::Matrix{T}) where {T}
    m, n = size(mat)
    @assert m == n
    @inbounds for i = 1 : n-1
        @simd ivdep for j = i+1 : n
            sym = half(T) * (mat[i, j] + mat[j, i])
            mat[i, j] = mat[j, i] = sym
        end
    end
    return mat
end


function spherical_riesz_hessian_spectral_gap(points)
    dim, num_points = size(points)
    hess = symmetrize!(spherical_riesz_hessian(points))
    vals = eigvals!(hess)
    num_expected_zeros = div(dim * (dim - 1), 2) + num_points
    expected_zero_vals = vals[1:num_expected_zeros]
    expected_nonzero_vals = vals[num_expected_zeros+1:end]
    @assert all(!signbit, expected_nonzero_vals)
    return (maximum(abs.(expected_zero_vals)) /
            minimum(expected_nonzero_vals))
end


function convex_hull_facets(points::Matrix{Float64})
    dim, num_points = size(points)
    buffer = IOBuffer()
    process = open(`$QCONVEX_PATH i`, buffer, write=true)
    println(process, dim)
    println(process, num_points)
    for j = 1 : num_points
        for i = 1 : dim
            print(process, ' ', points[i,j])
        end
        println(process)
    end
    close(process)
    while process_running(process)
        sleep(0.001)
    end
    first = true
    num_facets = 0
    result = Vector{Int}[]
    seek(buffer, 0)
    for line in eachline(buffer)
        if first
            num_facets = parse(Int, line)
            first = false
        else
            push!(result, [parse(Int, s) + 1 for s in split(line)])
        end
    end
    @assert num_facets == length(result)
    return result
end


function facet_normal_vector(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    result = zeros(T, dimension)
    for i = 1 : num_points - 2
        for j = i+1 : num_points - 1
            for k = j+1 : num_points
                normal = cross(points[:,j] - points[:,i],
                               points[:,k] - points[:,i])
                positive = all(signbit, dot(normal, points, dimension, t)
                                        for t = 1 : num_points)
                negative = all(!signbit, dot(normal, points, dimension, t)
                                         for t = 1 : num_points)
                @assert xor(positive, negative)
                if positive
                    result += normal
                else
                    result -= normal
                end
            end
        end
    end
    return normalize!(result)
end


function parallel_facet_distance(points::Matrix{T},
                                 facets::Vector{Vector{Int}}) where {T}
    _, adjacent_facets = adjacency_structure(facets)
    normals = [facet_normal_vector(points[:,facet]) for facet in facets]
    return minimum(one(T) - normals[i]' * normals[j]
                   for (i, j) in adjacent_facets)
end


################################################################### LOADING DATA


struct PCREORecord
    dimension::Int
    num_points::Int
    energy::Float64
    points::Matrix{Float64}
    facets::Vector{Vector{Int}}
    initial::Matrix{Float64}
end


function PCREORecord(path::AbstractString)
    if occursin(PCREO_DIRNAME_REGEX, path)
        path = joinpath(PCREO_DATABASE_DIRECTORY, path, path * ".csv")
    end
    filename = basename(path)
    m = match(PCREO_FILENAME_REGEX, filename)
    @assert !isnothing(m)
    dimension = parse(Int, m[1])
    num_points = parse(Int, m[2])
    uuid = m[3]
    data = split(read(path, String), "\n\n")
    @assert length(data) == 4
    header = split(data[1])
    @assert length(header) == 3
    @assert dimension == parse(Int, header[1])
    @assert num_points == parse(Int, header[2])
    energy = parse(Float64, header[3])
    points = hcat([[parse(Float64, strip(entry))
                    for entry in split(line, ',')]
                   for line in split(strip(data[2]), '\n')]...)
    @assert (dimension, num_points) == size(points)
    facets = [[parse(Int, strip(entry))
               for entry in split(line, ',')]
              for line in split(strip(data[3]), '\n')]
    initial = hcat([[parse(Float64, strip(entry))
                     for entry in split(line, ',')]
                    for line in split(strip(data[4]), '\n')]...)
    @assert (dimension, num_points) == size(initial)
    return PCREORecord(dimension, num_points, energy,
                       points, facets, initial)
end


########################################################## DISTANCES AND BUCKETS


function distances(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    num_pairs = (num_points * (num_points - 1)) >> 1
    result = Vector{T}(undef, num_pairs)
    p = 0
    for i = 1 : num_points-1
        for j = i+1 : num_points
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dimension
                @inbounds dist_sq += abs2(points[k,i] - points[k,j])
            end
            @inbounds result[p += 1] = unsafe_sqrt(dist_sq)
        end
    end
    return result
end


function labeled_distances(points::Matrix{T}) where {T}
    dimension, num_points = size(points)
    num_pairs = (num_points * (num_points - 1)) >> 1
    result = Vector{Tuple{T,Int,Int}}(undef, num_pairs)
    p = 0
    for i = 1 : num_points-1
        for j = i+1 : num_points
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dimension
                @inbounds dist_sq += abs2(points[k,i] - points[k,j])
            end
            @inbounds result[p += 1] = (unsafe_sqrt(dist_sq), i, j)
        end
    end
    return result
end


function bucket_by_first(items::Vector{T}, epsilon) where {T}
    result = Vector{T}[]
    if length(items) == 0
        return result
    end
    push!(result, [items[1]])
    for i = 2 : length(items)
        if abs(items[i][1] - result[end][end][1]) <= epsilon
            push!(result[end], items[i])
        else
            push!(result, [items[i]])
        end
    end
    return result
end


middle(x::Vector) = x[(length(x) + 1) >> 1]


####################################################################### ISOMETRY


positive_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(u1, v1)))


negative_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(v1, u1)))


function push_isometry!(isometries::Vector{Matrix{Float64}},
                        matrix::Matrix{Float64})
    @assert maximum(abs.(matrix' * matrix - one(matrix))) < 1.0e-12
    push!(isometries, matrix)
end


function candidate_isometries(a_points::Matrix{Float64},
                              b_points::Matrix{Float64})
    result = Matrix{Float64}[]
    a_buckets = bucket_by_first(sort!(labeled_distances(a_points)), 1.0e-10)
    b_buckets = bucket_by_first(sort!(labeled_distances(b_points)), 1.0e-10)
    a_spread = maximum(first.(last.(a_buckets)) - first.(first.(a_buckets)))
    b_spread = maximum(first.(last.(b_buckets)) - first.(first.(b_buckets)))
    @assert 0.0 <= a_spread < 1.0e-10
    @assert 0.0 <= b_spread < 1.0e-10
    if length.(a_buckets) == length.(b_buckets)
        # TODO: Verify that buckets are close to each other.
        bucket_index = minimum([(length(bucket), index)
                                for (index, bucket) in enumerate(a_buckets)
                                if 0.25 < middle(bucket)[1] < 1.75])[2]
        _, i, j = middle(a_buckets[bucket_index])
        for (_, k, l) in b_buckets[bucket_index]
            push_isometry!(result, positive_transformation_matrix(
                a_points[:,i], a_points[:,j], b_points[:,k], b_points[:,l]))
            push_isometry!(result, negative_transformation_matrix(
                a_points[:,i], a_points[:,j], b_points[:,k], b_points[:,l]))
            push_isometry!(result, positive_transformation_matrix(
                a_points[:,i], a_points[:,j], b_points[:,l], b_points[:,k]))
            push_isometry!(result, negative_transformation_matrix(
                a_points[:,i], a_points[:,j], b_points[:,l], b_points[:,k]))
        end
    end
    return result
end


function matching_distance(points::Matrix{T}, tree::KDTree) where {T}
    inds, dists = knn(tree, points, 1)
    @assert all(==(1), length.(inds))
    @assert all(==(1), length.(dists))
    if allunique(first.(inds))
        return maximum(first.(dists))
    else
        return typemax(T)
    end
end


###################################################################### ADJACENCY


function dict_push!(d::Dict{K,Vector{T}}, k::K, v::T) where {K,T}
    if haskey(d, k)
        push!(d[k], v)
    else
        d[k] = [v]
    end
    return d[k]
end


function dict_incr!(d::Dict{K,Int}, k::K) where {K}
    if haskey(d, k)
        d[k] += 1
    else
        d[k] = 1
    end
    return d[k]
end


function adjacency_structure(facets::Vector{Vector{Int}})
    pair_dict = Dict{Tuple{Int,Int},Vector{Int}}()
    for (k, facet) in enumerate(facets)
        n = length(facet)
        for i = 1 : n-1
            for j = i+1 : n
                dict_push!(pair_dict, minmax(facet[i], facet[j]), k)
            end
        end
    end
    adjacent_vertices = Vector{Tuple{Int,Int}}()
    adjacent_facets = Vector{Tuple{Int,Int}}()
    for (vertex_pair, incident_facets) in pair_dict
        if length(incident_facets) >= 2
            @assert length(incident_facets) == 2
            push!(adjacent_vertices, vertex_pair)
            push!(adjacent_facets, minmax(incident_facets...))
        end
    end
    return (adjacent_vertices, adjacent_facets)
end


function incidence_degrees(facets::Vector{Vector{Int}})
    degrees = Dict{Int,Int}()
    for facet in facets
        for vertex in facet
            dict_incr!(degrees, vertex)
        end
    end
    return degrees
end


function connected_components(adjacency_lists::Dict{Int,Vector{Int}})
    visited = Dict(v => false for (v, l) in adjacency_lists)
    components = Vector{Int}[]
    for (v, l) in adjacency_lists
        if !visited[v]
            visited[v] = true
            current_component = [v]
            to_visit = append!([], l)
            while !isempty(to_visit)
                w = pop!(to_visit)
                if !visited[w]
                    visited[w] = true
                    push!(current_component, w)
                    append!(to_visit, adjacency_lists[w])
                end
            end
            push!(components, current_component)
        end
    end
    @assert allunique(vcat(components...))
    return components
end


############################################################ TOPOLOGICAL DEFECTS


function defect_graph(facets::Vector{Vector{Int}})
    adjacent_vertices, _ = adjacency_structure(facets)
    adjacency_lists = Dict{Int,Vector{Int}}()
    for (v, w) in adjacent_vertices
        dict_push!(adjacency_lists, v, w)
        dict_push!(adjacency_lists, w, v)
    end
    degrees = Dict(v => length(l) for (v, l) in adjacency_lists)
    @assert degrees == incidence_degrees(facets)
    hexagonal_vertices = [v for (v, d) in degrees if d == 6]
    for k in hexagonal_vertices
        delete!(adjacency_lists, k)
        delete!(degrees, k)
    end
    for (v, l) in adjacency_lists
        deleteat!(adjacency_lists[v],
            [i for (i, w) in enumerate(l)
             if w in hexagonal_vertices])
    end
    return (adjacency_lists, degrees)
end


function defect_classes(facets::Vector{Vector{Int}})
    adjacency_lists, defect_degrees = defect_graph(facets)
    defect_components = connected_components(adjacency_lists)
    defect_counts = Dict{Vector{Tuple{Int,Tuple{Int,Int}}},Int}()
    for component in defect_components
        shape_counts = Dict{Tuple{Int,Int},Int}()
        for v in component
            dict_incr!(shape_counts,
                (length(adjacency_lists[v]), defect_degrees[v]))
        end
        shape_table = [(num, shape) for (shape, num) in shape_counts]
        sort!(shape_table; rev=true)
        dict_incr!(defect_counts, shape_table)
    end
    defect_table = [(num, defect) for (defect, num) in defect_counts]
    sort!(defect_table; rev=true)
    return defect_table
end


function shape_code(n::Int)
    if n == 3
        return 'T'
    elseif n == 4
        return 'S'
    elseif n == 5
        return 'P'
    elseif n == 6
        @assert n != 6
    elseif n == 7
        return 'H'
    elseif n == 8
        return 'O'
    elseif n == 9
        return 'N'
    elseif n == 10
        return 'D'
    elseif n == 11
        return 'U'
    else
        @assert false
    end
end


unicode_subscript_string(n::Int) = foldl(replace,
    map(Pair, Char.(0x30:0x39), Char.(0x2080:0x2089));
    init=string(n))


html_subscript_string(n::Int) = "<sub>" * string(n) * "</sub>"


unicode_defect_string(shape_table::Vector{Tuple{Int,Tuple{Int,Int}}}) =
    join([
        (num == 1 ? "" : string(num)) *
            shape_code(shape) * unicode_subscript_string(degree)
        for (num, (degree, shape)) in shape_table])


html_defect_string(shape_table::Vector{Tuple{Int,Tuple{Int,Int}}}) =
    join([
        (num == 1 ? "" : string(num)) *
            shape_code(shape) * html_subscript_string(degree)
        for (num, (degree, shape)) in shape_table])


####################################################################### SYMMETRY


function automorphism_group(points::Matrix{Float64})
    tree = KDTree(points)
    automorphisms = Matrix{Float64}[]
    for mat in candidate_isometries(points, points)
        if matching_distance(mat * points, tree) < 1.0e-12
            push!(automorphisms, mat)
        end
    end
    n = length(automorphisms)
    multiplication_table = Matrix{Int}(undef, n, n)
    for i = 1 : n
        for j = 1 : n
            dist, k = minimum([
                (norm(mat - automorphisms[i] * automorphisms[j]), index)
                for (index, mat) in enumerate(automorphisms)])
            @assert dist < 1.0e-12
            multiplication_table[i, j] = k
        end
    end
    for i = 1 : n
        @assert allunique(multiplication_table[:, i])
        @assert allunique(multiplication_table[i, :])
    end
    return (automorphisms, multiplication_table)
end


function rotation_axis(mat::Matrix{Float64})
    @assert size(mat) == (3, 3)
    @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-12
    decomp = svd(mat - one(mat))
    @assert 0.0 <= decomp.S[3] < 1.0e-12
    axis = decomp.V[:,3]
    @assert maximum(abs.(mat * axis - axis)) < 1.0e-12
    return axis
end


function order(x::Int, multiplication_table::Matrix{Int}, e::Int)
    m, n = size(multiplication_table)
    @assert m == n
    @assert 1 <= e <= n
    @assert 1 <= x <= n
    @assert issorted(multiplication_table[e,:])
    @assert issorted(multiplication_table[:,e])
    acc = x
    result = 1
    while acc != e
        acc = multiplication_table[acc,x]
        result += 1
    end
    return result
end


function is_pure_reflection(mat::Matrix{Float64})
    @assert size(mat) == (3, 3)
    return maximum(abs.(svd(mat + one(mat)).S - [2.0, 2.0, 0.0])) < 1.0e-12
end


function count_central_elements(multiplication_table::Matrix{Int})
    m, n = size(multiplication_table)
    @assert m == n
    return count(all(
        @inbounds multiplication_table[i,j] == multiplication_table[j,i]
        for j = 1 : n) for i = 1 : n)
end


function point_group(points::Matrix{Float64})

    matrices, multiplication_table = automorphism_group(points)

    n = length(matrices)
    @assert (n, n) == size(multiplication_table)

    if n == 1
        return "C_1" # Only one option: the trivial group.
    end

    identity_element = minimum([
        (maximum(abs.(mat - one(mat))), idx)
        for (idx, mat) in enumerate(matrices)])[2]
    @assert issorted(multiplication_table[identity_element,:])
    @assert issorted(multiplication_table[:,identity_element])

    if n == 2
        # Z2 can act on R3 via linear isometries in one of three ways:
        # by a 180-degree rotation, by a reflection, or by an inversion.
        non_identity_element = matrices[3 - identity_element]
        if det(non_identity_element) > 0
            return "C_2"
        else
            sigma = svd(non_identity_element + one(non_identity_element)).S
            is_reflection = (maximum(abs.(sigma - [2.0, 2.0, 0.0])) < 1.0e-12)
            is_inversion = (maximum(abs.(sigma)) < 1.0e-12)
            @assert xor(is_reflection, is_inversion)
            if is_reflection
                return "C_s"
            else
                return "C_i"
            end
        end
    end

    positive_matrices = [
        mat for (i, mat) in enumerate(matrices)
        if (i != identity_element) && (det(mat) > 0.0)]

    is_chiral = (length(positive_matrices) == n - 1)
    if !is_chiral
        @assert iseven(n)
        @assert (n >> 1) == length(positive_matrices) + 1
    end

    order_table = [order(x, multiplication_table, identity_element)
                   for x = 1 : n]
    max_order = maximum(order_table)
    @assert max_order > 1
    max_order_elements = [
        matrices[i] for (i, k) in enumerate(order_table)
        if k == max_order]
    is_cyclic = (max_order == n)

    num_central_elements = count_central_elements(multiplication_table)
    is_abelian = (num_central_elements == n)

    if is_cyclic
        @assert is_abelian
        if is_chiral
            # A cyclic group generated by a positive matrix must be C_n.
            return "C_$n"
        else
            # A cyclic group generated by a negative matrix can either be
            # S_n, in which case the group contains no pure reflection,
            # or C_nh for n odd, in which case there is a pure reflection.
            @assert iseven(n)
            @assert all(det(mat) < 0 for mat in max_order_elements)
            num_pure_refls = sum(is_pure_reflection.(matrices))
            if num_pure_refls == 0
                return "S_$n"
            else
                @assert num_pure_refls == 1
                @assert isodd(n >> 1)
                return "C_$(n >> 1)h"
            end
        end
    end

    if n == 4
        # A non-cyclic group of order 4 must be the Klein four-group.
        @assert !is_cyclic
        @assert is_abelian
        # The Klein four-group can act on R3 via linear isometries in
        # three non-equivalent ways. If all elements of the group are
        # orientation-preserving, then it acts by 180-degree rotations
        # around three mutually perpendicular axes.
        if is_chiral
            axes = Vector{Float64}[]
            for i = 1 : 4
                if i != identity_element
                    push!(axes, rotation_axis(matrices[i]))
                end
            end
            xy = abs(axes[1]' * axes[2])
            yz = abs(axes[2]' * axes[3])
            xz = abs(axes[1]' * axes[3])
            @assert max(xy, yz, xz) < 1.0e-12
            return "D_2"
        else
            # Otherwise, the four elements are:
            #     (1) the identity element,
            #     (2) a 180-degree rotation,
            #     (3) a pure reflection, and
            #     (4) a rotoinversion.
            # There are two cases: either the axis of rotation (2) is
            # perpendicular to the plane of reflection (3), or it is
            # contained in the plane.
            @assert length(positive_matrices) == 1
            axis = rotation_axis(positive_matrices[1])
            num_pos = 0
            num_neg = 0
            for mat in matrices
                if norm(mat * axis - axis) < 1.0e-12
                    num_pos += 1
                elseif norm(mat * axis + axis) < 1.0e12
                    num_neg += 1
                else
                    @assert false
                end
            end
            if (num_pos == 4) && (num_neg == 0)
                return "C_2v"
            elseif (num_pos == 2) && (num_neg == 2)
                return "C_2h"
            else
                @assert false
            end
        end
    end

    # The only non-cylic abelian point groups of order > 4 are C_nh
    # for n even, and D_2h, which "accidentally" happens to be abelian.
    if is_abelian
        @assert iseven(n)
        @assert iseven(n >> 1)
        @assert !is_cyclic
        @assert !is_chiral
        if max_order == (n >> 1)
            return "C_$(n >> 1)h"
        else
            @assert n == 8
            @assert max_order == 2
            return "D_2h"
        end
    end

    # To distinguish between the remaining point groups, we need to know
    # whether the group is axial (i.e., all group elements preserve a
    # common axis) or polyhedral. We determine this by computing the axes
    # of all maximal-order rotations.
    axes = [rotation_axis(mat)
            for mat in max_order_elements
            if det(mat) > 0.0]

    # If there are no maximal-order rotations, but only maximal-order
    # rotoinversions, then the group is either prismatic of odd order
    # or antiprismatic.
    if length(axes) == 0
        @assert iseven(n)
        @assert iseven(n >> 1)
        axes = [rotation_axis(-mat)
                for mat in max_order_elements]
        is_axial = all(abs(abs(a' * b) - 1.0) < 1.0e-12
                       for a in axes for b in axes)
        if is_axial
            num_pure_normal_refls = count(
                is_pure_reflection(mat) &&
                maximum(abs.(mat * axes[1] + axes[1])) < 1.0e-12
                for mat in matrices)
            if num_pure_normal_refls == 0
                return "D_$(n >> 2)d"
            else
                @assert num_pure_normal_refls == 1
                @assert isodd(n >> 2)
                return "D_$(n >> 2)h"
            end
        else
            if n == 120
                return "I_h"
            elseif n == 48
                return "O_h"
            else
                @assert n == 24
                if num_central_elements == 1
                    return "T_d"
                else
                    @assert num_central_elements == 2
                    return "T_h"
                end
            end
        end
    end

    is_axial = all(abs(abs(a' * b) - 1.0) < 1.0e-12
                   for a in axes for b in axes)
    if is_axial
        if n == 2 * max_order
            # There is one generator of non-rotations, which is either
            # a flip (yielding the dihedral group) or a reflection about
            # a plane containing the axis (yielding the pyramidal group).
            if is_chiral
                return "D_$(n >> 1)"
            else
                @assert all(det(mat) > 0.0 for mat in max_order_elements)
                @assert all(maximum(abs.(mat * axes[1] - axes[1])) < 1.0e-12
                            for mat in matrices)
                return "C_$(n >> 1)v"
            end
        else
            # The only remaining axial point group
            # is a prismatic group of even order.
            @assert n == 4 * max_order
            @assert iseven(n >> 2)
            num_pure_normal_refls = count(
                is_pure_reflection(mat) &&
                maximum(abs.(mat * axes[1] + axes[1])) < 1.0e-12
                for mat in matrices)
            @assert num_pure_normal_refls == 1
            return "D_$(n >> 2)h"
        end
    else
        @assert is_chiral
        if n == 12
            return "T"
        elseif n == 24
            return "O"
        else
            @assert n == 60
            return "I"
        end
    end
end


equatorial_points(height, offset, n) = [
    normalize!(append!([height], reim(cis(2pi * (x + offset) / n))))
    for x = 1 : n]


Cn_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(-0.2, 0.05, n)...,
    equatorial_points(0.1, 0.15, n)...)


Cnh_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(+0.2, 0.1, n)...,
    equatorial_points(-0.2, 0.1, n)...)


Cnv_point_configuration(n::Int) = hcat(
    equatorial_points(0.2, 0.0, n)...,
    equatorial_points(0.0, +0.1, n)...,
    equatorial_points(0.0, -0.1, n)...)


Sn_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(0.0, 0.2, n)...,
    [normalize!(append!([(-1)^x * 0.1],
        reim(cis(2pi * (x + 0.2) / n))))
     for x = 1 : n]...)


Dn_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(0.0, +0.1, n)...,
    equatorial_points(0.0, -0.1, n)...,
    equatorial_points(+0.2, +0.1, n)...,
    equatorial_points(-0.2, -0.1, n)...)


Dnh_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(0.0, +0.1, n)...,
    equatorial_points(0.0, -0.1, n)...,
    equatorial_points(+0.2, 0.0, n)...,
    equatorial_points(-0.2, 0.0, n)...)


Dnd_point_configuration(n::Int) = hcat(
    equatorial_points(0.0, 0.0, n)...,
    equatorial_points(0.0, +0.1, n)...,
    equatorial_points(0.0, -0.1, n)...,
    equatorial_points(+0.2, 0.0, n)...,
    equatorial_points(0.0, 0.5, n)...,
    equatorial_points(0.0, 0.4, n)...,
    equatorial_points(0.0, 0.6, n)...,
    equatorial_points(-0.2, 0.5, n)...)


Td_point_configuration() = [
    +sqrt(2/3) -sqrt(2/3)  0.0        0.0;
     0.0        0.0       +sqrt(2/3) -sqrt(2/3);
    -sqrt(1/3) -sqrt(1/3) +sqrt(1/3) +sqrt(1/3)]


Oh_point_configuration() = [
    +1.0 -1.0  0.0  0.0  0.0  0.0;
     0.0  0.0 +1.0 -1.0  0.0  0.0;
     0.0  0.0  0.0  0.0 +1.0 -1.0]


function test_point_group()
    point_group.(Cn_point_configuration.(1:10)) |> println
    point_group.(Cnh_point_configuration.(1:10)) |> println
    point_group.(Cnv_point_configuration.(1:10)) |> println
    point_group.(Sn_point_configuration.(2:2:20)) |> println
    point_group.(Dn_point_configuration.(1:10)) |> println
    point_group.(Dnh_point_configuration.(1:10)) |> println
    point_group.(Dnd_point_configuration.(1:10)) |> println
end


################################################################################

end # module PCREO
