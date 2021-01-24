module PCREO

using LinearAlgebra: cross
using NearestNeighbors: KDTree, knn

using DZOptimization: half, norm, normalize_columns!, unsafe_sqrt
using DZOptimization.ExampleFunctions:
    riesz_energy, riesz_gradient!, riesz_hessian!,
    constrain_riesz_gradient_sphere!, constrain_riesz_hessian_sphere!

export PCREO_DIRNAME_REGEX, PCREO_FILENAME_REGEX,
    PCREO_OUTPUT_DIRECTORY, PCREO_DATABASE_DIRECTORY,
    PCREO_GRAPH_DIRECTORY, PCREO_FACET_ERROR_DIRECTORY,
    riesz_energy, constrain_sphere!,
    spherical_riesz_gradient!, spherical_riesz_gradient,
    spherical_riesz_hessian, PCREORecord,
    distances, labeled_distances, bucket_by_first, middle,
    positive_transformation_matrix, negative_transformation_matrix,
    candidate_isometries, matching_distance,
    dict_push!, dict_incr!,
    adjacency_structure, incidence_degrees, connected_components,
    defect_graph, defect_classes, automorphism_group


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


################################################################################

end # module PCREO
