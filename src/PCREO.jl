module PCREO

using DZOptimization: half, normalize_columns!, step!
using DZOptimization.ExampleFunctions:
    riesz_energy, riesz_gradient!, riesz_hessian!,
    constrain_riesz_gradient_sphere!, constrain_riesz_hessian_sphere!
using MultiFloats: Float64x2, Float64x3
using StaticArrays: SArray, SVector, cross, norm

export constrain_sphere!, spherical_riesz_gradient!,
    spherical_riesz_gradient, spherical_riesz_hessian,
    run!, # convex_hull_facets,
    adjacency_structure,
    packing_radius, covering_radius, symmetrize!, parallel_facet_distance,
    PCREORecord,
    defect_graph, defect_classes, unicode_defect_string, html_defect_string,
    symmetrized_riesz_energy, symmetrized_riesz_gradient!,
    symmetrized_riesz_functors


########################################################### FILE NAMES AND PATHS


# const PCREO_DIRNAME_REGEX = Regex(
#     "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
#     "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\$")

# const PCREO_FILENAME_REGEX = Regex(
#     "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
#     "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\\.csv\$")

# @static if Sys.iswindows()
#     const PCREO_OUTPUT_DIRECTORY = "D:\\Data\\PCREOOutput"
# else
#     const PCREO_OUTPUT_DIRECTORY = "/home/dkzhang/pcreo-output"
# end

# const PCREO_DATABASE_DIRECTORY = "D:\\Data\\PCREODatabase"

# @static if Sys.iswindows()
#     const QCONVEX_PATH = "C:\\Programs\\qhull-2020.2\\bin\\qconvex.exe"
# else
#     const QCONVEX_PATH = "qconvex"
# end

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


################################################################### OPTIMIZATION


function run!(opt; quiet::Bool=true, framerate=10)
    if quiet
        while !opt.has_converged[]
            step!(opt)
        end
        return opt
    else
        last_print_time = time_ns()
        frame_time = round(Int, 1_000_000_000 / framerate)
        while !opt.has_converged[]
            step!(opt)
            if time_ns() - last_print_time >= frame_time
                println(opt.iteration_count[], '\t',
                        opt.current_objective_value[])
                last_print_time += frame_time
            end
        end
        println(opt.iteration_count[], '\t',
                opt.current_objective_value[])
        return opt
    end
end


###################################################################### ADJACENCY


# function convex_hull_facets(points::Vector{SVector{N,T}}) where {T,N}
#     @assert length(points) >= 4
#     num_retries = 0
#     while true
#         buffer = IOBuffer()
#         process = open(`$QCONVEX_PATH i`, buffer, write=true)
#         println(process, N)
#         println(process, length(points))
#         for point in points
#             for coord in point
#                 print(process, ' ', Float64(coord))
#             end
#             println(process)
#         end
#         close(process)
#         wait(process)
#         first = true
#         num_facets = 0
#         result = Vector{Int}[]
#         seek(buffer, 0)
#         for line in eachline(buffer)
#             if first
#                 num_facets = parse(Int, line)
#                 first = false
#             else
#                 push!(result, [parse(Int, s) + 1 for s in split(line)])
#             end
#         end
#         if (num_facets > 0) && (num_facets == length(result))
#             return result
#         else
#             # Sometimes, qconvex doesn't work, and we need to try again.
#             num_retries += 1
#             if num_retries >= 10
#                 error("qconvex failed to return a result after ten tries.")
#             end
#         end
#     end
# end


function dict_incr!(d::Dict{K,Int}, k::K) where {K}
    if haskey(d, k)
        d[k] += 1
    else
        d[k] = 1
    end
    return d[k]
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


function dict_push!(d::Dict{K,Vector{T}}, k::K, v::T) where {K,T}
    if haskey(d, k)
        push!(d[k], v)
    else
        d[k] = [v]
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


########################################################### GEOMETRIC PROPERTIES


function packing_radius(points::Vector{SVector{3,T}},
                        facets::Vector{Vector{Int}}) where {T}
    adjacent_vertices, _ = adjacency_structure(facets)
    result = typemax(T)
    for (i, j) in adjacent_vertices
        result = min(result, norm(points[i] - points[j]))
    end
    return half(T) * result
end


function spherical_circumcenter(points::Vector{SVector{3,T}},
                                facet::Vector{Int}) where {T}
    n = length(facet)
    result = zero(SVector{3,T})
    for i = 1 : n-2
        for j = i+1 : n-1
            for k = j+1 : n
                @inbounds a = points[facet[i]]
                @inbounds b = points[facet[j]]
                @inbounds c = points[facet[k]]
                normal = cross(a - b, b - c)
                norm2 = normal' * normal
                denom = norm2 + norm2
                alpha = ((b - c)' * (b - c)) * ((a - b)' * (a - c))
                beta  = ((a - c)' * (a - c)) * ((b - a)' * (b - c))
                gamma = ((a - b)' * (a - b)) * ((c - a)' * (c - b))
                result += (alpha * a + beta * b + gamma * c) / denom
            end
        end
    end
    return result / norm(result)
end


middle(x::AbstractVector) = x[(length(x) + 1) >> 1]


function covering_radius(points::Vector{SVector{3,T}},
                         facets::Vector{Vector{Int}}) where {T}
    result = zero(T)
    for facet in facets
        center = spherical_circumcenter(points, facet)
        radii = [norm(center - points[i]) for i in facet]
        # TODO: Why does this occasionally fail?
        # lo, hi = extrema(radii)
        # @assert 0.0 <= hi - lo <= epsilon
        result = max(result, maximum(radii))
    end
    return result
end


############################################################ CONVERGENCE TESTING


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


function facet_normal_vector(points::Vector{SVector{3,T}}) where {T}
    n = length(points)
    result = zero(SVector{3,T})
    for i = 1 : n-2
        for j = i+1 : n-1
            for k = j+1 : n
                normal = cross(points[j] - points[i],
                               points[k] - points[i])
                normal /= norm(normal)
                positive = all(!signbit, normal' * p for p in points)
                negative = all(signbit, normal' * p for p in points)
                @assert xor(positive, negative)
                if positive
                    result += normal
                else
                    result -= normal
                end
            end
        end
    end
    return result / norm(result)
end


function parallel_facet_distance(points::Vector{SVector{3,T}},
                                 facets::Vector{Vector{Int}}) where {T}
    _, adjacent_facets = adjacency_structure(facets)
    normals = [facet_normal_vector(points[facet]) for facet in facets]
    return minimum(one(T) - normals[i]' * normals[j]
                   for (i, j) in adjacent_facets)
end


################################################################### LOADING DATA


struct PCREORecord
    dimension::Int
    num_points::Int
    symmetry_group::String
    energy::Float64x3
    first_eigenvalue::Float64
    packing_radius::Float64x2
    covering_radius::Float64x2
    num_gradient_bits::Float64
    num_hessian_bits::Float64
    facet_distance::Float64x2
    points::Matrix{Float64x2}
    facets::Vector{Vector{Int}}
    initial_points::Matrix{Float64}
end


function PCREORecord(filepath::String)
    entries = split(read(filepath, String), "\n\n")
    @assert length(entries) == 5

    geometric_properties = split(entries[1], '\n')
    @assert length(geometric_properties) == 7

    convergence_measures = split(entries[2], '\n')
    @assert length(convergence_measures) == 3

    dimension = parse(Int, geometric_properties[1])
    @assert dimension == 3
    num_points = parse(Int, geometric_properties[2])
    symmetry_group = geometric_properties[3]
    if startswith(symmetry_group, '"') && endswith(symmetry_group, '"')
        symmetry_group = symmetry_group[2:end-1]
    end
    energy = Float64x3(geometric_properties[4])
    first_eigenvalue = parse(Float64, geometric_properties[5])
    packing_radius = Float64x2(geometric_properties[6])
    covering_radius = Float64x2(geometric_properties[7])

    num_gradient_bits = parse(Float64, convergence_measures[1])
    num_hessian_bits = parse(Float64, convergence_measures[2])
    facet_distance = Float64x2(convergence_measures[3])

    points = reshape([
        Float64x2(strip(entry))
        for line in split(entries[3], '\n')
        for entry in split(line, ',')
        if !isempty(strip(line))],
        dimension, num_points)

    facets = [
        [parse(Int, strip(entry))
         for entry in split(line, ',')]
        for line in split(entries[4], '\n')
        if !isempty(strip(line))]

    initial_points = reshape([
        parse(Float64, strip(entry))
        for line in split(entries[5], '\n')
        for entry in split(line, ',')
        if !isempty(strip(line))],
        dimension, :)

    return PCREORecord(dimension, num_points, symmetry_group,
        energy, first_eigenvalue,
        packing_radius, covering_radius,
        num_gradient_bits, num_hessian_bits, facet_distance,
        points, facets, initial_points)
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


function connected_components(adjacency_lists::Dict{V,Vector{V}}) where {V}
    visited = Dict{V,Bool}()
    for (v, l) in adjacency_lists
        visited[v] = false
    end
    components = Vector{V}[]
    for (v, l) in adjacency_lists
        if !visited[v]
            visited[v] = true
            current_component = [v]
            to_visit = copy(l)
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


####################################################### SYMMETRIZED RIESZ ENERGY


# Benchmarked in Julia 1.5.3 for zero allocations or exceptions.

# @benchmark symmetrized_riesz_energy(points, group, external_points) setup=(
#     points=randn(3, 10); group=chiral_tetrahedral_group(Float64);
#     external_points=SVector{3,Float64}.(eachcol(randn(3, 5))))

# view_asm(symmetrized_riesz_energy,
#     Matrix{Float64},
#     Vector{SArray{Tuple{3,3},Float64,2,9}},
#     Vector{SVector{3,Float64}})

function symmetrized_riesz_energy(
        points::AbstractMatrix{T},
        group::Vector{SArray{Tuple{N,N},T,2,M}},
        external_points::Vector{SArray{Tuple{N},T,1,N}}) where {T,N,M}
    dim, num_points = size(points)
    group_size = length(group)
    num_external_points = length(external_points)
    energy = zero(T)
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for j = 2 : group_size
            @inbounds g = group[j]
            energy += 0.5 * inv(norm(g*p - p))
        end
    end
    for i = 2 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for g in group
            gp = g * p
            for j = 1 : i-1
                @inbounds q = SVector{N,T}(view(points, 1:N, j))
                energy += inv(norm(gp - q))
            end
        end
    end
    energy *= group_size
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for g in group
            gp = g * p
            for j = 1 : num_external_points
                @inbounds q = external_points[j]
                energy += inv(norm(gp - q))
            end
        end
    end
    return energy
end


# Benchmarked in Julia 1.5.3 for zero allocations or exceptions.

# @benchmark symmetrized_riesz_gradient!(
#     grad, points, group, external_points) setup=(
#     points=randn(3, 10); grad=similar(points);
#     group=chiral_tetrahedral_group(Float64);
#     external_points=SVector{3,Float64}.(eachcol(randn(3, 5))))

# view_asm(symmetrized_riesz_gradient!,
#     Matrix{Float64}, Matrix{Float64},
#     Vector{SArray{Tuple{3,3},Float64,2,9}},
#     Vector{SVector{3,Float64}})

function symmetrized_riesz_gradient!(
        grad::AbstractMatrix{T},
        points::AbstractMatrix{T},
        group::Vector{SArray{Tuple{N,N},T,2,M}},
        external_points::Vector{SArray{Tuple{N},T,1,N}}) where {T,N,M}
    dim, num_points = size(points)
    group_size = length(group)
    num_external_points = length(external_points)
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        force = zero(SVector{N,T})
        for j = 2 : group_size
            @inbounds r = group[j] * p - p
            force += r / norm(r)^3
        end
        for j = 1 : num_points
            if i != j
                @inbounds q = SVector{N,T}(view(points, 1:N, j))
                for g in group
                    r = g * q - p
                    force += r / norm(r)^3
                end
            end
        end
        force *= group_size
        for j = 1 : num_external_points
            @inbounds q = external_points[j]
            for g in group
                r = g * q - p
                force += r / norm(r)^3
            end
        end
        @simd ivdep for j = 1 : N
            @inbounds grad[j,i] = force[j]
        end
    end
    return grad
end


struct SymmetrizedRieszEnergyFunctor{T}
    group::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points::Vector{SArray{Tuple{3},T,1,3}}
    external_energy::T
end


struct SymmetrizedRieszGradientFunctor{T}
    group::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points::Vector{SArray{Tuple{3},T,1,3}}
end


function (sref::SymmetrizedRieszEnergyFunctor{T})(
          points::AbstractMatrix{T}) where {T}
    return sref.external_energy + symmetrized_riesz_energy(
        points, sref.group, sref.external_points)
end


function (srgf::SymmetrizedRieszGradientFunctor{T})(
          grad::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T}
    symmetrized_riesz_gradient!(grad, points, srgf.group, srgf.external_points)
    constrain_riesz_gradient_sphere!(grad, points)
    return grad
end


function symmetrized_riesz_functors(
        ::Type{T}, group_function::Function,
        orbit_functions::Vector{Function}) where {T}
    group = group_function(T)::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points = vcat([orbit_function(T)::Vector{SArray{Tuple{3},T,1,3}}
                            for orbit_function in orbit_functions]...)
    external_points_matrix = Matrix{T}(undef, 3, length(external_points))
    for (i, point) in enumerate(external_points)
        @simd ivdep for j = 1 : 3
            @inbounds external_points_matrix[j,i] = point[j]
        end
    end
    external_energy = riesz_energy(external_points_matrix)
    return (SymmetrizedRieszEnergyFunctor{T}(group, external_points,
                                             external_energy),
            SymmetrizedRieszGradientFunctor{T}(group, external_points))
end


################################################################################

end # module PCREO
