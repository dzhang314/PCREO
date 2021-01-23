module PCREO

using LinearAlgebra: cross
using NearestNeighbors: KDTree, knn

using DZOptimization: half, normalize_columns!, unsafe_sqrt
using DZOptimization.ExampleFunctions:
    riesz_energy, riesz_gradient!, riesz_hessian!,
    constrain_riesz_gradient_sphere!, constrain_riesz_hessian_sphere!

export PCREO_DIRNAME_REGEX, PCREO_FILENAME_REGEX,
    PCREO_OUTPUT_DIRECTORY, PCREO_DATABASE_DIRECTORY,
    riesz_energy, constrain_sphere!,
    spherical_riesz_gradient!, spherical_riesz_gradient,
    spherical_riesz_hessian, PCREORecord,
    distances, labeled_distances, bucket_by_first, middle,
    positive_transformation_matrix, negative_transformation_matrix,
    matching_distance


########################################################### FILE NAMES AND PATHS


const PCREO_DIRNAME_REGEX = Regex(
    "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
    "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\$")

const PCREO_FILENAME_REGEX = Regex(
    "^PCREO-([0-9]{2})-([0-9]{4})-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
    "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\\.csv\$")

const PCREO_OUTPUT_DIRECTORY = "D:\\Data\\PCREO"

const PCREO_DATABASE_DIRECTORY = "D:\\Data\\PCREODatabase"


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


############################################################### ISOMETRY TESTING


positive_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(u1, v1)))


negative_transformation_matrix(u1, v1, u2, v2) =
    hcat(u2, v2, cross(u2, v2)) * inv(hcat(u1, v1, cross(v1, u1)))


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


################################################################################

end # module PCREO
