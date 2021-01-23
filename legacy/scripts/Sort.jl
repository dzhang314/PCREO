push!(LOAD_PATH, "C:\\Users\\Zhang\\Documents\\GitHub\\pcreo\\src")

using LinearAlgebra: I, ×
using Random: shuffle!
using UUIDs

using MultiFloats
using NearestNeighbors: KDTree, knn
using ProgressMeter: @showprogress

using DZLinearAlgebra
using RieszEnergy

use_clean_multifloat_arithmetic()
setprecision(64 * 3)

function split_into_runs(items::AbstractArray{T,N};
                         by=identity, threshold=zero(T)) where {T,N}
    runs = Vector{T}[]
    @inbounds if !isempty(items)
        current_run = T[items[1]]
        current_score = by(items[1])
        for i = 2 : length(items)
            item = items[i]
            next_score = by(item)
            if abs(next_score - current_score) <= threshold
                push!(current_run, item)
            else
                push!(runs, current_run)
                current_run = T[item]
            end
            current_score = next_score
        end
        push!(runs, current_run)
    end
    return runs
end

################################################################################

const NUM_BITS = 90
const EPSILON = inv(Float64x3(2)^NUM_BITS)

################################################################################

function candidate_isometries(p1, p2, q1, q2)
    return [
        hcat(q1, q2, q1 × q2) / hcat(p1, p2, p1 × p2),
        hcat(q1, q2, q1 × q2) / hcat(p1, p2, p2 × p1),
        hcat(q1, q2, q1 × q2) / hcat(p2, p1, p1 × p2),
        hcat(q1, q2, q1 × q2) / hcat(p2, p1, p2 × p1),
    ]
end

function riesz_energy_distances!(
        result::AbstractVector{Tuple{T,Int,Int}},
        points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    energy = zero(T)
    k = 0
    @inbounds for i = 1 : num_points
        for j = i+1 : num_points
            norm_sq = zero(T)
            @simd ivdep for d = 1 : dim
                temp = points[d,i] - points[d,j]
                norm_sq += temp * temp
            end
            k += 1
            dist = renormalize(unsafe_sqrt(norm_sq))
            energy += inv(dist)
            result[k] = (dist, i, j)
        end
    end
    return energy
end

function isocmp(x::Tuple{T,Int,Int}, y::Tuple{T,Int,Int}) where {T<:Real}
    @inbounds return MultiFloats._lt(x[1], y[1])
end

function isisomorphic(p::Matrix{T}, q::Matrix{T};
                      verbose::Bool=false) where {T <: Real}
    if size(p) != size(q)
        if verbose
            println("DIFFERENT SIZES")
        end
        return false
    end
    dim, num_points = size(p)
    num_pairs = div(num_points * (num_points - 1), 2)
    threshold = EPSILON * dim * num_points
    p_dist = Vector{Tuple{T,Int,Int}}(undef, num_pairs)
    q_dist = Vector{Tuple{T,Int,Int}}(undef, num_pairs)
    Ep = riesz_energy_distances!(p_dist, p)
    Eq = riesz_energy_distances!(q_dist, q)
    if !(abs((Ep - Eq) / (Ep + Eq)) <= threshold)
        if verbose
            println("DIFFERENT ENERGIES: ", abs((Ep - Eq) / (Ep + Eq)))
        end
        return false
    end
    sort!(p_dist; lt=isocmp)
    sort!(q_dist; lt=isocmp)
    @inbounds for i = 1 : num_pairs
        if !(abs(p_dist[i][1] - q_dist[i][1]) <= threshold)
            if verbose
                println("DIFFERENT DISTANCES: ",
                        abs(p_dist[i][1] - q_dist[i][1]))
            end
            return false
        end
    end
    p_runs = split_into_runs(p_dist; by=x->x[1], threshold=threshold)
    q_runs = split_into_runs(q_dist; by=x->x[1], threshold=threshold)
    filter!(run -> !(abs(run[end][1] - 2) <= threshold), p_runs)
    filter!(run -> !(abs(run[end][1] - 2) <= threshold), q_runs)
    if length.(p_runs) != length.(q_runs)
        if verbose
            println("DIFFERENT BARCODES")
        end
        return false
    end
    _, run_idx = findmin(length.(p_runs))
    _, pi, pj = p_runs[run_idx][1]
    for (_, qi, qj) in q_runs[run_idx]
        for R in candidate_isometries(p[:,pi], p[:,pj], q[:,qi], q[:,qj])
            @assert maximum(abs.(R' * R - I)) < threshold
            p_tree = KDTree(R * p)
            idxs, dists = knn(p_tree, q, 1)
            idxs = vcat(idxs...)
            dists = vcat(dists...)
            @assert length(idxs) == num_points
            @assert length(dists) == num_points
            if issetequal(idxs, 1:num_points) && (maximum(dists) < threshold)
                return true
            end
        end
    end
    if verbose
        println("DID NOT FIND ISOMETRY")
    end
    return false
end

################################################################################

const SOURCE_DIR = "C:/Users/Zhang/PCREO/PCREO"
const DATABASE_DIR = "C:/PCREO-DB"
const DATABASE_CACHE = Dict{Int,Vector{Tuple{Float64x3,String}}}()

function build_database_cache(num_points::Int)
    @assert !haskey(DATABASE_CACHE, num_points)
    println("Building database cache for size: ", num_points)
    identifier = "PCREO-" * lpad(num_points, 8, '0')
    cache_entry = Tuple{Float64x3,String}[]
    for foldername in readdir(DATABASE_DIR)
        if occursin(identifier, foldername)
            folderpath = joinpath(DATABASE_DIR, foldername)
            @assert isdir(folderpath)
            exemplarpath = joinpath(DATABASE_DIR,
                foldername, first(readdir(folderpath)))
            @assert isfile(exemplarpath)
            exemplar = Float64x3.(BigFloat.(readlines(exemplarpath)))
            @assert length(exemplar) % 3 == 0
            num_points = div(length(exemplar), 3)
            exemplar = reshape(exemplar, (3, num_points))
            @assert norm(riesz_gradient(exemplar)) <= EPSILON * length(exemplar)
            energy = riesz_energy(exemplar)
            push!(cache_entry, (energy, folderpath))
        end
    end
    DATABASE_CACHE[num_points] = cache_entry
end

function find_in_database(points::Matrix{Float64x3})
    dim, num_points = size(points)
    threshold = EPSILON * dim * num_points
    Ep = riesz_energy(points)
    if !haskey(DATABASE_CACHE, num_points)
        build_database_cache(num_points)
    end
    @assert haskey(DATABASE_CACHE, num_points)
    for (Eq, folderpath) in DATABASE_CACHE[num_points]
        if abs((Ep - Eq) / (Ep + Eq)) <= threshold
            @assert isdir(folderpath)
            exemplarpath = joinpath(folderpath, first(readdir(folderpath)))
            @assert isfile(exemplarpath)
            exemplar = Float64x3.(BigFloat.(readlines(exemplarpath)))
            @assert length(exemplar) % 3 == 0
            exemplar = reshape(exemplar, (3, div(length(exemplar), 3)))
            if isisomorphic(points, exemplar)
                return folderpath
            end
        end
    end
    return nothing
end

function add_to_database_entry(sourcepath::String, folderpath::String)
    mv(sourcepath, joinpath(folderpath, basename(sourcepath)))
end

function create_database_entry(points::Matrix{Float64x3}, sourcepath::String)
    dim, num_points = size(points)
    @assert haskey(DATABASE_CACHE, num_points)
    energy = riesz_energy(points)
    foldername = "PCREO-$(lpad(num_points, 8, '0'))-$(uppercase(string(uuid4())))"
    folderpath = joinpath(DATABASE_DIR, foldername)
    push!(DATABASE_CACHE[num_points], (energy, folderpath))
    mkdir(folderpath)
    add_to_database_entry(sourcepath, folderpath)
end

function main()
    for filename in readdir(SOURCE_DIR)
        path = joinpath(SOURCE_DIR, filename)
        if isfile(path)
            println("Reading: ", path)
            lines = readlines(path)
            @assert length(lines) % 3 == 0
            num_points = div(length(lines), 3)
            points = reshape(Float64x3.(BigFloat.(lines)), (3, num_points))
            if norm(riesz_gradient(points)) <= EPSILON * length(points)
                folderpath = find_in_database(points)
                if isnothing(folderpath)
                    println("Creating new database entry.")
                    create_database_entry(points, path)
                else
                    println("Matched existing database entry: ", folderpath)
                    add_to_database_entry(path, folderpath)
                end
            else
                println("WARNING: Gradient norm exceeds threshold.")
            end
        end
    end
end

# function readpoints(path)
#     lines = readlines(path)
#     @assert length(lines) % 3 == 0
#     reshape(Float64x3.(BigFloat.(lines)),
#             (3, div(length(lines), 3)))
# end

# function main()
#     for foldername in readdir(DATABASE_DIR)
#         println("Verifying directory: ", foldername)
#         folderpath = joinpath(DATABASE_DIR, foldername)
#         @assert isdir(folderpath)
#         exemplarpaths = [joinpath(folderpath, exemplarname)
#                          for exemplarname in readdir(folderpath)]
#         points = readpoints.(exemplarpaths)
#         sizes = unique!(size.(points))
#         @assert length(sizes) == 1
#         dim, num_points = sizes[1]
#         threshold = EPSILON * dim * num_points
#         Emin, Emax = extrema(riesz_energy.(points))
#         @assert (Emax - Emin) / (Emax + Emin) < threshold
#         for i = 1 : length(points)
#             for j = i+1 : length(points)
#                 if !isisomorphic(points[i], points[j]; verbose=true)
#                     println("ERROR:")
#                     println("    ", exemplarpaths[i])
#                     println("    ", exemplarpaths[j])
#                     println(isisomorphic(readpoints(exemplarpaths[i]), readpoints(exemplarpaths[j])))
#                 end
#             end
#         end
#     end
# end

main()
