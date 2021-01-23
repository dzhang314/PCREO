push!(LOAD_PATH, @__DIR__)

using NearestNeighbors
using PCREO
using ProgressMeter


function push_isometry!(isometries, matrix)
    @assert maximum(abs.(matrix' * matrix - one(matrix))) < 1.0e-12
    push!(isometries, matrix)
end


function unify(bucket, database)
    @assert iszero(last(bucket)[1] - first(bucket)[1])
    a_name = bucket[1][2]
    b_name = bucket[2][2]
    a = database[a_name]
    b = database[b_name]
    if size(a.points) != size(b.points)
        println("NOTE: $a_name and $b_name have identical energy" *
                " but different dimensionality.")
        return nothing
    end
    identical_facets = (sort!(length.(a.facets)) == sort!(length.(b.facets)))
    a_buckets = bucket_by_first(sort!(labeled_distances(a.points)), 1.0e-10)
    b_buckets = bucket_by_first(sort!(labeled_distances(b.points)), 1.0e-10)
    a_spread = maximum(first.(last.(a_buckets)) - first.(first.(a_buckets)))
    b_spread = maximum(first.(last.(b_buckets)) - first.(first.(b_buckets)))
    @assert 0.0 <= a_spread < 1.0e-10
    @assert 0.0 <= b_spread < 1.0e-10
    identical_barcodes = (length.(a_buckets) == length.(b_buckets))
    bucket_index = minimum([(length(bkt), idx)
                            for (idx, bkt) in enumerate(a_buckets)
                            if 0.25 < middle(bkt)[1] < 1.75])[2]
    candidate_isometries = Matrix{Float64}[]
    _, i, j = middle(a_buckets[bucket_index])
    for (_, k, l) in b_buckets[bucket_index]
        push_isometry!(candidate_isometries, positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l]))
        push_isometry!(candidate_isometries, negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l]))
        push_isometry!(candidate_isometries, positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k]))
        push_isometry!(candidate_isometries, negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k]))
    end
    b_tree = KDTree(b.points)
    dist = minimum(matching_distance(mat * a.points, b_tree)
                   for mat in candidate_isometries)
    @assert 0.0 <= dist
    if dist < 1.0e-10
        println("Unifying $a_name and $b_name.")
        if !identical_facets
            println("WARNING: $a_name and $b_name are isometric but have" *
                    " different facets.")
        end
        if !identical_barcodes
            println("WARNING: $a_name and $b_name are isometric but have" *
                    " different length barcodes.")
        end
        a_entries = readdir(joinpath(PCREO_DATABASE_DIRECTORY, a_name))
        b_entries = readdir(joinpath(PCREO_DATABASE_DIRECTORY, b_name))
        if length(a_entries) >= length(b_entries)
            for b_entry in b_entries
                mv(joinpath(PCREO_DATABASE_DIRECTORY, b_name, b_entry),
                   joinpath(PCREO_DATABASE_DIRECTORY, a_name, b_entry))
            end
            rm(joinpath(PCREO_DATABASE_DIRECTORY, b_name))
        else
            for a_entry in a_entries
                mv(joinpath(PCREO_DATABASE_DIRECTORY, a_name, a_entry),
                   joinpath(PCREO_DATABASE_DIRECTORY, b_name, a_entry))
            end
            rm(joinpath(PCREO_DATABASE_DIRECTORY, a_name))
        end
    else
        println("WARNING: $a_name and $b_name have identical energy" *
                " but are not isometric ($dist).")
    end
end


function main()
    while true
        dirnames = readdir(PCREO_DATABASE_DIRECTORY)
        database = Dict{String,PCREORecord}()
        counts = Dict{String,Int}()
        @showprogress for dirname in dirnames
            database[dirname] = PCREORecord(dirname)
            counts[dirname] = length(readdir(
                joinpath(PCREO_DATABASE_DIRECTORY, dirname)))
        end

        energy_buckets = bucket_by_first(
            sort!([(database[dirname].energy, dirname)
                for dirname in dirnames]),
            1.0e-10)

        for bucket in energy_buckets
            if length(bucket) > 1
                try
                    unify(bucket, database)
                catch e
                    if e isa AssertionError
                        println("ERROR: ", e)
                    else
                        rethrow(e)
                    end
                end
            end
        end
    end
end


main()
