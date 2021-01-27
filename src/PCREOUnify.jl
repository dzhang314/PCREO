using NearestNeighbors
using ProgressMeter

push!(LOAD_PATH, @__DIR__)
using PCREO


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
    candidate_isometries = Matrix{Float64}[]
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
        @showprogress for dirname in dirnames
            database[dirname] = PCREORecord(dirname)
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
