using NearestNeighbors: KDTree

push!(LOAD_PATH, @__DIR__)
using PCREO


function isometric(a::PCREORecord, b::PCREORecord)
    if (a.dimension != b.dimension) || (a.num_points != b.num_points)
        return false
    end
    if !isapprox(a.energy, b.energy; rtol=1.0e-14)
        return false
    end
    identical_facets = (sort!(length.(a.facets)) == sort!(length.(b.facets)))
    # if !all(isapprox.(first.(middle.(a_buckets)),
    #                   first.(middle.(b_buckets)); rtol=1.0e-14))
    #     return false
    # end
    b_tree = KDTree(b.points)
    dist = minimum(matching_distance(mat * a.points, b_tree)
                   for mat in candidate_isometries(a.points, b.points))
    if dist < 1.0e-10
        if !identical_facets
            println("WARNING: Found isometric point configurations" *
                    " with different facets.")
        end
        return true
    end
    return false
end


function add_to_database(dataname)
    prefix = dataname[1:13]
    datapath = joinpath(PCREO_OUTPUT_DIRECTORY, dataname)
    data = PCREORecord(datapath)
    for dirname in filter(startswith(prefix), readdir(PCREO_DATABASE_DIRECTORY))
        dirpath = joinpath(PCREO_DATABASE_DIRECTORY, dirname)
        @assert isdir(dirpath)
        reppath = joinpath(dirpath, dirname * ".csv")
        @assert isfile(reppath)
        representative = PCREORecord(reppath)
        if isometric(data, representative)
            mv(datapath, joinpath(dirpath, dataname))
            return dirname
        end
    end
    newdirname = dataname[1:end-4]
    newdirpath = joinpath(PCREO_DATABASE_DIRECTORY, newdirname)
    mkdir(newdirpath)
    mv(datapath, joinpath(newdirpath, dataname))
    return newdirname
end


function main()

    min_n = parse(Int, ARGS[1])
    max_n = parse(Int, ARGS[2])

    while true
        remaining = filter(startswith("PCREO"), readdir(PCREO_OUTPUT_DIRECTORY))
        if !isempty(remaining)
            name = rand(remaining)
            num_particles = parse(Int, name[10:13])
            if min_n <= num_particles <= max_n
                print(length(remaining), '\t', name, " => ")
                flush(stdout)
                try
                    found = add_to_database(name)
                    if occursin(found, name)
                        println("new")
                    else
                        println(found)
                    end
                catch e
                    if e isa AssertionError
                        println("ERROR: ", e)
                    else
                        rethrow(e)
                    end
                end
                flush(stdout)
            end
        end
    end

end


main()
