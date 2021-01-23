using DZOptimization
using LinearAlgebra: cross, det
using NearestNeighbors


function isometric(a::PCREORecord, b::PCREORecord)
    if (a.dimension != b.dimension) || (a.num_points != b.num_points)
        return false
    end
    if !isapprox(a.energy, b.energy; rtol=1.0e-14)
        return false
    end
    if sort!(length.(a.facets)) != sort!(length.(b.facets))
        return false
    end
    a_buckets = bucket_by_first(sort!(labeled_distances(a.points)), 1.0e-14)
    b_buckets = bucket_by_first(sort!(labeled_distances(b.points)), 1.0e-14)
    if length.(a_buckets) != length.(b_buckets)
        return false
    end
    if !all(isapprox.(first.(middle.(a_buckets)),
                      first.(middle.(b_buckets)); rtol=1.0e-14))
        return false
    end
    b_tree = KDTree(b.points)
    _, i, j = middle(a_buckets[1])
    for (_, k, l) in b_buckets[1]
        mat = positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,k], b.points[:,l])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = positive_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
        mat = negative_transformation_matrix(
            a.points[:,i], a.points[:,j], b.points[:,l], b.points[:,k])
        @assert maximum(abs.(mat' * mat - one(mat))) < 1.0e-14
        if isometric(mat * a.points, b_tree)
            return true
        end
    end
    return false
end


function add_to_database(dataname)
    prefix = dataname[1:13]
    datapath = joinpath(PCREO_DIRECTORY, dataname)
    data = read_pcreo_file(datapath)
    for dirname in filter(startswith(prefix), readdir(DATABASE_DIRECTORY))
        dirpath = joinpath(DATABASE_DIRECTORY, dirname)
        @assert isdir(dirpath)
        reppath = joinpath(dirpath, dirname * ".csv")
        @assert isfile(reppath)
        representative = read_pcreo_file(reppath)
        if isometric(data, representative)
            mv(datapath, joinpath(dirpath, dataname))
            return dirname
        end
    end
    newdirname = dataname[1:end-4]
    newdirpath = joinpath(DATABASE_DIRECTORY, newdirname)
    mkdir(newdirpath)
    mv(datapath, joinpath(newdirpath, dataname))
    return newdirname
end


function main()

    println("Checking validity of database...")
    for dirname in readdir(DATABASE_DIRECTORY)
        dirpath = joinpath(DATABASE_DIRECTORY, dirname)
        @assert isdir(dirpath)
        reppath = joinpath(dirpath, dirname * ".csv")
        @assert isfile(reppath)
        representative = read_pcreo_file(reppath)
    end

    while true
        remaining = filter(startswith("PCREO"), readdir(PCREO_DIRECTORY))
        if !isempty(remaining)
            name = rand(remaining)
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


main()
