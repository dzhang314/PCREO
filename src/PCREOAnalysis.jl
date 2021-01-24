using DZOptimization: normalize!
using LinearAlgebra: cross
using ProgressMeter

push!(LOAD_PATH, @__DIR__)
using PCREO


function adjacent_facets(facets::Vector{Vector{Int}})
    result = Tuple{Int,Int}[]
    for i = 1 : length(facets) - 1
        for j = i+1 : length(facets)
            common_vertices = intersect(facets[i], facets[j])
            if length(common_vertices) >= 2
                @assert length(common_vertices) == 2
                push!(result, (i, j))
            end
        end
    end
    return result
end


function valence_table(facets::Vector{Vector{Int}})
    n = 0
    for facet in facets
        for vertex in facet
            @assert vertex >= 1
            n = max(n, vertex)
        end
    end
    result = zeros(Int, n)
    for facet in facets
        @simd ivdep for vertex in facet
            @inbounds result[vertex] += 1
        end
    end
    return result
end


function normal_vector(points::Vector{Vector{T}}) where {T}
    result = zeros(T, length(first(points)))
    for i = 1 : length(points) - 2
        for j = i+1 : length(points) - 1
            for k = j+1 : length(points)
                normal = cross(points[j] - points[i], points[k] - points[i])
                positive = all(signbit, normal' * p for p in points)
                negative = all(!signbit, normal' * p for p in points)
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





function main()
    for dirname in readdir(PCREO_DATABASE_DIRECTORY)
        graphpath = joinpath(PCREO_GRAPH_DIRECTORY, dirname * ".g6")
        if isfile(graphpath)
            if countlines(graphpath) > 1
                println(dirname)
                dirpath = joinpath(PCREO_DATABASE_DIRECTORY, dirname)
                for filename in readdir(dirpath)
                    filepath = joinpath(dirpath, filename)
                    record = PCREORecord(filepath)
                    dot_products = Float64[]
                    for (i, j) in adjacent_facets(record.facets)
                        i_points = [record.points[:,v]
                                    for v in record.facets[i]]
                        j_points = [record.points[:,v]
                                    for v in record.facets[j]]
                        i_normal = normal_vector(i_points)
                        j_normal = normal_vector(j_points)
                        push!(dot_products, 1.0 - i_normal' * j_normal)
                    end
                    if minimum(dot_products) < 1.0e-12
                        mv(filepath, joinpath(
                            PCREO_FACET_ERROR_DIRECTORY, filename))
                    end
                end
            end
        else
            println("WARNING: ", graphpath, " does not exist.")
        end
    end
end


main()
