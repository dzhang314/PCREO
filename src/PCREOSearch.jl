using DZOptimization: BFGSOptimizer, normalize_columns!
using MultiFloats: Float64x2, Float64x3
using UUIDs: uuid4

push!(LOAD_PATH, @__DIR__)
using PCREO: run!, spherical_riesz_gradient_norm,
    spherical_riesz_hessian_spectral_gap,
    convex_hull_facets, parallel_facet_distance
using PCREOSymmetry: POLYHEDRAL_POINT_GROUPS, symmetrized_riesz_functors


function constrain_sphere!(points)
    normalize_columns!(points)
    return true
end


function generate_and_save_configuration(
        group::Function, orbits::Vector{Function},
        num_points::Int, max_points::Int)

    f1, g1! = symmetrized_riesz_functors(Float64, group, orbits)
    num_full_points = (num_points * length(f1.group) +
                       length(f1.external_points))

    if num_full_points > max_points
        return nothing
    end

    initial_points = normalize_columns!(randn(3, num_points))

    opt1 = BFGSOptimizer(f1, g1!, constrain_sphere!, initial_points, 1.0e-6)
    run!(opt1)

    f2, g2! = symmetrized_riesz_functors(Float64x2, group, orbits)
    opt2 = BFGSOptimizer(Float64x2, f2, g2!, constrain_sphere!, opt1)
    run!(opt2)

    f3, g3! = symmetrized_riesz_functors(Float64x3, group, orbits)
    opt3 = BFGSOptimizer(Float64x3, f3, g3!, constrain_sphere!, opt2)
    run!(opt3)

    full_points = Matrix{Float64x3}(undef, 3, num_full_points)
    k = 0
    for g in f3.group
        for col in eachcol(opt3.current_point)
            full_points[:,k+=1] .= g * col
        end
    end
    for point in f3.external_points
        full_points[:,k+=1] .= point
    end
    @assert k == num_full_points

    c2 = spherical_riesz_hessian_spectral_gap(full_points)
    if c2 < 1.0
        c1 = Float64(spherical_riesz_gradient_norm(full_points))
        facets = convex_hull_facets(Float64.(full_points))
        c3 = Float64(parallel_facet_distance(full_points, facets))
        prefix = "PCREO-03-$(lpad(num_full_points, 4, '0'))"
        filename = "$prefix-$(uuid4()).csv"
        open(filename, "w+") do io
            println(io, 3)
            println(io, num_full_points)
            println(io, opt3.current_objective_value[])
            println(io, '"', group, '"')
            if !isempty(orbits)
                println(io, join(['"' * string(f) * '"'
                                  for f in orbits], ", "))
            end
            println(io, c1)
            println(io, c2)
            println(io, c3)
            println(io)
            for col in eachcol(full_points)
                println(io, join(string.(col), ", "))
            end
            println(io)
            for facet in facets
                println(io, join(string.(facet), ", "))
            end
            println(io)
            for col in eachcol(initial_points)
                println(io, join(string.(col), ", "))
            end
        end
    end

    return nothing
end


function powerset(xs::Vector{T}) where {T}
    result = [T[]]
    for x in xs
        for i = 1 : length(result)
            push!(result, push!(copy(result[i]), x))
        end
    end
    return result
end


function main()
    while true
        for num_points = 1 : 100
            for (group_function, orbit_functions) in POLYHEDRAL_POINT_GROUPS
                for selected_orbits in powerset(orbit_functions)
                    generate_and_save_configuration(
                        group_function, selected_orbits, num_points, 1000)
                end
            end
        end
    end
end


main()
