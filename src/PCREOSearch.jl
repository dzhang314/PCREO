using DZOptimization: BFGSOptimizer, LBFGSOptimizer, step!
using DZOptimization.ExampleFunctions: riesz_energy
using LinearAlgebra: BLAS, eigvals!
using MultiFloats: Float64x2, Float64x3
using Printf: @printf, @sprintf
using StaticArrays: SVector, norm
using UUIDs: uuid4

push!(LOAD_PATH, @__DIR__)
using PCREO
using PointGroups

BLAS.set_num_threads(1)


function to_point_vector(points::AbstractMatrix{T}) where {T}
    dimension, num_points = size(points)
    @assert dimension == 3
    return [SVector{3,T}(view(points, :, i)) for i = 1 : num_points]
end


function to_point_matrix(points::Vector{SVector{N,T}}) where {T,N}
    result = Matrix{T}(undef, N, length(points))
    for (i, point) in enumerate(points)
        @simd ivdep for j = 1 : N
            @inbounds result[j,i] = point[j]
        end
    end
    return result
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


initial_step_size(::Type{Float64}) = Float64(2^-20)
initial_step_size(::Type{Float64x2}) = Float64x2(2^-40)
initial_step_size(::Type{Float64x3}) = Float64x3(2^-60)


function refine(points::Matrix{T}) where {T}
    constrain_sphere!(points)
    energy = riesz_energy(points)
    @assert isfinite(energy)
    while true
        opt = LBFGSOptimizer(
            riesz_energy, spherical_riesz_gradient!, constrain_sphere!,
            points, initial_step_size(T), 25)
        run!(opt; quiet=!(stdout isa Base.TTY))
        next_energy = opt.current_objective_value[]
        next_points = opt.current_point
        if next_energy < energy
            energy = next_energy
            points = next_points
        else
            break
        end
    end
    return (points, energy)
end


function save_point_configuration!(filename::String,
                                   points::Vector{SVector{3,Float64x3}},
                                   initial_points::Matrix{Float64})
    dimension = 3
    num_points = length(points)

    group_name, rotation = identify_point_group(
        isometries(points, 2^-60), 2^-60)
    @simd ivdep for i = 1 : num_points
        @inbounds points[i] = rotation * points[i]
    end

    point_matrix = to_point_matrix(points)
    hess = spherical_riesz_hessian(point_matrix)
    symmetrize!(hess)
    eigenvalues = sort!(eigvals!(Float64.(hess)))

    num_expected_zeros = div(dimension * (dimension - 1), 2) + num_points
    noise_floor = maximum(abs.(eigenvalues[1:num_expected_zeros]))
    first_eigenvalue = eigenvalues[num_expected_zeros + 1]

    if !(first_eigenvalue > noise_floor)
        return nothing
    end

    energy = riesz_energy(point_matrix)
    gradient_norm = maximum(norm.(to_point_vector(
        spherical_riesz_gradient(point_matrix))))

    facets = convex_hull_facets(points)
    packing_rad = packing_radius(points, facets)
    covering_rad = covering_radius(points, facets)

    open(filename, "w+") do io
        println(io, dimension)
        println(io, num_points)
        println(io, '"', group_name, '"')
        println(io, energy)
        println(io, first_eigenvalue)
        @printf(io, "%.25f\n", packing_rad)
        @printf(io, "%.25f\n", covering_rad)
        println(io)
        @printf(io, "%.3f\n", -log2(Float64(gradient_norm)))
        @printf(io, "%.3f\n", -log2(noise_floor / first_eigenvalue))
        @printf(io, "%.25f\n", parallel_facet_distance(points, facets))
        println(io)
        for point in points
            println(io, join([@sprintf("%+.25f", coord)
                              for coord in point], ", "))
        end
        println(io)
        num_digits = length(string(num_points))
        for facet in facets
            println(io, join([lpad(index, num_digits, ' ')
                              for index in facet], ", "))
        end
        println(io)
        for point in eachcol(initial_points)
            println(io, join([@sprintf("%+.17f", coord)
                              for coord in point], ", "))
        end
    end
    return filename
end


function generate_and_save_point_configuration(num_points::Int)
    initial_points = randn(3, num_points)
    constrain_sphere!(initial_points)
    points1, energy1 = refine(initial_points)
    points2, energy2 = refine(Float64x2.(points1))
    points3, energy3 = refine(Float64x3.(points2))
    try
        filename = save_point_configuration!(
            joinpath(PCREO_OUTPUT_DIRECTORY,
                "PCREO-03-$(lpad(num_points, 8, '0'))-$(uuid4()).csv"),
            to_point_vector(points3), initial_points)
        if isnothing(filename)
            println("Failed to generate stable point configuration.")
        else
            println("Saved $num_points-point configuration to $filename.")
        end
    catch e
        if e isa AssertionError
            println(e)
            filename = "FAIL-03-$(lpad(num_points, 8, '0'))-$(uuid4()).csv"
            open(filename, "w+") do io
                for point in eachcol(initial_points)
                    println(io, join(string.(point), ", "))
                end
            end
        else
            rethrow(e)
        end
    end
end


function generate_and_save_symmetric_point_configuration(
        group::Function, orbits::Vector{Function},
        num_points::Int, max_points::Int)

    f1, g1! = symmetrized_riesz_functors(Float64, group, orbits)
    num_full_points = (num_points * length(f1.group) +
                       length(f1.external_points))

    if num_full_points > max_points
        return nothing
    end

    initial_points = randn(3, num_points)
    constrain_sphere!(initial_points)

    opt1 = BFGSOptimizer(f1, g1!, constrain_sphere!, initial_points, 1.0e-6)
    run!(opt1; quiet=!(stdout isa Base.TTY))

    f2, g2! = symmetrized_riesz_functors(Float64x2, group, orbits)
    opt2 = BFGSOptimizer(Float64x2, f2, g2!, constrain_sphere!, opt1)
    run!(opt2; quiet=!(stdout isa Base.TTY))

    f3, g3! = symmetrized_riesz_functors(Float64x3, group, orbits)
    opt3 = BFGSOptimizer(Float64x3, f3, g3!, constrain_sphere!, opt2)
    run!(opt3; quiet=!(stdout isa Base.TTY))

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

    try
        filename = save_point_configuration!(
            joinpath(PCREO_OUTPUT_DIRECTORY,
                "PCREO-03-$(lpad(num_full_points, 8, '0'))-$(uuid4()).csv"),
            to_point_vector(full_points), initial_points)
        if isnothing(filename)
            println("Failed to generate stable point configuration.")
        else
            println("Saved $num_full_points-point configuration to $filename.")
        end
    catch e
        if e isa AssertionError
            println(e)
            filename = "FAIL-03-$(lpad(num_full_points, 8, '0'))-$(uuid4()).csv"
            open(filename, "w+") do io
                println(io, group)
                println(io, orbits)
                for point in eachcol(initial_points)
                    println(io, join(string.(point), ", "))
                end
            end
        else
            rethrow(e)
        end
    end
end


function main()
    while true
        for (group_function, orbit_functions) in CHIRAL_POLYHEDRAL_POINT_GROUPS
            for selected_orbits in powerset(orbit_functions)
                for num_points = 1 : 99
                    generate_and_save_symmetric_point_configuration(
                        group_function, selected_orbits, num_points, 499)
                end
            end
        end
        for num_points = 4 : 499
            generate_and_save_point_configuration(num_points)
        end
    end
end


main()
