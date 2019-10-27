println(raw" _____   _____")
println(raw"|  __ \ / ____|                David Zhang")
println(raw"| |__) | |     _ __ ___  ___")
println(raw"|  ___/| |    | '__/ _ \/ _ \    L--BFGS")
println(raw"| |    | |____| | |  __/ (_) |  Optimized")
println(raw"|_|     \_____|_|  \___|\___/  For Spheres")
println()
println("PCREO is free software distributed under the terms of the MIT license.")
println()

################################################################################

using Random
using UUIDs

using DZLinearAlgebra
using DZOptimization
using MultiFloats

set_zero_subnormals(true)
use_very_sloppy_multifloat_arithmetic()

const TERM = stdout isa Base.TTY

function rmk(args...)::Nothing
    if TERM
        print(stdout, "\r", args..., "\33[K")
        flush(stdout)
    end
end

function say(args...)::Nothing
    if TERM
        println(stdout, "\r", args..., "\33[K")
        flush(stdout)
    else
        println(stdout, args...)
    end
end

################################################################################

function riesz_energy(points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    energy = zero(T)
    @inbounds for i = 2 : num_points
        for j = 1 : i-1
            dist_sq = zero(T)
            @simd ivdep for k = 1 : dim
                dist = points[k,i] - points[k,j]
                dist_sq += dist * dist
            end
            energy += inv(unsafe_sqrt(dist_sq))
        end
    end
    return energy
end

function constrain_gradient!(grad::AbstractMatrix{T},
                             points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    @inbounds for i = 1 : num_points
        overlap = zero(T)
        @simd ivdep for k = 1 : dim
            overlap += points[k,i] * grad[k,i]
        end
        @simd ivdep for k = 1 : dim
            grad[k,i] -= overlap * points[k,i]
        end
    end
    return grad
end

function riesz_gradient!(grad::AbstractMatrix{T},
                         points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    @inbounds for j = 1 : num_points
        @simd ivdep for k = 1 : dim
            grad[k,j] = zero(T)
        end
        for i = 1 : num_points
            if i != j
                dist_sq = zero(T)
                @simd ivdep for k = 1 : dim
                    dist = points[k,i] - points[k,j]
                    dist_sq += dist * dist
                end
                inv_dist_cubed = unsafe_sqrt(dist_sq) / (dist_sq * dist_sq)
                @simd ivdep for k = 1 : dim
                    grad[k,j] += (points[k,i] - points[k,j]) * inv_dist_cubed
                end
            end
        end
    end
    constrain_gradient!(grad, points)
    return grad
end

function riesz_gradient(points::AbstractMatrix{T}) where {T<:Real}
    grad = similar(points)
    riesz_gradient!(grad, points)
    return grad
end

################################################################################

function rmk_table_row(opt)
    rmk(opt.current_iteration[1], " | ", opt.current_objective[1])
end

function say_table_row(opt)
    say(opt.current_iteration[1], " | ", opt.current_objective[1])
end

function run!(opt::ConstrainedLBFGSOptimizer{S1,S2,S3,T,N}
        ) where {S1,S2,S3,T<:Real,N}
    rmk_table_row(opt)
    last_rmk_time = time_ns()
    old_objective = opt.current_objective[1]
    threshold = eps(T) * length(opt.current_point) # * 2^div(precision(T), 13)
    threshold2 = threshold * threshold
    while true
        step!(opt)
        if time_ns() - last_rmk_time > UInt(100_000_000)
            rmk_table_row(opt)
            last_rmk_time = time_ns()
        end
        new_objective = opt.current_objective[1]
        objective_criterion = (abs(new_objective - old_objective) < threshold * old_objective)
        point_criterion = (norm2(opt.delta_point) < threshold2)
        if objective_criterion || point_criterion
            say_table_row(opt)
            return opt
        end
        old_objective = new_objective
    end
end

function optimize(::Type{T}, initial_points::Matrix{U},
                  m::Int) where {T<:Real,U<:Real}
    say("Optimizing ", size(initial_points),
        " points with ", precision(T), "-bit precision")
    opt = riesz_lbfgs_optimizer(T.(initial_points), m)
    run!(opt)
    return opt.current_point
end

riesz_lbfgs_optimizer(initial_point, m) = constrained_lbfgs_optimizer(
    riesz_energy, riesz_gradient!, normalize_columns!, initial_point, m)

function main(m::Int, dim::Int, num_points::Int)
    points = optimize(Float64, randn(dim, num_points), m)
    points = optimize(Float64x2, points, m)
    points = optimize(Float64x4, points, m)
    setprecision(100) do
        open("PCREO-$(lpad(num_points, 4, '0'))-$(uppercase(string(uuid4()))).txt", "w+") do io
            for x in points
                println(io, BigFloat(x))
            end
        end
    end
    say("Saved results to file.")
    say()
end

while true
    for i = 10 : 500
        main(10, 3, i)
    end
end
