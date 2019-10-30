module RieszEnergy

export riesz_energy, constrain_gradient!, riesz_gradient!, riesz_gradient

using DZLinearAlgebra: unsafe_sqrt

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

end # module RieszEnergy
