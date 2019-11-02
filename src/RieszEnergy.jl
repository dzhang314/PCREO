module RieszEnergy

export riesz_energy, constrain_gradient!, riesz_gradient!, riesz_gradient,
    unconstrained_riesz_gradient!, unconstrained_riesz_gradient,
    unconstrained_riesz_hessian!, unconstrained_riesz_hessian

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

function unconstrained_riesz_gradient!(
        grad::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T<:Real}
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
    return grad
end

function unconstrained_riesz_hessian!(
        hess::AbstractArray{T,4}, points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    @inbounds for k = 1 : num_points
        for s = 1 : num_points
            if s == k
                for l = 1 : dim
                    @simd ivdep for t = 1 : dim
                        hess[t,s,l,k] = zero(T)
                    end
                end
                for j = 1 : num_points
                    if j != k
                        dist_sq = zero(T)
                        @simd ivdep for d = 1 : dim
                            temp = points[d,k] - points[d,j]
                            dist_sq += temp * temp
                        end
                        dist = unsafe_sqrt(dist_sq)
                        dist_cb = dist * dist_sq
                        for l = 1 : dim
                            @simd ivdep for t = 1 : dim
                                plkj = points[l,k] - points[l,j]
                                ptkj = points[t,k] - points[t,j]
                                temp = (plkj * ptkj) / (dist_sq * dist_cb)
                                hess[t,s,l,k] += (temp + temp + temp)
                            end
                            hess[l,s,l,k] -= inv(dist_cb)
                        end
                    end
                end
            else
                dist_sq = zero(T)
                @simd ivdep for d = 1 : dim
                    temp = points[d,k] - points[d,s]
                    dist_sq += temp * temp
                end
                dist = unsafe_sqrt(dist_sq)
                dist_cb = dist * dist_sq
                for l = 1 : dim
                    @simd ivdep for t = 1 : dim
                        plks = points[l,k] - points[l,s]
                        ptsk = points[t,s] - points[t,k]
                        temp = (plks * ptsk) / (dist_sq * dist_cb)
                        hess[t,s,l,k] = (temp + temp + temp)
                    end
                    hess[l,s,l,k] += inv(dist_cb)
                end
            end
        end
    end
    return hess
end

function riesz_gradient!(grad::AbstractMatrix{T},
                         points::AbstractMatrix{T}) where {T<:Real}
    unconstrained_riesz_gradient!(grad, points)
    constrain_gradient!(grad, points)
    return grad
end

function unconstrained_riesz_gradient(points::AbstractMatrix{T}) where {T<:Real}
    grad = similar(points)
    unconstrained_riesz_gradient!(grad, points)
    return grad
end

function unconstrained_riesz_hessian(points::AbstractMatrix{T}) where {T<:Real}
    dim, num_points = size(points)
    hess = Array{T,4}(undef, dim, num_points, dim, num_points)
    unconstrained_riesz_hessian!(hess, points)
    return hess
end

function riesz_gradient(points::AbstractMatrix{T}) where {T<:Real}
    grad = similar(points)
    riesz_gradient!(grad, points)
    return grad
end

end # module RieszEnergy
