module PCREOSymmetry

using DZOptimization: norm
using DZOptimization.ExampleFunctions: riesz_energy,
    constrain_riesz_gradient_sphere!
using GenericSVD: svd
using LinearAlgebra: det, svd
using StaticArrays: SArray, SVector
using Suppressor: @suppress

export chiral_tetrahedral_group, full_tetrahedral_group, pyritohedral_group,
    chiral_octahedral_group, full_octahedral_group,
    chiral_icosahedral_group, full_icosahedral_group,
    tetrahedron_vertices, tetrahedron_edge_centers, tetrahedron_face_centers,
    tetrahedron_rotoinversion_centers, pyritohedron_vertices,
    octahedron_vertices, octahedron_edge_centers, octahedron_face_centers,
    icosahedron_vertices, icosahedron_edge_centers, icosahedron_face_centers,
    POLYHEDRAL_POINT_GROUPS,
    multiplication_table, count_central_elements, degenerate_orbits,
    symmetrized_riesz_energy, symmetrized_riesz_gradient!,
    symmetrized_riesz_functors


######################################################## POLYHEDRAL POINT GROUPS


function chiral_tetrahedral_group(::Type{T}) where {T}
    M = SArray{Tuple{3,3},T,2,9}
    _zero = zero(T)
    _one = one(T)

    return [
        M(+_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, +_one),
        M(+_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, +_one),

        M(_zero, _zero, +_one, +_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, -_one, +_one, _zero, _zero, _zero, -_one, _zero),
        M(_zero, _zero, -_one, -_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, +_one, -_one, _zero, _zero, _zero, -_one, _zero),

        M(_zero, +_one, _zero, _zero, _zero, +_one, +_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, -_one, +_one, _zero, _zero),
        M(_zero, +_one, _zero, _zero, _zero, -_one, -_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, +_one, -_one, _zero, _zero),
    ]
end


function full_tetrahedral_group(::Type{T}) where {T}
    M = SArray{Tuple{3,3},T,2,9}
    _zero = zero(T)
    _one = one(T)

    return [
        M(+_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, +_one),
        M(+_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, +_one),

        M(+_one, _zero, _zero, _zero, _zero, +_one, _zero, +_one, _zero),
        M(+_one, _zero, _zero, _zero, _zero, -_one, _zero, -_one, _zero),
        M(-_one, _zero, _zero, _zero, _zero, +_one, _zero, -_one, _zero),
        M(-_one, _zero, _zero, _zero, _zero, -_one, _zero, +_one, _zero),

        M(_zero, _zero, +_one, +_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, -_one, +_one, _zero, _zero, _zero, -_one, _zero),
        M(_zero, _zero, -_one, -_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, +_one, -_one, _zero, _zero, _zero, -_one, _zero),

        M(_zero, +_one, _zero, +_one, _zero, _zero, _zero, _zero, +_one),
        M(_zero, -_one, _zero, +_one, _zero, _zero, _zero, _zero, -_one),
        M(_zero, -_one, _zero, -_one, _zero, _zero, _zero, _zero, +_one),
        M(_zero, +_one, _zero, -_one, _zero, _zero, _zero, _zero, -_one),

        M(_zero, +_one, _zero, _zero, _zero, +_one, +_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, -_one, +_one, _zero, _zero),
        M(_zero, +_one, _zero, _zero, _zero, -_one, -_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, +_one, -_one, _zero, _zero),

        M(_zero, _zero, +_one, _zero, +_one, _zero, +_one, _zero, _zero),
        M(_zero, _zero, -_one, _zero, -_one, _zero, +_one, _zero, _zero),
        M(_zero, _zero, +_one, _zero, -_one, _zero, -_one, _zero, _zero),
        M(_zero, _zero, -_one, _zero, +_one, _zero, -_one, _zero, _zero),
    ]
end


pyritohedral_group(::Type{T}) where {T} =
    vcat(+chiral_tetrahedral_group(T),
         -chiral_tetrahedral_group(T))


function chiral_octahedral_group(::Type{T}) where {T}
    M = SArray{Tuple{3,3},T,2,9}
    _zero = zero(T)
    _one = one(T)

    return [
        M(+_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, +_one),
        M(+_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, +_one),

        M(_zero, _zero, +_one, +_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, -_one, +_one, _zero, _zero, _zero, -_one, _zero),
        M(_zero, _zero, -_one, -_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, +_one, -_one, _zero, _zero, _zero, -_one, _zero),

        M(_zero, +_one, _zero, _zero, _zero, +_one, +_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, -_one, +_one, _zero, _zero),
        M(_zero, +_one, _zero, _zero, _zero, -_one, -_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, +_one, -_one, _zero, _zero),

        M(+_one, _zero, _zero, _zero, _zero, +_one, _zero, -_one, _zero),
        M(+_one, _zero, _zero, _zero, _zero, -_one, _zero, +_one, _zero),
        M(-_one, _zero, _zero, _zero, _zero, +_one, _zero, +_one, _zero),
        M(-_one, _zero, _zero, _zero, _zero, -_one, _zero, -_one, _zero),

        M(_zero, -_one, _zero, +_one, _zero, _zero, _zero, _zero, +_one),
        M(_zero, +_one, _zero, +_one, _zero, _zero, _zero, _zero, -_one),
        M(_zero, +_one, _zero, -_one, _zero, _zero, _zero, _zero, +_one),
        M(_zero, -_one, _zero, -_one, _zero, _zero, _zero, _zero, -_one),

        M(_zero, _zero, +_one, _zero, -_one, _zero, +_one, _zero, _zero),
        M(_zero, _zero, -_one, _zero, +_one, _zero, +_one, _zero, _zero),
        M(_zero, _zero, +_one, _zero, +_one, _zero, -_one, _zero, _zero),
        M(_zero, _zero, -_one, _zero, -_one, _zero, -_one, _zero, _zero),
    ]
end


full_octahedral_group(::Type{T}) where {T} =
    vcat(+chiral_octahedral_group(T),
         -chiral_octahedral_group(T))


function chiral_icosahedral_group(::Type{T}) where {T}
    M = SArray{Tuple{3,3},T,2,9}
    _zero = zero(T)
    _one = one(T)
    two = _one + _one
    four = two + two
    five = four + _one
    half = inv(two)
    quarter = inv(four)
    hphi = quarter * (sqrt(five) + _one)
    hpsi = quarter * (sqrt(five) - _one)

    return [
        # Identity matrix
        M(+_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, +_one),

        # 180-degree rotations about coordinate axes
        M(+_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, +_one, _zero, _zero, _zero, -_one),
        M(-_one, _zero, _zero, _zero, -_one, _zero, _zero, _zero, +_one),

        # 120-degree rotations about [±1, ±1, ±1]
        M(_zero, _zero, +_one, +_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, -_one, +_one, _zero, _zero, _zero, -_one, _zero),
        M(_zero, _zero, -_one, -_one, _zero, _zero, _zero, +_one, _zero),
        M(_zero, _zero, +_one, -_one, _zero, _zero, _zero, -_one, _zero),
        M(_zero, +_one, _zero, _zero, _zero, +_one, +_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, -_one, +_one, _zero, _zero),
        M(_zero, +_one, _zero, _zero, _zero, -_one, -_one, _zero, _zero),
        M(_zero, -_one, _zero, _zero, _zero, +_one, -_one, _zero, _zero),

        M(+half, +hphi, +hpsi, +hphi, -hpsi, -half, -hpsi, +half, -hphi),
        M(+half, +hphi, +hpsi, -hphi, +hpsi, +half, +hpsi, -half, +hphi),
        M(+half, +hphi, -hpsi, +hphi, -hpsi, +half, +hpsi, -half, -hphi),
        M(+half, +hphi, -hpsi, -hphi, +hpsi, -half, -hpsi, +half, +hphi),
        M(+half, -hphi, +hpsi, +hphi, +hpsi, -half, +hpsi, +half, +hphi),
        M(+half, -hphi, +hpsi, -hphi, -hpsi, +half, -hpsi, -half, -hphi),
        M(+half, -hphi, -hpsi, +hphi, +hpsi, +half, -hpsi, -half, +hphi),
        M(+half, -hphi, -hpsi, -hphi, -hpsi, -half, +hpsi, +half, -hphi),

        M(-half, +hphi, +hpsi, +hphi, +hpsi, +half, +hpsi, +half, -hphi),
        M(-half, +hphi, +hpsi, -hphi, -hpsi, -half, -hpsi, -half, +hphi),
        M(-half, +hphi, -hpsi, +hphi, +hpsi, -half, -hpsi, -half, -hphi),
        M(-half, +hphi, -hpsi, -hphi, -hpsi, +half, +hpsi, +half, +hphi),
        M(-half, -hphi, +hpsi, +hphi, -hpsi, +half, -hpsi, +half, +hphi),
        M(-half, -hphi, +hpsi, -hphi, +hpsi, -half, +hpsi, -half, -hphi),
        M(-half, -hphi, -hpsi, +hphi, -hpsi, -half, +hpsi, -half, +hphi),
        M(-half, -hphi, -hpsi, -hphi, +hpsi, +half, -hpsi, +half, -hphi),

        M(+hphi, +hpsi, +half, +hpsi, +half, -hphi, -half, +hphi, +hpsi),
        M(+hphi, +hpsi, +half, -hpsi, -half, +hphi, +half, -hphi, -hpsi),
        M(+hphi, +hpsi, -half, +hpsi, +half, +hphi, +half, -hphi, +hpsi),
        M(+hphi, +hpsi, -half, -hpsi, -half, -hphi, -half, +hphi, -hpsi),
        M(+hphi, -hpsi, +half, +hpsi, -half, -hphi, +half, +hphi, -hpsi),
        M(+hphi, -hpsi, +half, -hpsi, +half, +hphi, -half, -hphi, +hpsi),
        M(+hphi, -hpsi, -half, +hpsi, -half, +hphi, -half, -hphi, -hpsi),
        M(+hphi, -hpsi, -half, -hpsi, +half, -hphi, +half, +hphi, +hpsi),

        M(-hphi, +hpsi, +half, +hpsi, -half, +hphi, +half, +hphi, +hpsi),
        M(-hphi, +hpsi, +half, -hpsi, +half, -hphi, -half, -hphi, -hpsi),
        M(-hphi, +hpsi, -half, +hpsi, -half, -hphi, -half, -hphi, +hpsi),
        M(-hphi, +hpsi, -half, -hpsi, +half, +hphi, +half, +hphi, -hpsi),
        M(-hphi, -hpsi, +half, +hpsi, +half, +hphi, -half, +hphi, -hpsi),
        M(-hphi, -hpsi, +half, -hpsi, -half, -hphi, +half, -hphi, +hpsi),
        M(-hphi, -hpsi, -half, +hpsi, +half, -hphi, +half, -hphi, -hpsi),
        M(-hphi, -hpsi, -half, -hpsi, -half, +hphi, -half, +hphi, +hpsi),

        M(+hpsi, +half, +hphi, +half, -hphi, +hpsi, +hphi, +hpsi, -half),
        M(+hpsi, +half, +hphi, -half, +hphi, -hpsi, -hphi, -hpsi, +half),
        M(+hpsi, +half, -hphi, +half, -hphi, -hpsi, -hphi, -hpsi, -half),
        M(+hpsi, +half, -hphi, -half, +hphi, +hpsi, +hphi, +hpsi, +half),
        M(+hpsi, -half, +hphi, +half, +hphi, +hpsi, -hphi, +hpsi, +half),
        M(+hpsi, -half, +hphi, -half, -hphi, -hpsi, +hphi, -hpsi, -half),
        M(+hpsi, -half, -hphi, +half, +hphi, -hpsi, +hphi, -hpsi, +half),
        M(+hpsi, -half, -hphi, -half, -hphi, +hpsi, -hphi, +hpsi, -half),

        M(-hpsi, +half, +hphi, +half, +hphi, -hpsi, -hphi, +hpsi, -half),
        M(-hpsi, +half, +hphi, -half, -hphi, +hpsi, +hphi, -hpsi, +half),
        M(-hpsi, +half, -hphi, +half, +hphi, +hpsi, +hphi, -hpsi, -half),
        M(-hpsi, +half, -hphi, -half, -hphi, -hpsi, -hphi, +hpsi, +half),
        M(-hpsi, -half, +hphi, +half, -hphi, -hpsi, +hphi, +hpsi, +half),
        M(-hpsi, -half, +hphi, -half, +hphi, +hpsi, -hphi, -hpsi, -half),
        M(-hpsi, -half, -hphi, +half, -hphi, +hpsi, -hphi, -hpsi, +half),
        M(-hpsi, -half, -hphi, -half, +hphi, -hpsi, +hphi, +hpsi, -half),
    ]
end


full_icosahedral_group(::Type{T}) where {T} =
    vcat(+chiral_icosahedral_group(T),
         -chiral_icosahedral_group(T))


############################################################## POLYHEDRAL ORBITS


function tetrahedron_vertices(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _one = one(T)
    irt3 = inv(sqrt(_one + _one + _one))
    return [
        V(+irt3, +irt3, +irt3),
        V(+irt3, -irt3, -irt3),
        V(-irt3, +irt3, -irt3),
        V(-irt3, -irt3, +irt3),
    ]
end


function tetrahedron_edge_centers(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _one = one(T)
    _zero = zero(T)
    return [
        V(+_one, _zero, _zero),
        V(-_one, _zero, _zero),
        V(_zero, +_one, _zero),
        V(_zero, -_one, _zero),
        V(_zero, _zero, +_one),
        V(_zero, _zero, -_one),
    ]
end


function tetrahedron_face_centers(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _one = one(T)
    irt3 = inv(sqrt(_one + _one + _one))
    return [
        V(-irt3, -irt3, -irt3),
        V(-irt3, +irt3, +irt3),
        V(+irt3, -irt3, +irt3),
        V(+irt3, +irt3, -irt3),
    ]
end


function tetrahedron_rotoinversion_centers(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _zero = zero(T)
    _one = one(T)
    irt2 = inv(sqrt(_one + _one))
    return [
        V(_zero, +irt2, +irt2),
        V(_zero, +irt2, -irt2),
        V(_zero, -irt2, +irt2),
        V(_zero, -irt2, -irt2),
        V(+irt2, _zero, +irt2),
        V(-irt2, _zero, +irt2),
        V(+irt2, _zero, -irt2),
        V(-irt2, _zero, -irt2),
        V(+irt2, +irt2, _zero),
        V(+irt2, -irt2, _zero),
        V(-irt2, +irt2, _zero),
        V(-irt2, -irt2, _zero),
    ]
end


function pyritohedron_vertices(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _one = one(T)
    irt3 = inv(sqrt(_one + _one + _one))
    return [
        V(+irt3, +irt3, +irt3),
        V(+irt3, +irt3, -irt3),
        V(+irt3, -irt3, +irt3),
        V(+irt3, -irt3, -irt3),
        V(-irt3, +irt3, +irt3),
        V(-irt3, +irt3, -irt3),
        V(-irt3, -irt3, +irt3),
        V(-irt3, -irt3, -irt3),
    ]
end


octahedron_vertices(::Type{T}) where {T} =
    tetrahedron_edge_centers(T)


octahedron_edge_centers(::Type{T}) where {T} =
    tetrahedron_rotoinversion_centers(T)


octahedron_face_centers(::Type{T}) where {T} =
    pyritohedron_vertices(T)


function icosahedron_vertices(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _zero = zero(T)
    _one = one(T)
    two = _one + _one
    five = two + two + _one
    half = inv(two)
    sqrt_five = sqrt(five)
    phi = half * (sqrt_five + _one)
    psi = half * (sqrt_five - _one)
    qdrt_five = sqrt(sqrt_five)
    a = inv(qdrt_five * sqrt(phi))
    b = inv(qdrt_five * sqrt(psi))
    return [
        V(_zero, +b, +a),
        V(_zero, -b, +a),
        V(_zero, +b, -a),
        V(_zero, -b, -a),
        V(+a, _zero, +b),
        V(+a, _zero, -b),
        V(-a, _zero, +b),
        V(-a, _zero, -b),
        V(+b, +a, _zero),
        V(-b, +a, _zero),
        V(+b, -a, _zero),
        V(-b, -a, _zero),
    ]
end


function icosahedron_edge_centers(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _zero = zero(T)
    _one = one(T)
    two = _one + _one
    four = two + two
    five = four + _one
    half = inv(two)
    quarter = inv(four)
    hphi = quarter * (sqrt(five) + _one)
    hpsi = quarter * (sqrt(five) - _one)
    return [
        V(+_one, _zero, _zero),
        V(-_one, _zero, _zero),
        V(_zero, +_one, _zero),
        V(_zero, -_one, _zero),
        V(_zero, _zero, +_one),
        V(_zero, _zero, -_one),
        V(+half, +hphi, +hpsi),
        V(+half, +hphi, -hpsi),
        V(+half, -hphi, +hpsi),
        V(+half, -hphi, -hpsi),
        V(-half, +hphi, +hpsi),
        V(-half, +hphi, -hpsi),
        V(-half, -hphi, +hpsi),
        V(-half, -hphi, -hpsi),
        V(+hpsi, +half, +hphi),
        V(-hpsi, +half, +hphi),
        V(+hpsi, +half, -hphi),
        V(-hpsi, +half, -hphi),
        V(+hpsi, -half, +hphi),
        V(-hpsi, -half, +hphi),
        V(+hpsi, -half, -hphi),
        V(-hpsi, -half, -hphi),
        V(+hphi, +hpsi, +half),
        V(+hphi, -hpsi, +half),
        V(-hphi, +hpsi, +half),
        V(-hphi, -hpsi, +half),
        V(+hphi, +hpsi, -half),
        V(+hphi, -hpsi, -half),
        V(-hphi, +hpsi, -half),
        V(-hphi, -hpsi, -half),
    ]
end


function icosahedron_face_centers(::Type{T}) where {T}
    V = SArray{Tuple{3},T,1,3}
    _zero = zero(T)
    _one = one(T)
    three = _one + _one + _one
    five = three + _one + _one
    six = three + three
    irt3 = inv(sqrt(three))
    a = sqrt((three + sqrt(five)) / six)
    b = sqrt((three - sqrt(five)) / six)
    return [
        V(_zero, +b, +a),
        V(_zero, -b, +a),
        V(_zero, +b, -a),
        V(_zero, -b, -a),
        V(+a, _zero, +b),
        V(+a, _zero, -b),
        V(-a, _zero, +b),
        V(-a, _zero, -b),
        V(+b, +a, _zero),
        V(-b, +a, _zero),
        V(+b, -a, _zero),
        V(-b, -a, _zero),
        V(+irt3, +irt3, +irt3),
        V(+irt3, +irt3, -irt3),
        V(+irt3, -irt3, +irt3),
        V(+irt3, -irt3, -irt3),
        V(-irt3, +irt3, +irt3),
        V(-irt3, +irt3, -irt3),
        V(-irt3, -irt3, +irt3),
        V(-irt3, -irt3, -irt3),
    ]
end


const POLYHEDRAL_POINT_GROUPS = [
    (chiral_tetrahedral_group, [tetrahedron_vertices,
        tetrahedron_edge_centers, tetrahedron_face_centers]),
    (full_tetrahedral_group, [
        tetrahedron_vertices, tetrahedron_edge_centers,
        tetrahedron_face_centers, tetrahedron_rotoinversion_centers]),
    (pyritohedral_group, [
        pyritohedron_vertices, tetrahedron_edge_centers]),
    (chiral_octahedral_group, [octahedron_vertices,
        octahedron_edge_centers, octahedron_face_centers]),
    (full_octahedral_group, [octahedron_vertices,
        octahedron_edge_centers, octahedron_face_centers]),
    (chiral_icosahedral_group, [icosahedron_vertices,
        icosahedron_edge_centers, icosahedron_face_centers]),
    (full_icosahedral_group, [icosahedron_vertices,
        icosahedron_edge_centers, icosahedron_face_centers]),
]


####################################################### ABSTRACT GROUP STRUCTURE


function inf_norm(xs::AbstractArray{T}) where {T}
    result = zero(T)
    @simd ivdep for x in xs
        result = max(result, abs(x))
    end
    return result
end


function multiplication_table(
        group::Vector{SArray{Tuple{N,N},T,2,M}}) where {N,T,M}
    n = length(group)
    result = zeros(Int, n, n)
    dist = zero(T)
    for i = 1 : n
        for j = 1 : n
            @inbounds d, result[i,j] = minimum(
                (inf_norm(group[i] * group[j] - group[k]), k)
                for k = 1 : n)
            dist = max(dist, d)
        end
    end
    return (result, dist)
end


function count_central_elements(mul_table::Matrix{Int})
    m, n = size(mul_table)
    @assert m == n
    return count(all(
        @inbounds mul_table[i,j] == mul_table[j,i]
        for j = 1 : n) for i = 1 : n)
end


################################################################ ORBIT STRUCTURE


function rotation_axis(mat::SArray{Tuple{3,3},T,2,9},
                       epsilon=4096*eps(T)) where {T}
    @assert inf_norm(mat' * mat - one(mat)) <= epsilon
    u, s, v = @suppress svd(mat - sign(det(mat)) * one(mat))
    @assert s[1] > epsilon
    @assert s[2] > epsilon
    @assert zero(T) <= s[3] <= epsilon
    return v[:,3]
end


function connected_components(adjacency_lists::Dict{V,Vector{V}}) where {V}
    visited = Dict{V,Bool}()
    for (v, l) in adjacency_lists
        visited[v] = false
    end
    components = Vector{V}[]
    for (v, l) in adjacency_lists
        if !visited[v]
            visited[v] = true
            current_component = [v]
            to_visit = copy(l)
            while !isempty(to_visit)
                w = pop!(to_visit)
                if !visited[w]
                    visited[w] = true
                    push!(current_component, w)
                    append!(to_visit, adjacency_lists[w])
                end
            end
            push!(components, current_component)
        end
    end
    @assert allunique(vcat(components...))
    return components
end


function degenerate_orbits(group::Vector{SArray{Tuple{3,3},T,2,9}},
                           epsilon=4096*eps(T)) where {T}
    Point = SArray{Tuple{3},T,1,3}
    points = Vector{Point}()
    for (i, g) in enumerate(group)
        try
            axis = rotation_axis(g, epsilon)
            push!(points, +axis)
            push!(points, -axis)
        catch e
            if !(e isa AssertionError)
                rethrow(e)
            end
        end
    end
    clusters = Vector{Tuple{Point,Vector{Point}}}()
    for point in points
        found = false
        for (center, cluster) in clusters
            if norm(point - center) <= epsilon
                found = true
                push!(cluster, point)
                break
            end
        end
        if !found
            push!(clusters, (point, [point]))
        end
    end
    @assert all(
        norm(p - q) <= epsilon
        for (_, cluster) in clusters
        for p in cluster for q in cluster)
    n = length(clusters)
    @assert all(
        @inbounds norm(clusters[i][1] - clusters[j][1]) > epsilon
        for i = 1 : n-1 for j = i+1 : n)
    adjacency_lists = Dict(i => Int[] for i = 1 : n)
    for (i, (center, _)) in enumerate(clusters)
        for g in group
            p = g * center
            dist, j = minimum(
                (norm(p - q), k)
                for (k, (q, _)) in enumerate(clusters))
            @assert dist <= epsilon
            if i != j
                @inbounds push!(adjacency_lists[i], j)
            end
        end
    end
    return [[@inbounds clusters[i][1] for i in comp]
            for comp in connected_components(adjacency_lists)]
end


####################################################### SYMMETRIZED RIESZ ENERGY


# Benchmarked in Julia 1.5.3 for zero allocations or exceptions.

# @benchmark symmetrized_riesz_energy(points, group, external_points) setup=(
#     points=randn(3, 10); group=chiral_tetrahedral_group(Float64);
#     external_points=SVector{3,Float64}.(eachcol(randn(3, 5))))

# view_asm(symmetrized_riesz_energy,
#     Matrix{Float64},
#     Vector{SArray{Tuple{3,3},Float64,2,9}},
#     Vector{SVector{3,Float64}})

function symmetrized_riesz_energy(
        points::AbstractMatrix{T},
        group::Vector{SArray{Tuple{N,N},T,2,M}},
        external_points::Vector{SArray{Tuple{N},T,1,N}}) where {T,N,M}
    dim, num_points = size(points)
    group_size = length(group)
    num_external_points = length(external_points)
    energy = zero(T)
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for j = 2 : group_size
            @inbounds g = group[j]
            energy += 0.5 * inv(norm(g*p - p))
        end
    end
    for i = 2 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for g in group
            gp = g * p
            for j = 1 : i-1
                @inbounds q = SVector{N,T}(view(points, 1:N, j))
                energy += inv(norm(gp - q))
            end
        end
    end
    energy *= group_size
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        for g in group
            gp = g * p
            for j = 1 : num_external_points
                @inbounds q = external_points[j]
                energy += inv(norm(gp - q))
            end
        end
    end
    return energy
end


# Benchmarked in Julia 1.5.3 for zero allocations or exceptions.

# @benchmark symmetrized_riesz_gradient!(
#     grad, points, group, external_points) setup=(
#     points=randn(3, 10); grad=similar(points);
#     group=chiral_tetrahedral_group(Float64);
#     external_points=SVector{3,Float64}.(eachcol(randn(3, 5))))

# view_asm(symmetrized_riesz_gradient!,
#     Matrix{Float64}, Matrix{Float64},
#     Vector{SArray{Tuple{3,3},Float64,2,9}},
#     Vector{SVector{3,Float64}})

function symmetrized_riesz_gradient!(
        grad::AbstractMatrix{T},
        points::AbstractMatrix{T},
        group::Vector{SArray{Tuple{N,N},T,2,M}},
        external_points::Vector{SArray{Tuple{N},T,1,N}}) where {T,N,M}
    dim, num_points = size(points)
    group_size = length(group)
    num_external_points = length(external_points)
    for i = 1 : num_points
        @inbounds p = SVector{N,T}(view(points, 1:N, i))
        force = zero(SVector{N,T})
        for j = 2 : group_size
            @inbounds r = group[j] * p - p
            force += r / norm(r)^3
        end
        for j = 1 : num_points
            if i != j
                @inbounds q = SVector{N,T}(view(points, 1:N, j))
                for g in group
                    r = g * q - p
                    force += r / norm(r)^3
                end
            end
        end
        force *= group_size
        for j = 1 : num_external_points
            @inbounds q = external_points[j]
            for g in group
                r = g * q - p
                force += r / norm(r)^3
            end
        end
        @simd ivdep for j = 1 : N
            @inbounds grad[j,i] = force[j]
        end
    end
    return grad
end


struct SymmetrizedRieszEnergyFunctor{T}
    group::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points::Vector{SArray{Tuple{3},T,1,3}}
    external_energy::T
end


struct SymmetrizedRieszGradientFunctor{T}
    group::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points::Vector{SArray{Tuple{3},T,1,3}}
end


function (sref::SymmetrizedRieszEnergyFunctor{T})(
          points::AbstractMatrix{T}) where {T}
    return sref.external_energy + symmetrized_riesz_energy(
        points, sref.group, sref.external_points)
end


function (srgf::SymmetrizedRieszGradientFunctor{T})(
          grad::AbstractMatrix{T}, points::AbstractMatrix{T}) where {T}
    symmetrized_riesz_gradient!(grad, points, srgf.group, srgf.external_points)
    constrain_riesz_gradient_sphere!(grad, points)
    return grad
end


function symmetrized_riesz_functors(
        ::Type{T}, group_function::Function,
        orbit_functions::Vector{Function}) where {T}
    group = group_function(T)::Vector{SArray{Tuple{3,3},T,2,9}}
    external_points = vcat([orbit_function(T)::Vector{SArray{Tuple{3},T,1,3}}
                            for orbit_function in orbit_functions]...)
    external_points_matrix = Matrix{T}(undef, 3, length(external_points))
    for (i, point) in enumerate(external_points)
        @simd ivdep for j = 1 : 3
            @inbounds external_points_matrix[j,i] = point[j]
        end
    end
    external_energy = riesz_energy(external_points_matrix)
    return (SymmetrizedRieszEnergyFunctor{T}(group, external_points,
                                             external_energy),
            SymmetrizedRieszGradientFunctor{T}(group, external_points))
end


end # module PCREOSymmetry
