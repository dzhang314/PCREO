module PCREOSymmetry

using DZOptimization: norm
using DZOptimization.ExampleFunctions: riesz_energy,
    constrain_riesz_gradient_sphere!
using GenericSVD: svd
using NearestNeighbors: KDTree, nn
using StaticArrays: SArray, SVector, cross, det, svd
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
    symmetrized_riesz_functors, isometric, isometries


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


################################################### DEGENERATE POLYHEDRAL ORBITS


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
    indices, dists = nn(
        KDTree(SVector{M,T}.(group)),
        [SVector{M,T}(g * h) for h in group for g in group])
    return (reshape(indices, n, n), inf_norm(dists))
end


function count_central_elements(mul_table::Matrix{Int})
    m, n = size(mul_table)
    @assert m == n
    return count(all(
        @inbounds mul_table[i,j] == mul_table[j,i]
        for j = 1 : n) for i = 1 : n)
end


################################################################ ORBIT STRUCTURE


function rotation_axis(mat::SArray{Tuple{3,3},T,2,9}, epsilon) where {T}
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


################################################################################


function labeled_distances(points::Vector{SVector{N,T}}) where {T,N}
    return [(sqrt(sum((points[i] - points[j]).^2)), i, j)
            for j = 2 : length(points)
            for i = 1 : j-1]
end


function bucket_by_first(items::Vector{T}, epsilon) where {T}
    result = Vector{T}[]
    if length(items) == 0
        return result
    end
    push!(result, [items[1]])
    for i = 2 : length(items)
        if abs(items[i][1] - result[end][end][1]) <= epsilon
            push!(result[end], items[i])
        else
            push!(result, [items[i]])
        end
    end
    for bucket in result
        @assert abs(bucket[end][1] - bucket[1][1]) <=epsilon
    end
    return result
end


middle(x::AbstractVector) = x[(length(x) + 1) >> 1]


function candidate_isometries(
        a_points::Vector{SVector{3,T}},
        b_points::Vector{SVector{3,T}}, epsilon) where {T}
    _one = one(T)
    two = _one + _one
    four = two + two
    one_fourth = inv(four)
    seven_fourths = (four + two + _one) * one_fourth
    result = SArray{Tuple{3,3},T,2,9}[]
    a_buckets = bucket_by_first(sort!(labeled_distances(a_points)), epsilon)
    b_buckets = bucket_by_first(sort!(labeled_distances(b_points)), epsilon)
    if length.(a_buckets) != length.(b_buckets)
        return result
    end
    a_midpoints = [middle(bucket)[1] for bucket in a_buckets]
    b_midpoints = [middle(bucket)[1] for bucket in b_buckets]
    for (a, b) in zip(a_midpoints, b_midpoints)
        if !(abs(a - b) <= epsilon)
            return result
        end
    end
    selected_bucket = minimum(
        (length(bucket), abs(one(T) - middle(bucket)[1]), i)
        for (i, bucket) in enumerate(a_buckets)
        if one_fourth <= middle(bucket)[1] <= seven_fourths)[3]
    _, i, j = middle(a_buckets[selected_bucket])
    u1, v1 = a_points[i], a_points[j]
    pos_mat = inv(hcat(u1, v1, cross(u1, v1)))
    neg_mat = inv(hcat(u1, v1, cross(v1, u1)))
    for (_, k, l) in b_buckets[selected_bucket]
        u2, v2 = b_points[k], b_points[l]
        fwd_mat = hcat(u2, v2, cross(u2, v2))
        rev_mat = hcat(v2, u2, cross(v2, u2))
        push!(result, fwd_mat * pos_mat)
        push!(result, fwd_mat * neg_mat)
        push!(result, rev_mat * pos_mat)
        push!(result, rev_mat * neg_mat)
    end
    for mat in result
        @assert inf_norm(mat' * mat - one(mat)) <= epsilon
    end
    return result
end


function isometric(
        a_points::Vector{SVector{3,T}},
        b_points::Vector{SVector{3,T}}, epsilon) where {T}
    b_tree = KDTree(b_points)
    for mat in candidate_isometries(a_points, b_points, epsilon)
        indices, distances = nn(b_tree, [mat * p for p in a_points])
        if allunique(indices) && (inf_norm(distances) <= epsilon)
            return true
        end
    end
    return false
end


function isometries(
        a_points::Vector{SVector{3,T}},
        b_points::Vector{SVector{3,T}}, epsilon) where {T}
    result = SArray{Tuple{3,3},T,2,9}[]
    b_tree = KDTree(b_points)
    for mat in candidate_isometries(a_points, b_points, epsilon)
        indices, distances = nn(b_tree, [mat * p for p in a_points])
        if allunique(indices) && (inf_norm(distances) <= epsilon)
            push!(result, mat)
        end
    end
    return result
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
