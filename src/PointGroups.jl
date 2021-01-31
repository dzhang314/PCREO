module PointGroups

using GenericSVD: svd
using NearestNeighbors: KDTree, nn
using StaticArrays: SArray, SVector, cross, det, norm, svd
using Suppressor: @suppress

export cyclic_group, rotoreflection_group, pinwheel_group, pyramid_group,
    dihedral_group, prismatic_group, antiprismatic_group,
    chiral_tetrahedral_group, full_tetrahedral_group, pyritohedral_group,
    chiral_octahedral_group, full_octahedral_group,
    chiral_icosahedral_group, full_icosahedral_group,
    tetrahedron_vertices, tetrahedron_edge_centers, tetrahedron_face_centers,
    tetrahedron_rotoinversion_centers, pyritohedron_vertices,
    octahedron_vertices, octahedron_edge_centers, octahedron_face_centers,
    icosahedron_vertices, icosahedron_edge_centers, icosahedron_face_centers,
    POLYHEDRAL_POINT_GROUPS,
    multiplication_table, count_central_elements, degenerate_orbits,
    isometric, isometries, identify_point_group


######################################################### 3D SYMMETRY OPERATIONS


function rotation_matrix_z(::Type{T}, i::Int, n::Int) where {T}
    _zero = zero(T)
    _one = one(T)
    if i % n == 0
        return SArray{Tuple{3,3},T,2,9}(
            +_one, _zero, _zero,
            _zero, +_one, _zero,
            _zero, _zero, +_one)
    elseif iseven(n) && (i % n) == (n >> 1)
        return SArray{Tuple{3,3},T,2,9}(
            -_one, _zero, _zero,
            _zero, -_one, _zero,
            _zero, _zero, +_one)
    end
    c = cospi(T(2*i) / T(n))
    s = sinpi(T(2*i) / T(n))
    return SArray{Tuple{3,3},T,2,9}(
        c, s, _zero, -s, c, _zero, _zero, _zero, _one)
end


function rotoreflection_matrix_z(::Type{T}, i::Int, n::Int) where {T}
    _zero = zero(T)
    _one = one(T)
    if i % n == 0
        return SArray{Tuple{3,3},T,2,9}(
            +_one, _zero, _zero,
            _zero, +_one, _zero,
            _zero, _zero, -_one)
    elseif iseven(n) && (i % n) == (n >> 1)
        return SArray{Tuple{3,3},T,2,9}(
            -_one, _zero, _zero,
            _zero, -_one, _zero,
            _zero, _zero, -_one)
    end
    c = cospi(T(2*i) / T(n))
    s = sinpi(T(2*i) / T(n))
    return SArray{Tuple{3,3},T,2,9}(
        c, s, _zero, -s, c, _zero, _zero, _zero, -_one)
end


function rotation_matrix_x_pi(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    return SArray{Tuple{3,3},T,2,9}(
        +_one, _zero, _zero,
        _zero, -_one, _zero,
        _zero, _zero, -_one)
end


function reflection_matrix_y(::Type{T}) where {T}
    _zero = zero(T)
    _one = one(T)
    return SArray{Tuple{3,3},T,2,9}(
        +_one, _zero, _zero,
        _zero, -_one, _zero,
        _zero, _zero, +_one)
end


############################################################# AXIAL POINT GROUPS


cyclic_group(::Type{T}, n::Int) where {T} =
    [rotation_matrix_z(T, i, n) for i = 0 : n-1]


function rotoreflection_group(::Type{T}, n::Int) where {T}
    @assert iseven(n)
    return [(iseven(i)
        ? rotation_matrix_z(T, i, n)
        : rotoreflection_matrix_z(T, i, n))
        for i = 0 : n-1]
end


pinwheel_group(::Type{T}, n::Int) where {T} = vcat(
    [rotation_matrix_z(T, i, n) for i = 0 : n-1],
    [rotoreflection_matrix_z(T, i, n) for i = 0 : n-1])


pyramid_group(::Type{T}, n::Int) where {T} = vcat(
    [rotation_matrix_z(T, i, n) for i = 0 : n-1],
    [rotation_matrix_z(T, i, n) * reflection_matrix_y(T) for i = 0 : n-1])


dihedral_group(::Type{T}, n::Int) where {T} = vcat(
    [rotation_matrix_z(T, i, n) for i = 0 : n-1],
    [rotation_matrix_z(T, i, n) * rotation_matrix_x_pi(T) for i = 0 : n-1])


prismatic_group(::Type{T}, n::Int) where {T} = vcat(
    [rotation_matrix_z(T, i, n) for i = 0 : n-1],
    [rotation_matrix_z(T, i, n) * rotation_matrix_x_pi(T) for i = 0 : n-1],
    [rotoreflection_matrix_z(T, i, n) for i = 0 : n-1],
    [rotation_matrix_z(T, i, n) * reflection_matrix_y(T) for i = 0 : n-1])


antiprismatic_group(::Type{T}, n::Int) where {T} = vcat(
    [rotation_matrix_z(T, i, n) for i = 0 : n-1],
    [rotoreflection_matrix_z(T, i, 2*n) for i = 1 : 2 : 2*n],
    [rotation_matrix_z(T, i, n) * reflection_matrix_y(T) for i = 0 : n-1],
    [rotoreflection_matrix_z(T, i, 2*n) * reflection_matrix_y(T)
     for i = 1 : 2 : 2*n])


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
                           epsilon) where {T}
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


############################################################# FINDING ISOMETRIES


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


##################################################### POINT GROUP IDENTIFICATION


function identify_point_group(group::Vector{SArray{Tuple{3,3},T,2,9}},
                              epsilon) where {T}
    n = length(group)
    mul_table, dist = multiplication_table(group)
    @assert dist <= epsilon
    @assert (n, n) == size(mul_table)

    id_dist, id = minimum(
        (inf_norm(mat - one(mat)), i)
        for (i, mat) in enumerate(group))
    @assert id_dist <= epsilon
    @assert issorted(view(mul_table, id, :))
    @assert issorted(view(mul_table, :, id))

    is_abelian = (mul_table == mul_table')

    order_table = zeros(Int, n)
    for g = 1 : n
        acc = g
        ord = 1
        while acc != id
            acc = mul_table[acc,g]
            ord += 1
        end
        order_table[g] = ord
    end
    @assert minimum(order_table) == 1
    max_order = maximum(order_table)
    is_cyclic = (max_order == n)

    is_chiral = true
    has_inversion = false
    has_pure_reflection = false
    has_max_order_negative_element = false

    _zero = zero(T)
    _one = one(T)
    two = _one + _one
    axes = zeros(SVector{3,T}, n)

    for (i, mat) in enumerate(group)
        u, s, v = @suppress svd(mat - one(mat); full=true)
        if inf_norm(s - [_zero, _zero, _zero]) <= epsilon # identity
            @assert !signbit(det(mat))
            @assert id == i
        elseif inf_norm(s - [two, two, two]) <= epsilon # pure inversion
            @assert signbit(det(mat))
            is_chiral = false
            if order_table[i] == max_order
                has_max_order_negative_element = true
            end
            @assert !has_inversion
            has_inversion = true
        elseif inf_norm(s - [two, two, _zero]) <= epsilon # 180-deg. rotation
            @assert !signbit(det(mat))
            axes[i] = v[:,3]
        elseif inf_norm(s - [two, _zero, _zero]) <= epsilon # pure reflection
            @assert signbit(det(mat))
            is_chiral = false
            if order_table[i] == max_order
                has_max_order_negative_element = true
            end
            has_pure_reflection = true
            axes[i] = v[:,1]
        elseif abs(s[1] - s[2]) <= epsilon # general rotation
            @assert !signbit(det(mat))
            @assert s[3] <= epsilon
            axes[i] = v[:,3]
        else # general rotoinversion
            @assert signbit(det(mat))
            is_chiral = false
            if order_table[i] == max_order
                has_max_order_negative_element = true
            end
            @assert abs(s[1] - two) <= epsilon
            @assert abs(s[2] - s[3]) <= epsilon
            axes[i] = v[:,1]
        end
    end

    if n == 1
        return "C_1"
    end

    if n == 2
        if is_chiral
            return "C_2"
        else
            return has_inversion ? "C_i" : "C_s"
        end
    end

    if n == 4
        if is_cyclic
            return is_chiral ? "C_4" : "S_4"
        elseif is_chiral
            return "D_2"
        else
            return has_inversion ? "C_2h" : "C_2v"
        end
    end

    if n == 8
        if is_cyclic
            return is_chiral ? "C_8" : "S_8"
        elseif is_chiral
            return "D_4"
        elseif is_abelian
            return (max_order == 2) ? "D_2h" : "C_4h"
        else
            return has_max_order_negative_element ? "D_2d" : "C_4v"
        end
    end

    if is_cyclic
        @assert is_abelian
        if is_chiral
            return "C_$n"
        elseif has_pure_reflection
            @assert iseven(n)
            @assert isodd(n >> 1)
            return "C_$(n >> 1)h"
        else
            @assert iseven(n)
            return "S_$n"
        end
    end

    if is_abelian
        @assert iseven(n)
        @assert iseven(n >> 1)
        return "C_$(n >> 1)h"
    end

    is_axial = (max_order >= 3) && all(
        abs(sqrt(abs(axes[i]' * axes[j])) - _one) <= epsilon
        for i = 1 : n, j = 1 : n
        if (order_table[i] == max_order &&
            order_table[j] == max_order))

    if is_axial
        @assert iseven(n)
        if is_chiral
            return "D_$(n >> 1)"
        elseif !has_max_order_negative_element
            return "C_$(n >> 1)v"
        else
            @assert iseven(n >> 1)
            if xor(iseven(n >> 2), has_inversion)
                return "D_$(n >> 2)d"
            else
                return "D_$(n >> 2)h"
            end
        end
    else
        if n == 12
            return "T"
        elseif n == 48
            return "O_h"
        elseif n == 60
            return "I"
        elseif n == 120
            return "I_h"
        else
            @assert n == 24
            if is_chiral
                return "O"
            else
                return has_inversion ? "T_h" : "T_d"
            end
        end
    end
end


################################################################################

end # module PointGroups
