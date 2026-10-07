# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This diagram shows what is meant by "lower", "upper", and "large":
#      +1   +1
#       |    |
# upper |    |
#       |    |
#      -1    |
#            | large
#      +1    |
#       |    |
# lower |    |
#       |    |
#      -1   -1
#
# That is, we are only concerned with 2:1 subdivision of a surface/element.
# Calculate forward projection matrix for discrete L2 projection from large to upper
#
# Note: This is actually an interpolation.
function calc_forward_upper(n_nodes, RealT = Float64)
    calc_forward_upper(n_nodes, Val(:gauss_lobatto), RealT)
end

function calc_forward_upper(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[j, i] = poly[i]
        end
    end

    return operator
end

# Gauss-node interpolation from the large *LGL* face onto the upper small face.
# `I[j, i] = ℓ_i^{LGL}((η_j^G + 1) / 2)`.
function calc_forward_upper(n_nodes, ::Val{:gauss}, RealT = Float64)
    return mortar_interpolation_lgl_to_half_gauss(n_nodes, +1, RealT)
end

# Calculate forward projection matrix for discrete L2 projection from large to lower
#
# Note: This is actually an interpolation.
function calc_forward_lower(n_nodes, RealT = Float64)
    calc_forward_lower(n_nodes, Val(:gauss_lobatto), RealT)
end

function calc_forward_lower(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] - 1), nodes, wbary)
        for i in 1:n_nodes
            operator[j, i] = poly[i]
        end
    end

    return operator
end

function calc_forward_lower(n_nodes, ::Val{:gauss}, RealT = Float64)
    return mortar_interpolation_lgl_to_half_gauss(n_nodes, -1, RealT)
end

# Interpolation from large-face LGL nodes to Gauss nodes on one 2:1 half.
# `sign_half = +1` → upper (`ξ = (η+1)/2`), `-1` → lower (`ξ = (η-1)/2`).
function mortar_interpolation_lgl_to_half_gauss(n_nodes, sign_half, RealT = Float64)
    lgl_nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss_nodes, _ = gauss_nodes_weights(n_nodes, RealT)
    mapped = similar(gauss_nodes)
    for j in eachindex(gauss_nodes)
        mapped[j] = 0.5f0 * (gauss_nodes[j] + sign_half)
    end
    return polynomial_interpolation_matrix(lgl_nodes, mapped)
end

# Reverse L² in Gauss space: `P = M_f^{-1} I^T M_m` with `I` large Gauss →
# small Gauss half, `M_f = W_G`, `M_m = W_G / 2`.
function mortar_l2_reverse_gauss_mass(n_nodes, sign_half, RealT = Float64)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    gauss_wbary = barycentric_weights(gauss_nodes)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (gauss_nodes[j] + sign_half),
                                                  gauss_nodes, gauss_wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * gauss_weights[j] / gauss_weights[i]
        end
    end
    return operator
end

function gauss_lobatto_vandermondes(n_nodes, RealT = Float64)
    gauss_nodes, _ = gauss_nodes_weights(n_nodes, RealT)
    lobatto_nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
    lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)
    return gauss2lobatto, lobatto2gauss
end

# MortarL2 reverse: LGL `f*` → Gauss, L² with `M_f = W_G`, then back to LGL SAT.
function calc_reverse_upper(n_nodes, ::Val{:gauss}, RealT = Float64)
    gauss2lobatto, lobatto2gauss = gauss_lobatto_vandermondes(n_nodes, RealT)
    return gauss2lobatto * mortar_l2_reverse_gauss_mass(n_nodes, +1, RealT) *
           lobatto2gauss
end

# Gauss-node mortar reverse: Gauss `f*` stays at Gauss, L² with `M_f = W_G`.
function calc_reverse_upper(n_nodes, ::Val{:gauss_nodes}, RealT = Float64)
    return mortar_l2_reverse_gauss_mass(n_nodes, +1, RealT)
end

function calc_reverse_lower(n_nodes, ::Val{:gauss}, RealT = Float64)
    gauss2lobatto, lobatto2gauss = gauss_lobatto_vandermondes(n_nodes, RealT)
    return gauss2lobatto * mortar_l2_reverse_gauss_mass(n_nodes, -1, RealT) *
           lobatto2gauss
end

function calc_reverse_lower(n_nodes, ::Val{:gauss_nodes}, RealT = Float64)
    return mortar_l2_reverse_gauss_mass(n_nodes, -1, RealT)
end

# `P = M_f^{-1} I^T M_m` with lumped `M_f = W_LGL`, `M_m = W_G / 2`.
# Maps small-face Gauss `f*` onto large-face LGL nodes for LGL SAT.
function mortar_l2_reverse_lgl_mass_from_gauss(n_nodes, sign_half, RealT = Float64)
    I_map = mortar_interpolation_lgl_to_half_gauss(n_nodes, sign_half, RealT)
    _, lgl_weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    _, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    operator = zeros(RealT, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        operator[i, j] = 0.5f0 * I_map[j, i] * gauss_weights[j] / lgl_weights[i]
    end
    return operator
end

# Reverse L² from LGL mortar nodes onto the large LGL face:
#   P = M_f^{-1} I^T M_m  with lumped `M_f = W_LGL`, `M_m = W_LGL / 2`.
function calc_reverse_upper(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)
    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end

# Calculate reverse projection matrix for discrete L2 projection from lower to large (Gauss-Lobatto
# version)
function calc_reverse_lower(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] - 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end

# Face mass that turns LGL-nodal `f*` into the Gauss-quadrature SAT:
#   Q = W_LGL^{-1} I_{L→G}^T W_G I_{L→G}
# so `du_face -= inverse_weights[endpoint] * (Q f*)`.
function gauss_lgl_face_mass(basis)
    RealT = real(basis)
    n = nnodes(basis)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n, RealT)
    I_lg = polynomial_interpolation_matrix(basis.nodes, gauss_nodes)
    Q = zeros(RealT, n, n)
    inv_w = inv.(basis.weights)
    for l in 1:n, k in 1:n
        acc = zero(RealT)
        for j in 1:n
            acc += I_lg[j, l] * gauss_weights[j] * I_lg[j, k]
        end
        Q[l, k] = inv_w[l] * acc
    end
    return Q
end

# Lift Gauss-nodal `f*` against LGL face basis with Gauss quadrature:
#   L = I_{L→G}^T W_G
# `I_{L→G}[k, l] = ℓ_l^{LGL}(η_k^G)`. SAT then applies lumped `M^{-1}`.
function gauss_face_lift_from_gauss_nodes(basis)
    RealT = real(basis)
    n = nnodes(basis)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n, RealT)
    I_lg = polynomial_interpolation_matrix(basis.nodes, gauss_nodes)
    L = zeros(RealT, n, n)
    for l in 1:n, k in 1:n
        L[l, k] = I_lg[k, l] * gauss_weights[k]
    end
    return L
end

function calc_identity_matrix(n_nodes, RealT = Float64)
    operator = zeros(RealT, n_nodes, n_nodes)
    for i in 1:n_nodes
        operator[i, i] = one(RealT)
    end
    return operator
end

# Composite Gauss quadrature of the LGL interpolant on the upper small face,
# tested against large-face LGL Lagrange polynomials, then divided by LGL
# weights so the result is an equivalent nodal flux for the LGL SAT.
#   R = W_LGL^{-1} I_map^T (W_G / 2) I_{L→G}
#   I_map[j, i] = ℓ_i^{LGL}((η_j^G + 1) / 2)
function calc_reverse_upper(n_nodes, ::Val{:gauss_quad}, RealT = Float64)
    lgl_nodes, lgl_weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    wbary_lgl = barycentric_weights(lgl_nodes)
    I_lgl_to_gauss = polynomial_interpolation_matrix(lgl_nodes, gauss_nodes, wbary_lgl)

    I_map = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (gauss_nodes[j] + 1),
                                                  lgl_nodes, wbary_lgl)
        for i in 1:n_nodes
            I_map[j, i] = poly[i]
        end
    end

    operator = zeros(RealT, n_nodes, n_nodes)
    for i in 1:n_nodes, k in 1:n_nodes
        acc = zero(RealT)
        for j in 1:n_nodes
            acc += I_map[j, i] * (gauss_weights[j] * 0.5f0) * I_lgl_to_gauss[j, k]
        end
        operator[i, k] = acc / lgl_weights[i]
    end
    return operator
end

function calc_reverse_lower(n_nodes, ::Val{:gauss_quad}, RealT = Float64)
    lgl_nodes, lgl_weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    wbary_lgl = barycentric_weights(lgl_nodes)
    I_lgl_to_gauss = polynomial_interpolation_matrix(lgl_nodes, gauss_nodes, wbary_lgl)

    I_map = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (gauss_nodes[j] - 1),
                                                  lgl_nodes, wbary_lgl)
        for i in 1:n_nodes
            I_map[j, i] = poly[i]
        end
    end

    operator = zeros(RealT, n_nodes, n_nodes)
    for i in 1:n_nodes, k in 1:n_nodes
        acc = zero(RealT)
        for j in 1:n_nodes
            acc += I_map[j, i] * (gauss_weights[j] * 0.5f0) * I_lgl_to_gauss[j, k]
        end
        operator[i, k] = acc / lgl_weights[i]
    end
    return operator
end
end # @muladd
