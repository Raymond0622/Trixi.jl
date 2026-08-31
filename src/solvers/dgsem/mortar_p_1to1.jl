# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    MortarP1to1(basis_own, basis_other)

1:1 p-nonconforming mortar operators for two adjacent faces that share geometry
but not quadrature.

Both traces are interpolated from their SBP/LGL nodes onto a composite
Gauss–Lobatto mortar with `max(n_own, n_other)` nodes (the finer face's LGL
rule). The Riemann flux is evaluated there, then L²-projected back onto
`basis_own` with lumped SBP (LGL) mass weights:

```math
f^{\\mathrm{own}}_i = \\sum_k V_{k i}\\, \\frac{w^{\\mathrm{mortar}}_k}{w^{\\mathrm{own}}_i}\\, f^{\\mathrm{mortar}}_k
```

This preserves the discrete surface integral,
`∑ᵢ wᵢ fᵢ = ∑ₖ wₖ fₖ`, so two sides that share the same mortar flux
remain globally conservative even when `n_own ≠ n_other`.

Used by [`BoundaryConditionCoupledPMortar`](@ref).
"""
struct MortarP1to1{RealT <: Real, NodesM, WeightsM, VOwn, VOther, POwn}
    nodes_mortar::NodesM
    weights_mortar::WeightsM
    vandermonde_own::VOwn     # n_mortar × n_own
    vandermonde_other::VOther # n_mortar × n_other
    project_own::POwn         # n_own × n_mortar
end

function MortarP1to1(basis_own::AbstractBasisSBP, basis_other::AbstractBasisSBP)
    RealT = promote_type(real(basis_own), real(basis_other))
    n_own = nnodes(basis_own)
    n_other = nnodes(basis_other)
    n_mortar = max(n_own, n_other)

    nodes_mortar, weights_mortar = gauss_lobatto_nodes_weights(n_mortar, RealT)
    vandermonde_own = polynomial_interpolation_matrix(basis_own.nodes, nodes_mortar)
    vandermonde_other = polynomial_interpolation_matrix(basis_other.nodes, nodes_mortar)

    # Lumped-mass L² projection mortar → own face quadrature.
    # P[i, k] = V[k, i] * w_mortar[k] / w_own[i]
    project_own = zeros(RealT, n_own, n_mortar)
    inverse_weights_own = basis_own.inverse_weights
    for i in 1:n_own
        for k in 1:n_mortar
            project_own[i, k] = vandermonde_own[k, i] * weights_mortar[k] *
                                inverse_weights_own[i]
        end
    end

    return MortarP1to1{RealT, typeof(nodes_mortar), typeof(weights_mortar),
                       typeof(vandermonde_own), typeof(vandermonde_other),
                       typeof(project_own)}(nodes_mortar, weights_mortar,
                                            vandermonde_own, vandermonde_other,
                                            project_own)
end

function Base.show(io::IO, mortar::MortarP1to1)
    @nospecialize mortar # reduce precompilation time

    print(io, "MortarP1to1(n_own=", size(mortar.project_own, 1),
          ", n_other=", size(mortar.vandermonde_other, 2),
          ", n_mortar=", length(mortar.nodes_mortar), ")")
    return nothing
end

@inline function nnodes_mortar(mortar::MortarP1to1)
    return length(mortar.nodes_mortar)
end

@inline function nnodes_own(mortar::MortarP1to1)
    return size(mortar.project_own, 1)
end

@inline function nnodes_other(mortar::MortarP1to1)
    return size(mortar.vandermonde_other, 2)
end

# Interpolate a face vector (nvars × n_src) onto nvars × n_dst using V (n_dst × n_src).
@inline function multiply_dimensionwise_face!(u_dst, vandermonde, u_src)
    nvars = size(u_src, 1)
    n_dst = size(vandermonde, 1)
    n_src = size(vandermonde, 2)
    for i in 1:n_dst
        for v in 1:nvars
            acc = zero(eltype(u_dst))
            for j in 1:n_src
                acc = acc + vandermonde[i, j] * u_src[v, j]
            end
            u_dst[v, i] = acc
        end
    end
    return u_dst
end
end # @muladd
