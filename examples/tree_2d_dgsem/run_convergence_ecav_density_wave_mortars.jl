using Printf
using Trixi

# Checkerboard density wave, polydeg = 3 (expected EOC ≈ 4).
# Same mortar / SAT wiring as `elixir_ecav_blast.jl`: ECAV, weak volume, LGL SAT.
# LGL entropy uses `reverse_quad = "gauss_lobatto"` (blast default). Gauss entropy
# uses Gauss-node reverse `P = W_G^{-1} I^T (W_G/2)` and Gauss SAT on hanging faces.
elixir = joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")
iterations = 3

cases = (
    (:l2, "l2", "gauss_lobatto", "gauss_lobatto",
     "MortarL2: LGL traces/SAT, conserved interp"),
    (:entropy_lgl, "entropy", "gauss_lobatto", "gauss_lobatto",
     "MortarEntropy: LGL traces/SAT, entropy interp, L² reverse LGL masses, copy small-face fluxes"),
    (:entropy_gauss, "entropy", "gauss", "gauss",
     "MortarEntropy: Gauss traces/Riemann (M_f = W_G), Gauss SAT"),
)

means = Dict{Symbol, Any}()  
for (key, mortar_type, nodes, reverse_quad, title) in cases
    println("="^80)
    println(title)
    println("="^80)
    eocs, _ = convergence_test(elixir, iterations;
                               shock_capturing = "ecav",
                               mortar_type = mortar_type,
                               mortar_nodes = nodes,
                               reverse_quad = reverse_quad)
    means[key] = Trixi.calc_mean_convergence(eocs)
end

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 3, expected ≈ 4), density wave, ECAV + weak volume")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:l2]...)
@printf("L2  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:l2]...)
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:l2]...)
@printf("L∞  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:linf]...)
@printf("L∞  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:linf]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:linf]...)
