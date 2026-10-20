using Printf
using Trixi

# Checkerboard density wave, all-Gauss, polydeg = 3 (expected EOC ≈ 4).
# Levels 3→7.
elixir = joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")
iterations = 5

println("="^80)
println("MortarEntropy Gauss volume/surface/mortars (Gauss SAT), checkerboard")
println("="^80)
eocs, _ = convergence_test(elixir, iterations;
                           shock_capturing = "ecav",
                           mortar_type = "entropy",
                           mortar_nodes = "gauss_lobatto",
                           reverse_quad = "gauss_lobatto",
                           checkerboard = false,
                           initial_refinement_level = 3,
                           make_plots = false)
means = Trixi.calc_mean_convergence(eocs)

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 3, expected ≈ 4)")
println("All-Gauss ECAV + weak volume, checkerboard density wave, levels 3→7")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:l2]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:linf]...)
