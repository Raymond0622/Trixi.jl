using Printf
using Trixi

# All-Gauss density wave, no ECAV (hyperbolic only). polydeg = 3.
# Levels 5→8.
elixir = joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")
iterations = 4

println("="^80)
println("MortarEntropy Gauss volume/surface/mortars, no ECAV, checkerboard")
println("levels 5→8")
println("="^80)
eocs, _ = convergence_test(elixir, iterations;
                           shock_capturing = "none",
                           mortar_type = "entropy",
                           mortar_nodes = "gauss",
                           reverse_quad = "gauss",
                           checkerboard = true,
                           initial_refinement_level = 5,
                           make_plots = false)
means = Trixi.calc_mean_convergence(eocs)

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 3, expected ≈ 4)")
println("All-Gauss, no ECAV, checkerboard density wave, levels 5→8")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:l2]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:linf]...)
