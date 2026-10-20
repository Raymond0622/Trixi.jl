using Printf
using Trixi

# Checkerboard isentropic vortex, all-Gauss, polydeg = 4 (expected EOC ≈ 5).
# Levels 3→7. Same `convergence_test` overrides as
# `run_convergence_ecav_density_wave_mortars.jl`.
elixir = joinpath(@__DIR__, "elixir_ecav_2d_isentropic_vortex.jl")
iterations = 5

println("="^80)
println("MortarEntropy Gauss volume/surface/mortars (Gauss SAT), checkerboard")
println("="^80)
eocs, _ = convergence_test(elixir, iterations;
                           shock_capturing = "ecav",
                           mortar_type = "entropy",
                           mortar_nodes = "gauss",
                           reverse_quad = "gauss",
                           checkerboard = true,
                           initial_refinement_level = 3,
                           make_plots = false)
means = Trixi.calc_mean_convergence(eocs)

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 4, expected ≈ 5)")
println("All-Gauss, no ECAV, weak volume, checkerboard isentropic vortex, levels 3→7")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:l2]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:linf]...)
