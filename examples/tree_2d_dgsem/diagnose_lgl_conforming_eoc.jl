using Printf
using Trixi

eocs, _ = convergence_test(joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl"), 3;
                           mortar_type = "entropy",
                           mortar_nodes = "gauss",
                           reverse_quad = "gauss",
                           shock_capturing = "none",
                           make_plots = false,
                           checkerboard = false,
                           tspan = (0.0, 0.1),
                           saveat = 0.1,
                           initial_refinement_level = 3)
mean = Trixi.calc_mean_convergence(eocs)
@printf("Gauss volume, none, conforming, N=3  L2=%.3f  L∞=%.3f\n",
        mean[:l2][1], mean[:linf][1])
