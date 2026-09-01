using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Manufactured-solution Euler on a checkerboard (2:1) TreeMesh.
#
# Default mortar is `MortarL2`. Switch with `trixi_include` / `convergence_test`:
#   convergence_test(elixir, 4; mortar_type = MortarEntropy)

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.

polydeg = 3
mortar_type = MortarL2
volume_flux = flux_ranocha
mortar = mortar_type(LobattoLegendreBasis(polydeg))
solver = DGSEM(polydeg = polydeg,
               surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
               mortar = mortar)

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000, periodicity = true)

# Checkerboard refinement: every other leaf is refined so neighboring cells differ
# by one level. Every coarse–fine face is then a 2:1 mortar.
# Read the level from the tree so `convergence_test` / `trixi_include` overrides of
# `initial_refinement_level` still produce a scaled checkerboard.
 

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat=0.05,# solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
using Plots
entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, (entropy_integral .- entropy_integral[1]) ./ entropy_integral[1], xlabel = "t", ylabel = "∫ S dV / |Ω|",
     legend = false, title = "entropy integral")
savefig("entropy_integral.png")