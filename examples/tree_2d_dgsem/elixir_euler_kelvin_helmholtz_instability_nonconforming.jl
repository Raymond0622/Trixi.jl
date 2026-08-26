using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
# Kelvin-Helmholtz instability on a nonconforming TreeMesh using entropy mortars
# and entropy-correction artificial viscosity

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
mu() = 0.0
prandtl_number() = 0.71
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = ParabolicFormulationLocalDG()

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    RealT = eltype(x)
    slope = 15
    B = tanh(slope * x[2] + 7.5f0) - tanh(slope * x[2] - 7.5f0)
    rho = 0.5f0 + 0.75f0 * B
    v1 = 0.5f0 * (B - 1)
    v2 = convert(RealT, 0.1) * sinpi(2 * x[1])
    p = 1
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

initial_condition = initial_condition_kelvin_helmholtz_instability

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral, MortarEntropy(basis))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 100_000, periodicity = true)

# Checkerboard refinement: every other leaf is refined so neighboring cells differ
# by one level. Every coarse–fine face is then a 2:1 mortar.
dx = (coordinates_max[1] - coordinates_min[1]) / 2^initial_refinement_level
cells_to_refine = Int[]
for cell_id in Trixi.leaf_cells(mesh.tree)
    x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
    ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
    iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
    if iseven(ix + iy)
        push!(cells_to_refine, cell_id)
    end
end
Trixi.refine!(mesh.tree, cells_to_refine)

N = polydeg
nodes = Trixi.get_nodes(basis)
VDM, _ = Trixi.vandermonde_legendre(nodes, N)
filter = [((i - 1) / N)^2 for i in 1:(N + 1)]

semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                             VDM, filter,
                                             initial_condition, solver;
                                             combine_rhs = Trixi.True(),
                                             solver_parabolic = solver_parabolic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 20,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat = 0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

using Plots
pd = PlotData2D(sol)
plot(getmesh(pd))
savefig("kelvin_helmholtz_mesh.png")

anim = @animate for k in eachindex(sol.u)
    pd = PlotData2D(sol.u[k], semi)
    plot(pd["rho"], title = "t = $(round(sol.t[k]; digits = 2))",
         clims = (0.5, 2.0))
end
gif(anim, "kelvin_helmholtz.gif", fps = 12)

# Domain-integrated entropy at each saved solution time (same quadrature as AnalysisCallback)
entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
     legend = false, title = "entropy integral")
savefig("entropy_integral.png")
