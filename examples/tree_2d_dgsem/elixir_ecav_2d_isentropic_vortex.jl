using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: I
using Trixi

###############################################################################
# Isentropic vortex with ECAV on a checkerboard nonconforming TreeMesh.
# Toggle `mortar_type` (`MortarL2` / `MortarEntropy`) via `convergence_test`.
# Physical NS viscosity is off (`mu = 0`); no SVV.
# The exact solution is time-dependent so `convergence_test` can use a short tspan.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = ParabolicFormulationBassiRebay1()

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case of
- Chi-Wang Shu (1997)
  Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
  Schemes for Hyperbolic Conservation Laws
  [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # Domain [-10, 10]^2; vortex advects with velocity (1, 1).
    RealT = eltype(x)
    inicenter = SVector(zero(RealT), zero(RealT))
    iniamplitude = 5
    rho = one(RealT)
    v1 = one(RealT)
    v2 = one(RealT)
    vel = SVector(v1, v2)
    p = convert(RealT, 25)
    rt = p / rho
    domain_length = convert(RealT, 20)

    cent = inicenter + vel * t
    cent = x - cent
    # Periodic wrap so the exact solution is valid at any t
    cent = SVector(cent[1] - domain_length * round(cent[1] / domain_length),
                   cent[2] - domain_length * round(cent[2] / domain_length))
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude / (2 * convert(RealT, pi)) * exp(0.5f0 * (1 - r2))
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2
    rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_isentropic_vortex

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = FluxLaxFriedrichs(max_abs_speed)
volume_flux = flux_ranocha
# Default mortar is `MortarL2`. Switch with `trixi_include` / `convergence_test`:
#   convergence_test(elixir, 4; mortar_type = MortarEntropy)
mortar_type = MortarEntropy
mortar = mortar_type(basis)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
#volume_integral = VolumeIntegralWeakForm();
solver = DGSEM(basis, surface_flux, volume_integral,
               mortar)

coordinates_min = (-10.0, -10.0)
coordinates_max = (10.0, 10.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 400_000, periodicity = true)

# Checkerboard refinement: every other leaf is refined so neighboring cells differ
# by one level. Every coarse–fine face is then a 2:1 mortar.
# Read the level from the tree so `convergence_test` / `trixi_include` overrides of
# `initial_refinement_level` still produce a scaled checkerboard.
level = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
dx = (coordinates_max[1] - coordinates_min[1]) / 2^level
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

VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)

semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             VDM = VDM, filter = filter,
                                             ecav_choice = :ecav,
                                             combine_rhs = Trixi.True(),
                                             solver_parabolic = solver_parabolic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 0.8)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat = 0.05,
            ode_default_options()..., callback = callbacks)

# using Plots
# pd = PlotData2D(sol)
# plot(getmesh(pd), title = "mesh")
# savefig("mesh.png")
# plot(pd["rho"], title = "rho at t = $(round(sol.t[end]; digits = 3))")
# plot!(getmesh(pd))
# savefig("rho.png")

# anim = @animate for k in eachindex(sol.u)
#     pd = PlotData2D(sol.u[k], semi)
#     plot(pd["rho"], clims=(0.96, 1.0), title = "rho, t = $(round(sol.t[k]; digits = 2))")
#     plot!(getmesh(pd))
# end
# gif(anim, "elixir_ecav_2d_isentropic_vortex.gif", fps = 10)

# Domain-integrated entropy at each saved solution time (same quadrature as AnalysisCallback)
entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
     legend = false, title = "entropy integral")
savefig("entropy_integral.png")