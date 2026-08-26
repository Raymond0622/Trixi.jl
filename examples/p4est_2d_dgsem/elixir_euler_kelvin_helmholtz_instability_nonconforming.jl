using OrdinaryDiffEqLowStorageRK
using Trixi
using OrdinaryDiffEqSSPRK

###############################################################################
# 2D modified Sod (Toro) extruded in y, with ECAV.
# No positivity limiter, no extra NS viscosity, no SVV.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
mu() = 0.0
prandtl_number() = 0.73
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = ParabolicFormulationLocalDG()
function initial_condition_riemann1(coords, t, equations::CompressibleEulerEquations2D)
    x, y = coords

    if 0 < x < 0.5 
        if 0 < y < 0.5
            rho = 0.8;
            v1 = 0
            v2 = 0;
            p = 1
        else 
            rho = 1
            v1 = 3 / sqrt(17)
            v2 = 0
            p = 1
        end
    else 
        if 0 < y < 0.5
            rho = 1
            v1 = 0
            v2 = 3/sqrt(17)
            p = 1
        else
            rho = 17 * 0.03125
            v1 = 0
            v2 = 0;
            p = 0.4;
        end
    end

    return prim2cons(SVector(rho, v1, v2, p), equations);
end

function initial_condition_modified_sod_2d(x, t, equations)
    if x[1] < 0.3
        rho = 1.0
        v1 = 0.75
        v2 = 0.0
        p = 1.0
    else
        rho = 0.125
        v1 = 0.0
        v2 = 0.0
        p = 0.1
    end
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_modified_sod_2d
initial_condition = initial_condition_riemann1

surface_flux = FluxLaxFriedrichs(max_abs_speed)
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_flux = flux_ranocha
volume_integral=VolumeIntegralFluxDifferencing(volume_flux)
#volume_integral = VolumeIntegralWeakForm()
#solver = DGSEM(basis, surface_flux, volume_integral, MortarEntropy(basis))

volume_flux = flux_central
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(polydeg)
indicator = IndicatorEntropyCorrection(equations, basis)

volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

# volume_integral=VolumeIntegralWeakForm()
solver = DGSEM(basis, surface_flux, volume_integral, MortarEntropy(basis))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 6
# Non-periodic in x (shock tube), periodic in y (2D extrusion).
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 100_000, periodicity = (true, true))

dx = (coordinates_max[1] - coordinates_min[1]) / 2^Trixi.minimum_level(mesh.tree)
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

# Left face is subsonic inflow: prescribe the left state. Right face is outflow.
# y must be periodic on a mesh with periodicity = (false, true).
boundary_conditions_hyperbolic = (; x_neg = boundary_condition_do_nothing,
                                  x_pos = boundary_condition_do_nothing,
                                  y_neg = boundary_condition_periodic,
                                  y_pos = boundary_condition_periodic)


# semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
#                                              VDM, filter,
#                                              initial_condition, solver;
#                                              combine_rhs = Trixi.True(),
#                                              solver_parabolic = solver_parabolic,
#                                              boundary_conditions = (boundary_conditions_hyperbolic,
#                                                                     boundary_conditions_parabolic))

# semi = SemidiscretizationHyperbolic(mesh, equations,
#                                              initial_condition, solver;
#                                              boundary_conditions = (boundary_conditions_hyperbolic))

semi = SemidiscretizationHyperbolic(mesh, equations,
                                             initial_condition, solver;
                                             boundary_conditions=boundary_condition_periodic)
###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43();
            abstol = 1e-8, reltol = 1e-6, saveat = 0.01,
            ode_default_options()..., callback = callbacks);

using Plots
pd = PlotData2D(sol)
plot(getmesh(pd))
savefig("modified_sod_2d_mesh.png")

pd = PlotData2D(sol.u[end], semi)
plot(pd["rho"], clims = (0.1, 1.05))
savefig("modified_sod_2d_rho.png")

anim = @animate for k in eachindex(sol.u)
    pd = PlotData2D(sol.u[k], semi)
    plot(pd["rho"], title = "t = $(round(sol.t[k]; digits = 2))",
         clims = (0.1, 1.05))
end
gif(anim, "modified_sod_2d.gif", fps = 12)

entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
     legend = false, title = "entropy integral")
savefig("entropy_integral.png")

