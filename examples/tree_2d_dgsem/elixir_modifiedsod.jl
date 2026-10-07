using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# Toro modified Sod (Sec. 6.4) extruded uniformly in `y` on a checkerboard
# TreeMesh. Entropy-conservative flux differencing: `flux_ranocha` in the
# volume *and* on the surface. No ECAV, no adaptive / shock-capturing volume
# integral.
#
# Mortar switches (strings so `trixi_include` can override them):
#   mortar_type   = "entropy" | "l2"
#   mortar_nodes  = "gauss_lobatto" | "gauss"   (entropy only)
#   reverse_quad  = "gauss" | "gauss_lobatto"   (LGL entropy mortars only)
# `reverse_quad` is not named `reverse_quadrature` so `trixi_include` cannot
# rewrite the `MortarEntropy(...; reverse_quadrature = ...)` keyword.

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_modified_sod_2d(x, t, equations)

Toro modified Sod (Sec. 6.4) extruded uniformly in `y`: left sonic rarefaction,
contact, and shock. Discontinuity at `x = 0.3`.
"""
function initial_condition_modified_sod_2d(x, t, equations)
    if x[1] < 0.3
        return prim2cons(SVector(1.0, 0.75, 0.0, 1.0), equations)
    else
        return prim2cons(SVector(0.125, 0.0, 0.0, 0.1), equations)
    end
end
initial_condition = initial_condition_modified_sod_2d

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = FluxLaxFriedrichs(max_abs_speed)
volume_flux = flux_central

# "ecav"     → entropy-correction AV + standard DG volume integral
# "adaptive" → VolumeIntegralAdaptive (entropy-correction FV switch), hyperbolic only
# String so `trixi_include` / `convergence_test` can override it.
shock_capturing = "adaptive"
indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                            VolumeIntegralWeakForm(),
                                            VolumeIntegralPureLGLFiniteVolume(surface_flux))

mortar_type = "entropy"
mortar_nodes = "gauss"
reverse_quad = "gauss"
if mortar_type == "l2"
    mortar = MortarL2(basis)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
elseif mortar_nodes == "gauss"
    mortar = MortarEntropy(basis; nodes = :gauss)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
else
    mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                           reverse_quadrature = Symbol(reverse_quad))
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
end
solver = DGSEM(basis, surface_integral, volume_integral, mortar)

# Mesh switch (so `trixi_include` can override it):
#   checkerboard = true  → 2:1 hanging faces (nmortars > 0)
#   checkerboard = false → uniform conforming TreeMesh
checkerboard = true;

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 5
periodicity = (false, true)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = periodicity)

if checkerboard
    # Full checkerboard: hanging faces on every other cell.
    n_base = 2^initial_refinement_level
    dx = (coordinates_max[1] - coordinates_min[1]) / n_base
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
end

# Dirichlet on the left is only valid while the rarefaction stays off `x = 0`.
boundary_conditions = (;
                       x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
stepsize_callback = StepsizeCallback(cfl = 1.0)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution)

# `saveat` / `run_solve` are assignments so `trixi_include` can override them.
saveat = 0.05
run_solve = true
if run_solve
    try
        sol = solve(ode, SSPRK43();
                    abstol = 1e-8, reltol = 1e-6,
                    saveat = saveat,
                    ode_default_options()..., callback = callbacks)
    catch e
        @warn "SSPRK43 solve aborted" exception = (e, catch_backtrace())
        sol = nothing
    end
end

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
#             dt = 1.0,
#             saveat = range(first(tspan), last(tspan); length = 21),
#             ode_default_options()..., callback = callbacks)

if run_solve && sol !== nothing
    entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
else
    entropy_integral = Float64[]
end

make_plots = true
if make_plots && run_solve && sol !== nothing
    using Plots
    pd = PlotData2D(sol)
    plot(getmesh(pd), title = "modified Sod mesh")
    savefig("sod_mesh.png")
    plot(pd["rho"], title = "rho at t = $(round(sol.t[end]; digits = 3))")
    plot!(getmesh(pd))
    savefig("sod_rho.png")
    plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
         legend = false, title = "entropy integral")
    savefig("sod_entropy_integral.png")
end
