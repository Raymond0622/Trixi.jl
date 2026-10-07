using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

###############################################################################
# Density wave with ECAV on a checkerboard TreeMesh.
# Volume integral: `volume_form = "fluxdiff"` → split-form
# `VolumeIntegralFluxDifferencing(flux_ranocha)`; `"weak"` → `VolumeIntegralWeakForm`
# (blast-style). Physical NS viscosity is off (`mu = 0`). The exact solution is
# time-dependent so `convergence_test` can use a short tspan.
#
# Mortar / capturing switches (strings so `trixi_include` / `convergence_test`
# can override them):
#   mortar_type      = "entropy" | "l2"
#   mortar_nodes     = "gauss_lobatto" | "gauss"   (entropy only)
#   reverse_quad     = "gauss" | "gauss_lobatto"   (LGL entropy mortars only)
#   shock_capturing  = "ecav" | "none"
# `mortar_nodes = "gauss"` → Gauss traces/Riemann, L² reverse
# `P = M_f^{-1} I^T M_m` with lumped `M_f = W_LGL`, `M_m = W_G / 2`.
# SAT is always LGL `SurfaceIntegralWeakForm` (`f*/ω`).
# `reverse_quad` is not named `reverse_quadrature` so `trixi_include` cannot
# rewrite the `MortarEntropy(...; reverse_quadrature = ...)` keyword.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

# Trixi's stock wave uses amplitude 0.98 (`rho` down to 0.02). Entropy mortar
# traces then fail in ECAV `entropy2cons` (`(-V5)^γ` with V5 > 0).
function initial_condition_density_wave_ecav(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, 0.5) * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end
initial_condition = initial_condition_density_wave_ecav

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_flux = flux_ranocha
surface_flux = flux_ranocha
# `volume_form` is a string so `trixi_include` / `convergence_test` can override it.
volume_form = "weak"
if volume_form == "fluxdiff"
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
else
    volume_integral = VolumeIntegralWeakForm()
end
 
mortar_type = "mortar_entropy"
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

# Mesh switch (so `trixi_include` / `convergence_test` can override it):
#   checkerboard = true  → 2:1 hanging faces (nmortars > 0)
#   checkerboard = false → uniform conforming TreeMesh
checkerboard = true;

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = true)

if checkerboard
    # Read the level from the tree so `convergence_test` overrides of
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
end

# "ecav" → entropy-correction AV (combined hyperbolic + viscous RHS, mortar ψ
#          residual). "none" → hyperbolic only.
shock_capturing = "ecav"
if shock_capturing == "ecav"
    # Identity filter: required by the constructor, unused for ECAV-only.
    VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
    filter = ones(polydeg + 1)
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 VDM = VDM, filter = filter,
                                                 ecav_choice = :ecav,
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
else
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
end

###############################################################################
# ODE solvers, callbacks etc.

# Short tspan: the wave is exact at any t, and t = 2 hits near-vacuum
# oscillations (`rho = 1 + 0.98 sin(...)`) that break ECAV `entropy2cons`.
tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
save_analysis = false
analysis_output_directory = "out"
analysis_filename = "analysis.dat"
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,),
                                     save_analysis = save_analysis,
                                     output_directory = analysis_output_directory,
                                     analysis_filename = analysis_filename)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
cfl = 0.8
stepsize_callback = StepsizeCallback(cfl = cfl)
time_integrator = "ssprk43"
if time_integrator == "ssprk43"
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)
else
    callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                            stepsize_callback)
end

###############################################################################
# run the simulation

# `dt` is overwritten by `StepsizeCallback`. `time_integrator` / `saveat` are
# assignments so `trixi_include` can override them.
saveat = 0.01
if time_integrator == "ssprk43"
    sol = solve(ode, SSPRK43();
                abstol = 1.0e-6, reltol = 1.0e-4, saveat = saveat,
                ode_default_options()..., callback = callbacks)
else
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1.0, saveat = saveat,
                ode_default_options()..., callback = callbacks)
end

entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
open("entropy_integral.csv", "w") do io
    println(io, "t,entropy")
    for (t, S) in zip(sol.t, entropy_integral)
        println(io, t, ",", S)
    end
end
println("entropy integral  S(0) = ", entropy_integral[1],
        "  S(end) = ", entropy_integral[end],
        "  ΔS = ", entropy_integral[end] - entropy_integral[1])
try
    using Plots
    plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
         legend = false, title = "entropy integral")
    savefig("entropy_integral.png")
catch e
    println("Plots unavailable (", typeof(e), "); wrote entropy_integral.csv")
end
