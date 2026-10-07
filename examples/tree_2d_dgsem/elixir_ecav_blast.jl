using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

###############################################################################
# Weak blast wave (Hennemann & Gassner 2020, Sec. 6.3) on a checkerboard TreeMesh.
# Volume integral is `VolumeIntegralWeakForm` (same as the ECAV vortex / density-wave
# elixirs), not entropy-conservative flux differencing.
#
# Mortar / capturing switches (strings so `trixi_include` can override them):
#   mortar_type      = "entropy" | "l2"
#   mortar_nodes     = "gauss_lobatto" | "gauss"   (entropy only)
#   reverse_quad     = "gauss" | "gauss_lobatto"   (LGL entropy mortars only)
#   shock_capturing  = "ecav" | "none"
# `mortar_nodes = "gauss"` → Gauss traces/Riemann, L² reverse
# `P = M_f^{-1} I^T M_m` with lumped `M_f = W_LGL`, `M_m = W_G / 2`.
# SAT is always LGL `SurfaceIntegralWeakForm` (`f*/ω`).
# `reverse_quad` is not named `reverse_quadrature` so `trixi_include` cannot
# rewrite the `MortarEntropy(...; reverse_quadrature = ...)` keyword.

equations = CompressibleEulerEquations2D(1.4)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

# Fraction of the Hennemann–Gassner (Sec. 6.3) jump. `1` is the original weak
# blast (`ρ, |v|, p = 1.1691, 0.1882, 1.245` inside `r = 0.5`). String/scalar
# assignment so `trixi_include` can override it.
blast_strength = 0.1;
function initial_condition_weaker_blast_wave(x, t, equations)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    RealT = eltype(x)
    α = convert(RealT, blast_strength)
    rho_in = 1 + α * convert(RealT, 0.1691)
    v_in = α * convert(RealT, 0.1882)
    p_in = 1 + α * convert(RealT, 0.245)
    if r > 0.5f0
        rho = one(RealT)
        v1 = zero(RealT)
        v2 = zero(RealT)
        p = one(RealT)
    else
        rho = rho_in
        v1 = v_in * cos_phi
        v2 = v_in * sin_phi
        p = p_in
    end
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weaker_blast_wave

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

volume_flux = flux_ranocha
surface_flux = flux_ranocha
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral = VolumeIntegralWeakForm()

mortar_type = "entropy"
mortar_nodes = "gauss_lobatto"
reverse_quad = "gauss_lobatto"
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
checkerboard = false;

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
initial_refinement_level = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = true)

if checkerboard
    # Full checkerboard: hanging faces on every other cell, including periodic wraps.
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

# "ecav" → entropy-correction AV (combined hyperbolic + viscous RHS, mortar ψ
#          residual). "none" → hyperbolic flux differencing only.
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

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 300
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 300,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
cfl = 0.8
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

# `saveat` / `run_solve` are assignments so `trixi_include` can override them.
saveat = 0.05
run_solve = true

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat = saveat,
            ode_default_options()..., callback = callbacks)

entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
open("blast_entropy_integral.csv", "w") do io
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
         legend = false, title = "blast entropy integral")
    savefig("blast_entropy_integral.png")
catch e
    println("Plots unavailable (", typeof(e), "); wrote blast_entropy_integral.csv")
end


