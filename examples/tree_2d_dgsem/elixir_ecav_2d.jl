using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi
using Revise

###############################################################################
# 2D Riemann problem with ECAV on a checkerboard nonconforming TreeMesh.
# Uses MortarEntropy. Physical NS viscosity is off (`mu = 0`); no SVV.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
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

function initial_condition_kelvin_helmholtz_instability(x, t, equations)

    # A = .8
    A = 3 / 7
    rho1 = 0.5 * one(A) # recover original with A = 3/7
    rho2 = rho1 * (1 + A) / (1 - A)

    # B is a discontinuous function with value 1 for -.5 <= x <= .5 and 0 elsewhere
    slope = 15
    B = 0.5 * (tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5))

    rho = rho1 + B * (rho2 - rho1)  # rho ∈ [rho_1, rho_2]
    v1 = B - 0.5                    # v1  ∈ [-.5, .5]
    v2 = 0.1 * sin(2 * pi * x[1]) 
    # v2 = 0.1 * sin(2 * pi * x[1]) * (1 + .01 * sin(pi * x[1]) * sin(pi * x[2])) # symmetry breaking
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability
initial_condition = initial_condition_riemann1

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = FluxLaxFriedrichs(max_abs_speed)
volume_flux = flux_central

indicator_ec = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
#volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)
volume_integral = VolumeIntegralWeakForm()
volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = first)
volume_flux = flux_central
surface_flux = flux_lax_friedrichs

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral, MortarL2(basis))


coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = true)

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

# Identity filter: required by the merged constructor, unused for ECAV-only.
VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)

semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             VDM = VDM, filter = filter,
                                             ecav_choice = :ecav,
                                             combine_rhs = Trixi.True(),
                                             solver_parabolic = solver_parabolic)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
    boundary_conditions=Trixi.boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43();
            abstol = 1e-8, reltol = 1e-6, saveat=0.01,
            ode_default_options()..., callback = callbacks)

using Plots
pd = PlotData2D(sol)

plot(getmesh(pd), title = "checkerboard mesh")
savefig("mesh.png")
plot(pd["rho"], clims=(0.4, 2.0), title = "rho at t = $(round(sol.t[end]; digits = 3))")
plot!(getmesh(pd))
savefig("rho.png")

entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, entropy_integral)