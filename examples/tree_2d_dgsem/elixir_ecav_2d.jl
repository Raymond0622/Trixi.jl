using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: I
using Trixi
using Revise

###############################################################################
# 2D Riemann problem with ECAV. Checkerboard 2:1 refinement inside each
# quadrant; faces on the quadrant boundaries stay conforming.

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

    if x < 0.5
        if y < 0.5
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
        if y < 0.5
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

function Trixi.compute_coefficients!(backend::Nothing, u,
                                     func::typeof(initial_condition_riemann1), t,
                                     mesh::TreeMesh{2}, equations, dg::DG, cache)
        @show "hi"
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            x_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg,
                                           i, j, element)
            if i == 1 # left boundary node
                x_node = SVector(nextfloat(x_node[1]), x_node[2])
            elseif i == nnodes(dg) # right boundary node
                x_node = SVector(prevfloat(x_node[1]), x_node[2])
            end
            if j == 1 # bottom boundary node
                x_node = SVector(x_node[1], nextfloat(x_node[2]))
            elseif j == nnodes(dg) # top boundary node
                x_node = SVector(x_node[1], prevfloat(x_node[2]))
            end

            u_node = func(x_node, t, equations)
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end
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
#volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)

#indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                         alpha_max = 1.0,
#                                         alpha_min = 0.001,
#                                         alpha_smooth = true,
#                                         variable = first)
#volume_flux = flux_central
#surface_flux = flux_lax_friedrichs

#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg = volume_flux,
#                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral, MortarEntropy(basis))


coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = true)

# Checkerboard inside each quadrant. Cells that touch a quadrant boundary stay
# coarse so those faces are conforming (same level on both sides):
#   x = 0.5 (ix = n/2-1 and n/2), y = 0.5 (iy = n/2-1 and n/2),
#   and the periodic wraps x = 0/1, y = 0/1 (which also glue different quadrants).
n_base = 2^initial_refinement_level
dx = (coordinates_max[1] - coordinates_min[1]) / n_base
ix_jump = (n_base ÷ 2 - 1, n_base ÷ 2)  # faces at x = 0.5
iy_jump = (n_base ÷ 2 - 1, n_base ÷ 2)  # faces at y = 0.5
cells_to_refine = Int[]
for cell_id in Trixi.leaf_cells(mesh.tree)
    x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
    ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
    iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
    on_quadrant_boundary = ix in (0, n_base - 1, ix_jump...) ||
                           iy in (0, n_base - 1, iy_jump...)
    on_quadrant_boundary = false  # full checkerboard: hanging faces on the slips
    if iseven(ix + iy) && !on_quadrant_boundary
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

# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
#     boundary_conditions=Trixi.boundary_condition_periodic)

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
stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution)

sol = solve(ode, SSPRK43(); abstol=1e-8, reltol=1e-6,
            saveat = 0.05,
            ode_default_options()..., callback = callbacks)

using Plots
pd = PlotData2D(sol)
plot(getmesh(pd), title = "mesh")
savefig("mesh.png")
plot(pd["rho"], title = "rho at t = $(round(sol.t[end]; digits = 3))")
plot!(getmesh(pd))
savefig("rho.png")

u0 = Trixi.wrap_array(ode.u0, mesh, equations, solver, semi.cache)
all(all(all.(u0[v, :, :, e] .== u0[v, 1, 1, e] for e in Trixi.eachelement(solver, semi.cache)))
    for v in Trixi.eachvariable(equations))
###############################################################################
# rhs_combined! from the IC: the first du is the KH seed on the slips.

u_ode = copy(ode.u0)
du_ode = similar(u_ode)

@unpack mesh, equations, boundary_conditions, source_terms = semi
@unpack equations_parabolic, boundary_conditions_parabolic = semi
@unpack solver, solver_parabolic, cache, cache_parabolic = semi
(; equations_artificial_viscosity) = semi.artificial_viscosity

u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# Face nodes on x = x0 (vary y) or y = y0 (vary x). `side` is -1 if the element
# is left/below the face, +1 if right/above.
function gather_slip_nodes(u, cache, dg, equations; x0 = 0.5, dim = 1, atol = 1e-12)
    s = Float64[]
    vtau = Float64[]
    cons_v = Float64[]
    side = Int[]
    v_index = dim == 1 ? 3 : 2  # x=0.5 slip → v2; y=0.5 slip → v1
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            xy = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg,
                                       i, j, element)
            if abs(xy[dim] - x0) > atol
                continue
            end
            face_index = dim == 1 ? i : j
            u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
            prim = cons2prim(u_node, equations)
            push!(s, xy[3 - dim])
            push!(vtau, prim[v_index])
            push!(cons_v, u_node[v_index])
            push!(side, face_index == 1 ? 1 : -1)
        end
    end
    p = sortperm(s)
    return s[p], vtau[p], cons_v[p], side[p]
end

function demean_by_segment(s, vals; cut = 0.5)
    out = copy(vals)
    for mask in (s .< cut, s .>= cut)
        if any(mask)
            out[mask] .-= sum(vals[mask]) / count(mask)
        end
    end
    return out
end

dt = 1.0e-4
n_steps = 3

using Plots

let t = 0.0
    # First residual from the IC — this is the mesh-induced v_τ(y) seed.
    Trixi.rhs_combined!(du, u, t, mesh,
                        equations, equations_parabolic,
                        equations_artificial_viscosity,
                        boundary_conditions, boundary_conditions_parabolic,
                        source_terms,
                        solver, solver_parabolic, cache, cache_parabolic)

    y, _, du_rho_v2, side_x = gather_slip_nodes(du, cache, solver, equations;
                                                x0 = 0.5, dim = 1)
    x, _, du_rho_v1, side_y = gather_slip_nodes(du, cache, solver, equations;
                                                x0 = 0.5, dim = 2)
    du_v2_pert = demean_by_segment(y, du_rho_v2)
    du_v1_pert = demean_by_segment(x, du_rho_v1)
    println("x=0.5  rms(du ρv₂ − segment mean) = ",
            sqrt(sum(abs2, du_v2_pert) / length(du_v2_pert)))
    println("y=0.5  rms(du ρv₁ − segment mean) = ",
            sqrt(sum(abs2, du_v1_pert) / length(du_v1_pert)))

    left = side_x .== -1
    right = side_x .== 1
    below = side_y .== -1
    above = side_y .== 1

    p_mesh = plot(getmesh(PlotData2D(u_ode, semi)); title = "checkerboard")
    p_x = scatter(y[left], du_v2_pert[left]; xlabel = "y", ylabel = "du(ρv₂) − mean",
                  label = "left of x=0.5", ms = 3, title = "seed on SW–SE slip")
    scatter!(p_x, y[right], du_v2_pert[right]; label = "right of x=0.5", ms = 3)
    p_y = scatter(x[below], du_v1_pert[below]; xlabel = "x", ylabel = "du(ρv₁) − mean",
                  label = "below y=0.5", ms = 3, title = "seed on SW–NW slip")
    scatter!(p_y, x[above], du_v1_pert[above]; label = "above y=0.5", ms = 3)
    plot(p_mesh, p_x, p_y; layout = (1, 3), size = (1500, 450))
    savefig("perturbation.png")

    for step in 1:n_steps
        Trixi.rhs_combined!(du, u, t, mesh,
                            equations, equations_parabolic,
                            equations_artificial_viscosity,
                            boundary_conditions, boundary_conditions_parabolic,
                            source_terms,
                            solver, solver_parabolic, cache, cache_parabolic)
        @. u_ode = u_ode + dt * du_ode
        t += dt
        println("step = $step, t = $t, max|du| = $(maximum(abs, du_ode))")
    end

    y_u, v2, _, side_u = gather_slip_nodes(u, cache, solver, equations;
                                           x0 = 0.5, dim = 1)
    v2_pert = demean_by_segment(y_u, v2)
    left_u = side_u .== -1
    right_u = side_u .== 1
    println("after Euler, rms(v₂ − segment mean) on x=0.5 = ",
            sqrt(sum(abs2, v2_pert) / length(v2_pert)))

    pd = PlotData2D(u_ode, semi; solution_variables = cons2prim)
    p_rho = plot(pd["rho"]; clims = (0.4, 2.0),
                 title = "rho at t = $(round(t; digits = 6))")
    plot!(p_rho, getmesh(pd))
    p_v2 = scatter(y_u[left_u], v2_pert[left_u]; xlabel = "y",
                   ylabel = "v₂ − segment mean", label = "left of x=0.5", ms = 3,
                   title = "v₂ wrinkle after Euler")
    scatter!(p_v2, y_u[right_u], v2_pert[right_u]; label = "right of x=0.5", ms = 3)
    plot(p_rho, p_v2; layout = (1, 2), size = (1100, 450))
    savefig("rho.png")
end