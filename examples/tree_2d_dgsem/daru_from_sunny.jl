using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi
using StartUpDG
using Plots
using LinearAlgebra
using Revise

function changetoStartUpDG(du, cache)
    
    n_nodes_2d = nnodes(dg)^ndims(mesh)
    n_elements = nelements(dg, cache)
    nvars = nvariables(equations)
    uEltype = Float64
    x = reshape(view(cache.elements.node_coordinates, 1, :, :, :), n_nodes_2d,
                n_elements)
    y = reshape(view(cache.elements.node_coordinates, 2, :, :, :), n_nodes_2d,
                n_elements)
    du_trixi_remapped = similar(u0)
    du_extracted = StructArray{SVector{nvars, uEltype}}(ntuple(_ -> similar(x,
                                                                           (n_nodes_2d,
                                                                            n_elements)),
                                                            nvars))
    for element in eachelement(dg, cache)
        sk = 1
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(du, equations, dg, i, j, element)
            du_extracted[sk, element] = u_node
            sk += 1
        end
    end  
    remapped_x = similar(md.x);
    remapped_y = similar(md.y)

    skip = n_elements
    l = Int(sqrt(n_elements))
    idxskip = Int(sqrt(n_elements)/2)
    mymap  = Vector{Int64}(undef, n_elements)

    function fillmap(x, y, idx, j, k) 
        if (div(skip, 2^j) == 0) 
            #@show idx, x, y
            mymap[idx] = (y - 1) * l + x
            return
        end
        pat = div(skip, 2^j)
        cat = div(idxskip, 2^k)
        fillmap(x + cat, y, idx + pat, j + 2, k + 1);
        fillmap(x, y + cat, idx + 2* pat, j + 2, k + 1);
        fillmap(x + cat, y + cat, idx + 3* pat, j + 2, k + 1);
        fillmap(x, y, idx, j + 2, k + 1);
    end

    fillmap(1, 1, 1, 2, 0)
    for i in range(1, n_elements)
        #@show mymap[i]
        du_trixi_remapped[:, mymap[i]] = du_extracted[:, i]
        remapped_x[:, mymap[i]] = x[:, i]
        remapped_y[:, mymap[i]] = y[:, i]
    end
    return du_trixi_remapped, remapped_x, remapped_y
end

prandtl_number() = 0.73

function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    inicenter = SVector(0.25, 0.25)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    r0 = 0.25f0
    rho = r > r0 ? one(1e-1) : RealT(1.1691)
    v1 = r > r0 ? zero(RealT) : RealT(0.1882) * cos_phi
    v2 = r > r0 ? zero(RealT) : RealT(0.1882) * sin_phi
    # p = r > 0.25f0 ? RealT(1.0E-2) : RealT(1.245)
    p = r > r0 ? RealT(1e-1) : RealT(1.245)

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

function initial_condition_daru(x, t, equations)
    RealT = eltype(x)
    v1 = zero(RealT)
    v2 = zero(RealT)

    rho_rr = 120.0
    # rho_rr = 60.0

    rho = x[1] > 0.5f0 ? 1.2 : rho_rr
    p = x[1] > 0.5f0 ? 1.2 / equations.gamma : rho_rr / equations.gamma
    # rho = x[1] > 0.5f0 ? 1.2 : 24.0
    # p = x[1] > 0.5f0 ? 1.2 / equations.gamma : 24.0 / equations.gamma
    # rho = 59.4 * tanh(-25*(x[1] - 0.5)) + 60.6
    if abs(x[1] - 0.5f0) < 1e3 * eps()
        rho = 0.5 * (1.2 + rho_rr)
    end
    p = rho / equations.gamma

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
tspan = (0.0, 1.0)
initial_condition = initial_condition_daru
periodicity = (false, false)
@inline mu() = 5e-3 # Re = 200
# @inline mu() = 2e-3  # Re = 500
# @inline mu() = 0.0013333333333333333 # Re = 750
#@inline mu() = 1e-3 # Re = 1000

# coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
# coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
# # initial_condition = initial_condition_blast_wave
# # tspan = (0.0, 1.5)
# initial_condition = initial_condition_kelvin_helmholtz_instability
# tspan = (0.0, 5.0)
# periodicity = (true, true)
# # periodicity = (false, false)
# mu() = 1e-6

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
# Conservative variables for Schlieren: use full Local-DG gradient of rho.
equations_schlieren = CompressibleNavierStokesDiffusion2D(equations, mu = 0.0,
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

dg = DGSEM(polydeg = 1, surface_flux = FluxLaxFriedrichs(max_abs_speed),
           volume_integral = VolumeIntegralWeakForm())
#         #    volume_integral = VolumeIntegralFluxDifferencing(flux_shima_etal))

# surface_flux = FluxLaxFriedrichs(max_abs_speed)
# basis = LobattoLegendreBasis(3)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max = 0.25,
#                                          alpha_min = 0.001,
#                                          alpha_smooth = true,
#                                          variable = density_pressure)
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg = flux_shima_etal,
#                                                 #  volume_flux_dg = flux_central,
#                                                  volume_flux_fv = surface_flux)           
# dg = DGSEM(basis, surface_flux, volume_integral)

# Create a uniformly refined mesh with periodic boundaries
initial_refinement_level = 7
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = periodicity, n_cells_max = 400_000)

# BC types
boundary_condition_noslip_wall = BoundaryConditionNavierStokesWall(NoSlip((x, t, equations_parabolic) -> (0.0,
                                                                                                          0.0)),
                                                                   Adiabatic((x, t, equations_parabolic) -> 0.0))

# define inviscid boundary conditions
boundary_conditions_hyperbolic = (; x_neg = boundary_condition_slip_wall,
                                  x_pos = boundary_condition_slip_wall,
                                  y_neg = boundary_condition_slip_wall,
                                  y_pos = boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_noslip_wall,
                                 x_pos = boundary_condition_noslip_wall,
                                 y_neg = boundary_condition_noslip_wall,
                                 y_pos = boundary_condition_noslip_wall)

solver_parabolic = ParabolicFormulationLocalDG()

if all(mesh.tree.periodicity .== true)
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg;
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic)
    # semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                              initial_condition, dg;
    #                                              solver_parabolic = solver_parabolic)

else
    N = Trixi.polydeg(dg)
    filter = [((i - 1)/N)^2 for i in 1:N+1]
    #filter = [(i - 1) < 5 ? 0 : 1 for i in 1:N+1]
    #filter = [1 for i in 1:N+1]
    nodes = Trixi.get_nodes(dg.basis)
    VDM = vandermonde(Line(), N, nodes)
    ecav_choice = :ecav
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic), 
                                            initial_condition, dg;
                                                VDM, filter, ecav_choice,
                                            combine_rhs = Trixi.True(),
                                            solver_parabolic = solver_parabolic,
                                            boundary_conditions = (boundary_conditions_hyperbolic,
                                                                boundary_conditions_parabolic));

    #     semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                                 initial_condition, dg;
    #                                                 solver_parabolic = solver_parabolic,
    #                                                 boundary_conditions = (boundary_conditions_hyperbolic, 
    #                                                                         boundary_conditions_parabolic))
end

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
callbacks = CallbackSet(summary_callback, alive_callback)
# analysis_interval = 1000
# analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
# callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback) #, amr_callback)

###############################################################################
# run the simulation

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-6, 1.0e-6),
                                                     variables = (Trixi.density, pressure))
# solver = SSPRK43(stage_limiter!, stage_limiter!)
solver = SSPRK43()

sol = solve(ode, solver; abstol = 1e-6, reltol = 1e-4, # dt = 1e-8,
            saveat = 0.01, ode_default_options()..., callback = callbacks)

using JLD2
@save "DaruTenaudRe1000_polydeg_$(Trixi.polydeg(dg.basis))_elements_$(2^initial_refinement_level).jld2" sol semi

# @save "DaruTenaudRe1000_polydeg_$(Trixi.polydeg(dg.basis))_elements_$(2^initial_refinement_level)_shock_capturing_amax_p5.jld2" sol
function schlieren_normalize!(schlieren; beta = 10.0)
    schlieren_min, schlieren_max = extrema(schlieren)
    if schlieren_max > schlieren_min
        @. schlieren = exp(-beta * (schlieren - schlieren_min) /
                           (schlieren_max - schlieren_min))
    else
        fill!(schlieren, one(eltype(schlieren)))
    end
    return schlieren
end

"""
    density_schlieren_dg(u_ode, semi; beta=10.0, t=0.0)

Compute a density Schlieren field on `TreeMesh` using Trixi's Local-DG gradient
(`calc_gradient!`), including volume, interface, and boundary terms. This differs
from a bare `derivative_matrix` contraction, which only applies the reference
element operator without DG surface lifts.
"""
function density_schlieren_dg(u_ode, semi::Union{SemidiscretizationArtificialViscosity,
                                                  SemidiscretizationHyperbolicParabolic};
                                beta = 10.0, t = 0.0)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array_native(u_ode, semi)

    (; parabolic_container) = semi.cache_parabolic
    (; u_transformed, gradients) = parabolic_container
    gradients_x, gradients_y = gradients

    Trixi.transform_variables!(u_transformed, u, mesh, equations_schlieren, dg, cache)
    Trixi.calc_gradient!(gradients, u_transformed, t, mesh, equations_schlieren,
                         semi.boundary_conditions_parabolic, dg, semi.solver_parabolic,
                         cache)

    n_nodes = nnodes(dg)
    schlieren = zeros(n_nodes, n_nodes, nelements(dg, cache))

    for element in Trixi.eachelement(dg, cache),
        j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)

        drho_dx = gradients_x[1, i, j, element]
        drho_dy = gradients_y[1, i, j, element]
        schlieren[i, j, element] = sqrt(drho_dx^2 + drho_dy^2)
    end

    return schlieren_normalize!(schlieren; beta)
end

gr() # good backend for GIFs

data = load("DaruTenaudRe1000_polydeg_3_elements_512.jld2");
sol = data["sol"];
anim = @animate for k in eachindex(sol.u)
    schlieren = density_schlieren_dg(sol.u[end], semi; beta = 10.0, t = sol.t[end])
    pd = ScalarPlotData2D(schlieren, semi; variable_name = "density Schlieren (LDG)")
    plot(pd;
         title = "density Schlieren, t = $(round(sol.t[end], digits = 3))",
         aspect_ratio = :equal,
         ylims=(0.0, 0.5),
         clims = (0.0, 1.0),
         color = :grays)
end

gif(anim, "test.gif", fps = 10)

data = load("DaruTenaudRe1000_polydeg_3_elements_256.jld2");
sol = data["sol"];
N = 1
Ny = 512;
Nx = Ny;
rd = RefElemData(Quad(), SBP(), N) 

problem = DaruTenaud
(VX, VY), EToV = uniform_mesh(rd.element_type, Nx, Ny)
domain_change_x = problem.domain_change_x
domain_change_y = problem.domain_change_y
VX = domain_change_x.(VX, 0, equations)
VY = domain_change_y.(VY, 0, equations)
md = MeshData((VX, VY), EToV, rd; 
            is_periodic=false);
u0 = problem.initial_condition.(SVector.(md.x, md.y), 0, equations)

u, _, _ = changetoStartUpDG(Trixi.wrap_array(parent(sol.u[1]), semi), semi.cache);

@gif for u in sol.u
    u, _, _ = changetoStartUpDG(Trixi.wrap_array(parent(u), semi), semi.cache);
    rho = getindex.(parent(u), 1)
    vx = getindex.(parent(u), 2) ./ rho;
    vy = getindex.(parent(u), 3) ./ rho;
    vnorm = sqrt.(vx.^2 + vy.^2)
    scatter(vec(md.x), vec(md.y), 
            zcolor=vec(vnorm), 
            msw=0, ms=4, legend=false, ratio=1)
end fps=10

# StartUpDG cross-check: compare LDG Schlieren above with explicit StartUpDG gradient.
function compute_gradient!(dudx, dudy, u, cache)
    (; md, rd) = cache
    dudx .= md.rxJ .* (rd.Dr * u) + md.sxJ .* (rd.Ds * u)
    dudy .= md.ryJ .* (rd.Dr * u) + md.syJ .* (rd.Ds * u)
    uf = rd.Vf * u
    uP = uf[md.mapP]
    dudx .+= rd.LIFT * ((0.5 * (uP - uf)) .* md.nxJ)
    dudy .+= rd.LIFT * ((0.5 * (uP - uf)) .* md.nyJ)
    dudx ./= md.J
    dudy ./= md.J
    return nothing
end

dudx = similar(md.x)
dudy = similar(md.x)
schlieren_startupdg = similar(md.x)
@gif for (k, u_ode) in enumerate(sol.u)
    u_nodes, _, _ = changetoStartUpDG(Trixi.wrap_array(u_ode, semi), semi.cache)
    rho = getindex.(u_nodes, 1)
    compute_gradient!(dudx, dudy, rho, (; md, rd, equations))
    g = sqrt.(dudx .^ 2 + dudy .^ 2)
    gmax, gmin = extrema(g)
    schlieren_startupdg .= exp.(-10 * (g .- gmin) / (gmax - gmin))
    scatter(vec(md.x), vec(md.y);
            zcolor = vec(schlieren_startupdg),
            msw = 0, ms = 4, legend = false, aspect_ratio = :equal,
            title = "StartUpDG Schlieren, t = $(round(sol.t[k], digits = 3))")
end fps = 5

coef= semi.cache.artificial_viscosity.coefficients;
res = repeat(coef, inner = (Trixi.polydeg(dg) + 1)^2 * 4)

coef_t, _, _ = changetoStartUpDG(Trixi.wrap_array(res , semi), semi.cache)

scatter(vec(md.x), vec(md.y), zcolor = vec(log.(getindex.(coef_t, 1) .+ 100*eps())), 
        msw = 0, ms=4, legend=false, aspect_ratio=:equal, ylims=(0, 0.5), xlims=(0.0, 1.0))