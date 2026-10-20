using LinearAlgebra: I, dot
using Printf
using Trixi
using Revise

# Isolate the ECAV viscous operator from `rhs_combined!` in
# `dg_2d_artificial_viscosity.jl`, with coefficient left at 1 (skip
# `calc_ecav_coefficients!`). Same calls, same equation objects:
#   transform_variables!(..., equations_parabolic)          # cons → entropy
#   calc_gradient!(..., equations_parabolic, ...)           # BR1 ∇w
#   calc_parabolic_fluxes!(..., equations_artificial_viscosity, ...)
#       # g = κ (∂u/∂w) ∇w, κ = 1
#   skip accum_viscous_fluxes!                              # NS μ = 0, not ECAV
#   calc_divergence!(..., equations_parabolic, ...)
#   apply_jacobian_parabolic! then negate                   # combined-RHS −1/J
#
# Directional split: after AV fluxes, keep only g_x or only g_y, then diverge.
# Cubic polynomial: (∂u/∂w)∇w = ∇u, so the density slot is ρ_x / ρ_xx.
# ρ = 5 + x³ + x²y + xy² + y³  (offset 5 so ρ > 0 on [-1,1]²).

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
eqs_av = Trixi.default_artificial_viscosity(equations)

function initial_condition_density_wave_ecav(x, t, equations)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    # ρ = 5 + Σ_{i=1}^{4} x^{i-1} y^{4-i} = 5 + y³ + x y² + x² y + x³
    rho = 1 + 0.9 * sinpi(2 * (x[1] + x[2]))
    # for i in 1:4
    #     rho = rho + x[1]^(i - 1) * x[2]^(4 - i)
    # end
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end

checkerboard = true;
mortar_type = "entropy"
initial_refinement_level = 2;
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_ranocha
volume_integral = VolumeIntegralWeakForm()
if mortar_type == "l2"
    mortar = MortarL2(basis)
else
    mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                            reverse_quadrature = :gauss_lobatto)
end
surface_integral = SurfaceIntegralWeakForm(surface_flux)
solver = DGSEM(basis, surface_integral, volume_integral, mortar)
solver_parabolic = Trixi.ParabolicFormulationLocalDG()

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max;
                initial_refinement_level = 2, periodicity = true)
if checkerboard
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

VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)
semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                        initial_condition_density_wave_ecav,
                        solver;
                        VDM = VDM, filter = filter,
                        ecav_choice = :ecav,
                        combine_rhs = Trixi.True(),
                        equations_artificial_viscosity = eqs_av,
                        solver_parabolic = solver_parabolic,
                        boundary_conditions = (boundary_condition_periodic,
                                            boundary_condition_periodic))
u_ode = Trixi.compute_coefficients(0.0, semi)
u = Trixi.wrap_array(u_ode, semi)
du_xx = Trixi.wrap_array_native(similar(u_ode), semi)
du_yy = Trixi.wrap_array_native(similar(u_ode), semi) 

Trixi.prolong2mortars!(semi.cache, u, mesh, equations,
    mortar, solver)
# semi.cache.mortars.u_upper[:, :, 3, 15]   
# Trixi.get_surface_node_vars(semi.cache.mortars.u_upper,
#     equations, solver, 3, 15)
mortars = semi.cache.mortars
mortar_id = 2
large_element = mortars.neighbor_ids[3, mortar_id]
upper_element = mortars.neighbor_ids[2, mortar_id]
lower_element = mortars.neighbor_ids[1, mortar_id]
orientation = mortars.orientations[mortar_id]
large_sides = mortars.large_sides[mortar_id]

# for mortar_id in Trixi.eachmortar(solver, semi.cache)
    

function fill_exact!(dest, analytic, semi)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    x = cache.elements.node_coordinates
    for element in Trixi.eachelement(dg, cache)
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            dest[i, j, element] = analytic(x[1, i, j, element], x[2, i, j, element])
        end
    end
    return dest
end

function reset_surface_cache!(cache)
    fill!(cache.elements.surface_flux_values, 0)
    fill!(cache.interfaces.u, 0)
    return nothing
end

# One second-derivative component only: rebuild viscous fluxes from ∇w,
# keep g_x or g_y, drop the other, then one `calc_divergence!` into a fresh `du`.
function d2rho_one_direction!(du, direction, gradients, u_transformed, u, semi, eqs_av)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av, dg, cache)
    fx, fy = flux_parabolic
    if direction == 1
        fill!(fy, 0)
    else
        fill!(fx, 0)
    end
    reset_surface_cache!(cache)
    # `Trixi.set_zero!` uses `@threaded` over a PtrArray and SIGSEGVs on large
    # MortarL2 checkerboard meshes under Julia 1.12.
    du .= 0
    Trixi.calc_divergence!(du, flux_parabolic, u, mesh, equations_parabolic,
                           semi.boundary_conditions_parabolic, dg,
                           semi.solver_parabolic, cache, 0.0)
    Trixi.apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    return nothing
end


mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi);
u_ode = Trixi.compute_coefficients(0.0, semi)
u = Trixi.wrap_array(u_ode, semi)

# Trixi.get_node_vars(u, equations, dg,
###############################################################################
# Lagrange basis on Gauss volume nodes (same ℓ_i that build `boundary_interpolation`)
"""
    lagrange_basis_polynomials(ξ, nodes)
Evaluate every Lagrange basis polynomial associated with `nodes` at `ξ`:
```math
ℓ_i(ξ) = ∏_{j ≠ i} (ξ - ξ_j) / (ξ_i - ξ_j)
```
`ℓ_i(ξ_k) = δ_{ik}`. Trixi's `boundary_interpolation[:, 1]` is this vector at
`ξ = -1`, column 2 at `ξ = +1`.
"""
function lagrange_basis_polynomials(ξ, nodes)
    n = length(nodes)
    ℓ = zeros(eltype(ξ), n)
    for i in 1:n
        v = one(ξ)
        for j in 1:n
            if j != i
                v *= (ξ - nodes[j]) / (nodes[i] - nodes[j])
            end
        end
        ℓ[i] = v
     end
    return ℓ
end
using Plots
function plot_gauss_lagrange_basis(polydeg; nplot = 401)
    basis = GaussLegendreBasis(polydeg)
    nodes = collect(Trixi.get_nodes(basis))
    n = length(nodes)
    L = basis.boundary_interpolation
    ξ = range(-1, 1; length = nplot)
    ℓ = zeros(nplot, n)
    for (k, x) in enumerate(ξ)
        ℓ[k, :] = lagrange_basis_polynomials(x, nodes)
    end
    plt = plot(ξ, ℓ;
               xlabel = "ξ",
               ylabel = "ℓᵢ(ξ)",
               title = "Lagrange basis, Gauss volume nodes (polydeg = $polydeg)",
               legend = :outertopright,
               label = permutedims(["ℓ_$i" for i in 1:n]),
               linewidth = 2)
    hline!([0, 1]; color = :gray, linestyle = :dot, label = "")
    vline!([-1, 1]; color = :gray, linestyle = :dash, label = "")
    scatter!(nodes, ones(n);
             marker = :circle, color = :black, label = "ℓᵢ(ξᵢ) = 1")
    scatter!(fill(-1, n), L[:, 1];
             marker = :star, color = :red, label = "L(ξ = -1)")
    scatter!(fill(1, n), L[:, 2];
             marker = :star, color = :blue, label = "L(ξ = +1)")
    return plt
end

plot_gauss_lagrange_basis(3)

"""
    interpolate_lagrange(ξ, u_nodal, nodes)

Nodal interpolant ``p(ξ) = ∑_i u_i ℓ_i(ξ)`` on `nodes`.
"""
function interpolate_lagrange(ξ, u_nodal, nodes)
    return dot(u_nodal, lagrange_basis_polynomials(ξ, nodes))
end

"""
    plot_gauss_interpolant(polydeg, u_fun; nplot = 401)

Sample `u_fun` at Gauss volume nodes and plot the Lagrange interpolant.
The endpoint values of that sum equal `L' * u_nodal`, i.e. the two columns of
`boundary_interpolation` dotted with the nodal vector.
"""
function plot_gauss_interpolant(polydeg, u_fun = ξ -> sinpi(ξ); nplot = 401)
    basis = GaussLegendreBasis(polydeg)
    nodes = collect(Trixi.get_nodes(basis))
    L = basis.boundary_interpolation
    u_nodal = u_fun.(nodes)

    ξ = range(-1, 1; length = nplot)
    p = [interpolate_lagrange(x, u_nodal, nodes) for x in ξ]
    p_from_L = L' * u_nodal
    p_from_sum = (interpolate_lagrange(-1.0, u_nodal, nodes),
                  interpolate_lagrange(1.0, u_nodal, nodes))

    println("u_nodal          = ", u_nodal)
    println("L' * u_nodal     = ", p_from_L)
    println("∑ u_i ℓ_i(-1,+1) = ", p_from_sum)
    @assert p_from_L ≈ collect(p_from_sum)

    plt = plot(ξ, u_fun.(ξ);
               xlabel = "ξ",
               ylabel = "u",
               title = "Gauss interpolant vs L' u (polydeg = $polydeg)",
               label = "u(ξ)",
               linestyle = :dash,
               color = :gray,
               legend = :outertopright)
    plot!(ξ, p; linewidth = 2, color = :black, label = "p(ξ) = ∑ uᵢ ℓᵢ(ξ)")
    scatter!(nodes, u_nodal; marker = :circle, color = :black, label = "u at Gauss nodes")
    scatter!([-1.0, 1.0], p_from_L;
             marker = :star, markersize = 10, color = :red,
             label = "L' * u_nodal")
    scatter!([-1.0, 1.0], collect(p_from_sum);
             marker = :x, markersize = 8, color = :blue,
             label = "∑ uᵢ ℓᵢ(±1)")
    vline!([-1, 1]; color = :gray, linestyle = :dash, label = "")
    return plt
end

plt_interp = plot_gauss_interpolant(polydeg)
savefig(plt_interp, joinpath(@__DIR__, "lagrange_gauss_interpolant.png"))
display(plt_interp)