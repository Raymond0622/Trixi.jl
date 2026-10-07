using LinearAlgebra: I
using Printf
using Trixi

# Where is MortarEntropy checkerboard ρ_x error largest:
# LGL volume-interior nodes vs Gauss surface-quadrature nodes on each face
# (hanging vs conforming). Gauss face nodes do not include LGL corners (±1).

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
eqs_av = Trixi.default_artificial_viscosity(equations)

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

analytic_rho_x(x, y) = π * cospi(2 * (x + y))

function make_semi(; checkerboard, mortar_type, initial_refinement_level)
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    surface_flux = flux_ranocha
    volume_integral = VolumeIntegralWeakForm()
    if mortar_type == "l2"
        mortar = MortarL2(basis)
    else
        mortar = MortarEntropy(basis; nodes = :gauss,
                               reverse_quadrature = :gauss)
    end
    surface_integral = SurfaceIntegralWeakFormGauss(surface_flux)
    solver = DGSEM(basis, surface_integral, volume_integral, mortar)
    solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()
    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = initial_refinement_level,
                    n_cells_max = 400_000, periodicity = true)
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
    return SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition_density_wave_ecav,
                                                 solver;
                                                 VDM = VDM, filter = filter,
                                                 ecav_choice = :ecav,
                                                 combine_rhs = Trixi.True(),
                                                 equations_artificial_viscosity = eqs_av,
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
end

function rho_x_fields(semi)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = Trixi.compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    (; u_transformed, gradients) = semi.cache_parabolic.parabolic_container
    eqs_av_semi = semi.artificial_viscosity.equations_artificial_viscosity
    Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
    Trixi.calc_gradient!(gradients, u_transformed, 0.0, mesh, equations_parabolic,
                         semi.boundary_conditions_parabolic, dg,
                         semi.solver_parabolic, cache)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av_semi, dg, cache)
    gx, _ = flux_parabolic
    n = Trixi.nnodes(dg)
    ne = Trixi.nelements(dg, cache)
    num = Array{Float64}(undef, n, n, ne)
    ex = similar(num)
    num .= gx[1, :, :, :]
    x = cache.elements.node_coordinates
    for element in Trixi.eachelement(dg, cache)
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            ex[i, j, element] = analytic_rho_x(x[1, i, j, element], x[2, i, j, element])
        end
    end
    return num, ex
end

function hanging_face_sets(dg, cache)
    ne = Trixi.nelements(dg, cache)
    hanging = [UInt8(0) for _ in 1:ne]
    hanging_large = [UInt8(0) for _ in 1:ne]
    hanging_small = [UInt8(0) for _ in 1:ne]
    mortars = cache.mortars
    for mortar in Trixi.eachmortar(dg, cache)
        large_dir, small_dir = Trixi.mortar_face_directions(mortars.large_sides[mortar],
                                                            mortars.orientations[mortar])
        large_el = mortars.neighbor_ids[3, mortar]
        small_lo = mortars.neighbor_ids[1, mortar]
        small_up = mortars.neighbor_ids[2, mortar]
        hanging[large_el] |= UInt8(0x01 << (large_dir - 1))
        hanging[small_lo] |= UInt8(0x01 << (small_dir - 1))
        hanging[small_up] |= UInt8(0x01 << (small_dir - 1))
        hanging_large[large_el] |= UInt8(0x01 << (large_dir - 1))
        hanging_small[small_lo] |= UInt8(0x01 << (small_dir - 1))
        hanging_small[small_up] |= UInt8(0x01 << (small_dir - 1))
    end
    return hanging, hanging_large, hanging_small
end

struct Bin
    n::Int
    l2acc::Float64
    linf::Float64
    absacc::Float64
end
Bin() = Bin(0, 0.0, 0.0, 0.0)
function add(b::Bin, err, w)
    return Bin(b.n + 1, b.l2acc + w * err^2, max(b.linf, err), b.absacc + err)
end

function face_kind(direction, hang_mask, hang_large, hang_small)
    bit = UInt8(0x01 << (direction - 1))
    (hang_large & bit) != 0 && return :hanging_large
    (hang_small & bit) != 0 && return :hanging_small
    return :conforming
end

# LGL face trace → Gauss surface nodes (no ±1 endpoints / corners).
function lgl_face_to_gauss(num, coords, element, direction, I_lg)
    n = size(num, 1)
    if direction == 1
        ρ = num[1, :, element]
        x = coords[1, 1, :, element]
        y = coords[2, 1, :, element]
    elseif direction == 2
        ρ = num[n, :, element]
        x = coords[1, n, :, element]
        y = coords[2, n, :, element]
    elseif direction == 3
        ρ = num[:, 1, element]
        x = coords[1, :, 1, element]
        y = coords[2, :, 1, element]
    else
        ρ = num[:, n, element]
        x = coords[1, :, n, element]
        y = coords[2, :, n, element]
    end
    return I_lg * ρ, I_lg * x, I_lg * y
end

function print_bins(bins, labels, total_l2sq)
    @printf("%-24s  %10s  %12s  %10s  %12s  %10s\n",
            "nodes", "count", "L∞", "mean |e|", "√(Σ w e²)", "% of L2²")
    for (k, lab) in labels
        b = bins[k]
        b.n == 0 && continue
        share = 100 * b.l2acc / max(total_l2sq, eps())
        @printf("%-24s  %10d  %12.4e  %10.3e  %12.4e  %9.2f%%\n",
                lab, b.n, b.linf, b.absacc / b.n, sqrt(b.l2acc), share)
    end
end

function diagnose(semi, title)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    num, ex = rho_x_fields(semi)
    hanging, hanging_large, hanging_small = hanging_face_sets(dg, cache)
    n = Trixi.nnodes(dg)
    coords = cache.elements.node_coordinates
    g_nodes, g_weights = Trixi.gauss_nodes_weights(n)
    I_lg = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, g_nodes)

    vol = Dict(k => Bin() for k in (:interior, :all))
    surf = Dict(k => Bin() for k in (:conforming, :hanging_small, :hanging_large,
                                     :all_hanging, :all))

    max_vol = (kind = :none, err = 0.0, i = 0, j = 0, element = 0, x = 0.0, y = 0.0)
    max_surf = (kind = :none, err = 0.0, direction = 0, k = 0, element = 0, x = 0.0, y = 0.0)

    for element in Trixi.eachelement(dg, cache)
        J1 = inv(cache.elements.inverse_jacobian[element])
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            on_face = (i == 1 || i == n || j == 1 || j == n)
            on_face && continue
            err = abs(num[i, j, element] - ex[i, j, element])
            w = dg.basis.weights[i] * dg.basis.weights[j] * J1^2
            vol[:interior] = add(vol[:interior], err, w)
            vol[:all] = add(vol[:all], err, w)
            if err > max_vol.err
                max_vol = (kind = :interior, err = err, i = i, j = j, element = element,
                           x = coords[1, i, j, element], y = coords[2, i, j, element])
            end
        end

        for direction in 1:4
            kind = face_kind(direction, hanging[element], hanging_large[element],
                             hanging_small[element])
            ρg, xg, yg = lgl_face_to_gauss(num, coords, element, direction, I_lg)
            for k in 1:n
                err = abs(ρg[k] - analytic_rho_x(xg[k], yg[k]))
                w = g_weights[k] * J1
                surf[kind] = add(surf[kind], err, w)
                surf[:all] = add(surf[:all], err, w)
                if kind != :conforming
                    surf[:all_hanging] = add(surf[:all_hanging], err, w)
                end
                if err > max_surf.err
                    max_surf = (kind = kind, err = err, direction = direction, k = k,
                                element = element, x = xg[k], y = yg[k])
                end
            end
        end
    end

    println("="^88)
    println(title)
    println("volume: LGL interior nodes     surface: Gauss face quadrature (no corners)")
    println("="^88)
    println("LGL volume interior")
    print_bins(vol, (:interior => "interior (volume)",), vol[:all].l2acc)
    println()
    println("Gauss surface integral of ρ_x error")
    print_bins(surf,
               (:conforming => "conforming face",
                :hanging_small => "hanging small-face",
                :hanging_large => "hanging large-face",
                :all_hanging => "all hanging-face",
                :all => "all Gauss face nodes"),
               surf[:all].l2acc)
    println()
    @printf("max |ρ_x| volume interior = %.4e  at (i,j,el)=(%d,%d,%d)  (x,y)=(%.5f, %.5f)\n",
            max_vol.err, max_vol.i, max_vol.j, max_vol.element, max_vol.x, max_vol.y)
    @printf("max |ρ_x| Gauss surface   = %.4e  at %s  dir=%d  gauss k=%d  el=%d  (x,y)=(%.5f, %.5f)\n",
            max_surf.err, max_surf.kind, max_surf.direction, max_surf.k, max_surf.element,
            max_surf.x, max_surf.y)
    println()
    return vol, surf, max_vol, max_surf
end

let
    tiny = make_semi(; checkerboard = false, mortar_type = "l2",
                     initial_refinement_level = 2)
    rho_x_fields(tiny)
end
GC.gc()

level = 7
for (mortar_type, title) in (("entropy", "MortarEntropy checkerboard"),
                             ("l2", "MortarL2 checkerboard (control)"))
    semi = make_semi(; checkerboard = true, mortar_type = mortar_type,
                     initial_refinement_level = level)
    diagnose(semi, "$title  ρ_x  Gauss surface  level = $level  polydeg = 3")
    GC.gc()
end
