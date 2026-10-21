# Smoke test: TreeMesh 3D Gauss-Legendre volume nodes + MortarEntropy, with and
# without ECAV, on a conforming mesh and a checkerboard with hanging faces.
using LinearAlgebra: I
using Trixi

function initial_condition_density_wave_ecav(x, t, equations::CompressibleEulerEquations3D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    v3 = convert(RealT, 0.3)
    rho = 1 + convert(RealT, 0.1) * sinpi(2 * (x[1] + x[2] + x[3] - t * (v1 + v2 + v3)))
    p = 20
    rho_e = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2 + v3^2)
    return SVector(rho, rho * v1, rho * v2, rho * v3, rho_e)
end

function checkerboard_refine!(mesh, coordinates_min, coordinates_max)
    level = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
    dx = (coordinates_max[1] - coordinates_min[1]) / 2^level
    cells = Int[]
    for cell_id in Trixi.leaf_cells(mesh.tree)
        x, y, z = Trixi.cell_coordinates(mesh.tree, cell_id)
        ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
        iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
        iz = round(Int, (z - coordinates_min[3]) / dx - 0.5)
        if iseven(ix + iy + iz)
            push!(cells, cell_id)
        end
    end
    Trixi.refine!(mesh.tree, cells)
    return mesh
end

function run_case(; nodes, ecav, checkerboard, polydeg = 2, level = 1)
    equations = CompressibleEulerEquations3D(1.4)
    equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = 0.0,
                                                              Prandtl = 0.73,
                                                              gradient_variables = GradientVariablesEntropy())
    solver_parabolic = ParabolicFormulationLocalDG()
    if nodes == :gauss
        basis = GaussLegendreBasis(polydeg)
        mortar = MortarEntropy(basis; nodes = :gauss, reverse_quadrature = :gauss)
    else
        basis = LobattoLegendreBasis(polydeg)
        mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                               reverse_quadrature = :gauss_lobatto)
    end
    solver = DGSEM(basis, SurfaceIntegralWeakForm(flux_ranocha), VolumeIntegralWeakForm(),
                   mortar)
    coordinates_min = (-1.0, -1.0, -1.0)
    coordinates_max = (1.0, 1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = level, periodicity = true)
    checkerboard && checkerboard_refine!(mesh, coordinates_min, coordinates_max)

    if ecav
        n = polydeg + 1
        semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                     initial_condition_density_wave_ecav,
                                                     solver;
                                                     VDM = Matrix{Float64}(I, n, n),
                                                     filter = ones(n),
                                                     ecav_choice = :ecav,
                                                     combine_rhs = Trixi.True(),
                                                     solver_parabolic = solver_parabolic,
                                                     boundary_conditions = (boundary_condition_periodic,
                                                                            boundary_condition_periodic))
    else
        semi = SemidiscretizationHyperbolic(mesh, equations,
                                            initial_condition_density_wave_ecav, solver;
                                            boundary_conditions = boundary_condition_periodic)
    end

    ode = semidiscretize(semi, (0.0, 1.0e-3))
    du = similar(ode.u0)
    if ecav
        Trixi.rhs!(du, ode.u0, semi, 0.0)
    else
        Trixi.rhs_hyperbolic!(du, ode.u0, semi, 0.0)
    end
    label = "nodes=$(nodes) ecav=$(ecav) checkerboard=$(checkerboard)"
    @assert all(isfinite, du) "non-finite du for $(label)"
    n_mortars = length(semi.cache.mortars.orientations)
    msg = "$(label): elements=$(Trixi.nelements(solver, semi.cache)) mortars=$(n_mortars) max|du|=$(maximum(abs, du))"
    if ecav
        r = semi.cache.artificial_viscosity.coefficients
        msg *= " residual[$(minimum(r)), $(maximum(r))]"
        @assert all(isfinite, r) "non-finite residual for $(label)"
    end
    println(msg)
    return nothing
end

for nodes in (:gauss, :lgl)
    for ecav in (false, true)
        for checkerboard in (false, true)
            run_case(; nodes, ecav, checkerboard)
        end
    end
end
println("ok")
