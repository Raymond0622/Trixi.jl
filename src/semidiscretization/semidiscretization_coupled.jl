# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    SemidiscretizationCoupled

A struct used to bundle multiple semidiscretizations.
[`semidiscretize`](@ref) will return an `ODEProblem` that synchronizes time steps between the semidiscretizations.
Each call of `rhs!` will call `rhs!` for each semidiscretization individually.
The semidiscretizations can be coupled by gluing meshes together using [`BoundaryConditionCoupled`](@ref).

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct SemidiscretizationCoupled{S, Indices} <:
               AbstractSemidiscretization
    semis::S
    u_indices::Indices # u_ode[u_indices[i]] is the part of u_ode corresponding to semis[i]
    performance_counter::PerformanceCounter
end
# We assume some properties of the fields of the semidiscretization, e.g.,
# the `equations` and the `mesh` should have the same dimension. We check these
# properties in the outer constructor defined below. While we could ensure
# them even better in an inner constructor, we do not use this approach to
# simplify the integration with Adapt.jl for GPU usage, see
# https://github.com/trixi-framework/Trixi.jl/pull/2677#issuecomment-3591789921

"""
    SemidiscretizationCoupled(semis...)

Create a coupled semidiscretization that consists of the semidiscretizations passed as arguments.
"""
function SemidiscretizationCoupled(semis...)
    @assert all(semi -> ndims(semi) == ndims(semis[1]), semis) "All semidiscretizations must have the same dimension!"

    # Number of coefficients for each semidiscretization
    n_coefficients = zeros(Int, length(semis))
    for i in 1:length(semis)
        _, equations, _, _ = mesh_equations_solver_cache(semis[i])
        n_coefficients[i] = ndofs(semis[i]) * nvariables(equations)
    end

    # Compute range of coefficients associated with each semidiscretization and allocate coupled BCs
    u_indices = Vector{UnitRange{Int}}(undef, length(semis))
    for i in 1:length(semis)
        offset = sum(n_coefficients[1:(i - 1)]) + 1
        u_indices[i] = range(offset, length = n_coefficients[i])

        allocate_coupled_boundary_conditions(semis[i])
    end

    performance_counter = PerformanceCounter()

    return SemidiscretizationCoupled{typeof(semis), typeof(u_indices)}(semis, u_indices,
                                                                       performance_counter)
end

function Base.show(io::IO, semi::SemidiscretizationCoupled)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationCoupled($(semi.semis))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationCoupled)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationCoupled")
        summary_line(io, "#spatial dimensions", ndims(semi.semis[1]))
        summary_line(io, "#systems", nsystems(semi))
        for i in eachsystem(semi)
            summary_line(io, "system", i)
            mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
            summary_line(increment_indent(io), "mesh", mesh |> typeof |> nameof)
            summary_line(increment_indent(io), "equations",
                         equations |> typeof |> nameof)
            summary_line(increment_indent(io), "initial condition",
                         semi.semis[i].initial_condition)
            # no boundary conditions since that could be too much
            summary_line(increment_indent(io), "source terms",
                         semi.semis[i].source_terms)
            summary_line(increment_indent(io), "solver", solver |> typeof |> nameof)
        end
        summary_line(io, "total #DOFs per field", ndofsglobal(semi))
        summary_footer(io)
    end
end

function print_summary_semidiscretization(io::IO, semi::SemidiscretizationCoupled)
    show(io, MIME"text/plain"(), semi)
    println(io, "\n")
    for i in eachsystem(semi)
        mesh, equations, solver, _ = mesh_equations_solver_cache(semi.semis[i])
        summary_header(io, "System #$i")

        summary_line(io, "mesh", mesh |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), mesh)

        summary_line(io, "equations", equations |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), equations)

        summary_line(io, "solver", solver |> typeof |> nameof)
        show(increment_indent(io), MIME"text/plain"(), solver)

        summary_footer(io)
        println(io, "\n")
    end
end

@inline Base.ndims(semi::SemidiscretizationCoupled) = ndims(semi.semis[1])

@inline nsystems(semi::SemidiscretizationCoupled) = length(semi.semis)

@inline eachsystem(semi::SemidiscretizationCoupled) = Base.OneTo(nsystems(semi))

@inline Base.real(semi::SemidiscretizationCoupled) = promote_type(real.(semi.semis)...)

@inline function Base.eltype(semi::SemidiscretizationCoupled)
    return promote_type(eltype.(semi.semis)...)
end

@inline function ndofs(semi::SemidiscretizationCoupled)
    return sum(ndofs, semi.semis)
end

"""
    ndofsglobal(semi::SemidiscretizationCoupled)

Return the global number of degrees of freedom associated with each scalar variable across all MPI ranks, and summed up over all coupled systems.
This is the same as [`ndofs`](@ref) for simulations running in serial or
parallelized via threads. It will in general be different for simulations
running in parallel with MPI.
"""
@inline function ndofsglobal(semi::SemidiscretizationCoupled)
    return sum(ndofsglobal, semi.semis)
end

function compute_coefficients(t, semi::SemidiscretizationCoupled)
    @unpack u_indices = semi

    u_ode = Vector{real(semi)}(undef, u_indices[end][end])

    for i in eachsystem(semi)
        # Call `compute_coefficients` in `src/semidiscretization/semidiscretization.jl`
        u_ode[u_indices[i]] .= compute_coefficients(t, semi.semis[i])
    end

    return u_ode
end

@inline function get_system_u_ode(u_ode, index, semi::SemidiscretizationCoupled)
    @view u_ode[semi.u_indices[index]]
end

# Same as `foreach(enumerate(something))`, but without allocations.
#
# Note that compile times may increase if this is used with big tuples.
@inline foreach_enumerate(func, collection) = foreach_enumerate(func, collection, 1)
@inline foreach_enumerate(func, collection::Tuple{}, index) = nothing

@inline function foreach_enumerate(func, collection, index)
    element = first(collection)
    remaining_collection = Base.tail(collection)

    func((index, element))

    # Process remaining collection
    return foreach_enumerate(func, remaining_collection, index + 1)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationCoupled, t)
    @unpack u_indices = semi

    time_start = time_ns()

    @trixi_timeit timer() "copy to coupled boundaries" begin
        foreach(semi.semis) do semi_
            return copy_to_coupled_boundary!(semi_.boundary_conditions, u_ode, semi,
                                             semi_)
        end
    end

    @trixi_timeit timer() "p-mortar fluxes" begin
        foreach(semi.semis) do semi_
            return calc_coupled_p_mortar_fluxes!(semi_.boundary_conditions, u_ode, semi,
                                                 semi_)
        end
    end

    # Call rhs! for each semidiscretization
    foreach_enumerate(semi.semis) do (i, semi_)
        u_loc = get_system_u_ode(u_ode, i, semi)
        du_loc = get_system_u_ode(du_ode, i, semi)
        return rhs!(du_loc, u_loc, semi_, t)
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

################################################################################
### AnalysisCallback
################################################################################

"""
    AnalysisCallbackCoupled(semi, callbacks...)

Combine multiple analysis callbacks for coupled simulations with a
[`SemidiscretizationCoupled`](@ref). For each coupled system, an indididual
[`AnalysisCallback`](@ref) **must** be created and passed to the `AnalysisCallbackCoupled` **in
order**, i.e., in the same sequence as the indidvidual semidiscretizations are stored in the
`SemidiscretizationCoupled`.

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
struct AnalysisCallbackCoupled{CB}
    callbacks::CB
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb_coupled::DiscreteCallback{<:Any, <:AnalysisCallbackCoupled})
    @nospecialize cb_coupled # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb_coupled)
    else
        analysis_callback_coupled = cb_coupled.affect!

        summary_header(io, "AnalysisCallbackCoupled")
        for (i, cb) in enumerate(analysis_callback_coupled.callbacks)
            summary_line(io, "Callback #$i", "")
            show(increment_indent(io), MIME"text/plain"(), cb)
        end
        summary_footer(io)
    end
end

# Convenience constructor for the coupled callback that gets called directly from the elixirs
function AnalysisCallbackCoupled(semi_coupled, callbacks...)
    if length(callbacks) != nsystems(semi_coupled)
        error("an AnalysisCallbackCoupled requires one AnalysisCallback for each semidiscretization")
    end

    analysis_callback_coupled = AnalysisCallbackCoupled{typeof(callbacks)}(callbacks)

    # This callback is triggered if any of its subsidiary callbacks' condition is triggered
    condition = (u, t, integrator) -> any(callbacks) do callback
        return callback.condition(u, t, integrator)
    end

    return DiscreteCallback(condition, analysis_callback_coupled,
                            save_positions = (false, false),
                            initialize = initialize!)
end

# This method gets called during initialization from OrdinaryDiffEq's `solve(...)`
function initialize!(cb_coupled::DiscreteCallback{Condition, Affect!}, u_ode_coupled, t,
                     integrator) where {Condition, Affect! <: AnalysisCallbackCoupled}
    analysis_callback_coupled = cb_coupled.affect!
    semi_coupled = integrator.p
    du_ode_coupled = first(get_tmp_cache(integrator))

    # Loop over coupled systems' callbacks and initialize them individually
    for i in eachsystem(semi_coupled)
        cb = analysis_callback_coupled.callbacks[i]
        semi = semi_coupled.semis[i]
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)
        du_ode = get_system_u_ode(du_ode_coupled, i, semi_coupled)
        initialize!(cb, u_ode, du_ode, t, integrator, semi)
    end
end

# This method gets called from OrdinaryDiffEq's `solve(...)`
function (analysis_callback_coupled::AnalysisCallbackCoupled)(integrator)
    semi_coupled = integrator.p
    u_ode_coupled = integrator.u
    du_ode_coupled = first(get_tmp_cache(integrator))

    # Loop over coupled systems' callbacks and call them individually
    for i in eachsystem(semi_coupled)
        @unpack condition = analysis_callback_coupled.callbacks[i]
        analysis_callback = analysis_callback_coupled.callbacks[i].affect!
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)

        # Check condition and skip callback if it is not yet its turn
        if !condition(u_ode, integrator.t, integrator)
            continue
        end

        semi = semi_coupled.semis[i]
        du_ode = get_system_u_ode(du_ode_coupled, i, semi_coupled)
        analysis_callback(u_ode, du_ode, integrator, semi)
    end
end

# used for error checks and EOC analysis
function (cb::DiscreteCallback{Condition, Affect!})(sol) where {Condition,
                                                                Affect! <:
                                                                AnalysisCallbackCoupled}
    semi_coupled = sol.prob.p
    u_ode_coupled = sol.u[end]
    @unpack callbacks = cb.affect!

    uEltype = real(semi_coupled)
    l2_error_collection = uEltype[]
    linf_error_collection = uEltype[]
    for i in eachsystem(semi_coupled)
        analysis_callback = callbacks[i].affect!
        @unpack analyzer = analysis_callback
        cache_analysis = analysis_callback.cache

        semi = semi_coupled.semis[i]
        u_ode = get_system_u_ode(u_ode_coupled, i, semi_coupled)

        l2_error, linf_error = calc_error_norms(u_ode, sol.t[end], analyzer, semi,
                                                cache_analysis)
        append!(l2_error_collection, l2_error)
        append!(linf_error_collection, linf_error)
    end

    return (; l2 = l2_error_collection, linf = linf_error_collection)
end

################################################################################
### SaveSolutionCallback
################################################################################

# Save mesh for a coupled semidiscretization, which contains multiple meshes internally
function save_mesh(semi::SemidiscretizationCoupled, output_directory, timestep = 0)
    for i in eachsystem(semi)
        mesh, _, _, _ = mesh_equations_solver_cache(semi.semis[i])

        if mesh.unsaved_changes
            mesh.current_filename = save_mesh_file(mesh, output_directory; system = i,
                                                   timestep = timestep)
            mesh.unsaved_changes = false
        end
    end
end

@inline function save_solution_file(semi::SemidiscretizationCoupled, u_ode,
                                    solution_callback,
                                    integrator)
    @unpack semis = semi

    for i in eachsystem(semi)
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        save_solution_file(semis[i], u_ode_slice, solution_callback, integrator,
                           system = i)
    end
end

################################################################################
### StepsizeCallback
################################################################################
# In case of coupled system, use minimum timestep over all systems
function calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic,
                      semi::SemidiscretizationCoupled)
    dt = minimum(eachsystem(semi)) do i
        u_ode_slice = get_system_u_ode(u_ode, i, semi)
        return calculate_dt(u_ode_slice, t, cfl_hyperbolic, cfl_parabolic,
                            semi.semis[i])
    end

    return dt
end

function update_cleaning_speed!(semi_coupled::SemidiscretizationCoupled,
                                glm_speed_callback, dt, t)
    @unpack glm_scale, cfl, semi_indices = glm_speed_callback

    if length(semi_indices) == 0
        throw("Since you have more than one semidiscretization you need to specify the 'semi_indices' for which the GLM speed needs to be calculated.")
    end

    # Check that all MHD semidiscretizations received a GLM cleaning speed update.
    for (semi_index, semi) in enumerate(semi_coupled.semis)
        if (typeof(semi.equations) <: AbstractIdealGlmMhdEquations &&
            !(semi_index in semi_indices))
            error("Equation of semidiscretization $semi_index needs to be included in 'semi_indices' of 'GlmSpeedCallback'.")
        end
    end

    for semi_index in semi_indices
        semi = semi_coupled.semis[semi_index]
        mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

        # compute time step for GLM linear advection equation with c_h=1 (redone due to the possible AMR)
        c_h_deltat = calc_dt_for_cleaning_speed(cfl(t),
                                                mesh, equations, solver, cache)

        # c_h is proportional to its own time step divided by the complete MHD time step
        # We use @reset here since the equations are immutable (to work on GPUs etc.).
        # Thus, we need to modify the equations field of the semidiscretization.
        @reset equations.c_h = glm_scale * c_h_deltat / dt
        semi.equations = equations
    end

    return semi_coupled
end

################################################################################
### Equations
################################################################################

"""
    BoundaryConditionCoupled(other_semi_index, indices, uEltype, coupling_converter)

Boundary condition to glue two meshes together. Solution values at the boundary
of another mesh will be used as boundary values. This requires the use
of [`SemidiscretizationCoupled`](@ref). The other mesh is specified by `other_semi_index`,
which is the index of the mesh in the tuple of semidiscretizations.

Note that the elements and nodes of the two meshes at the coupled boundary must coincide.
This is currently only implemented for [`StructuredMesh`](@ref).

# Arguments
- `other_semi_index`: the index in `SemidiscretizationCoupled` of the semidiscretization
                      from which the values are copied
- `indices::Tuple`: node/cell indices at the boundary of the mesh in the other
                    semidiscretization. See examples below.
- `uEltype::Type`: element type of solution
- `coupling_converter::CouplingConverter`: function to call for converting the solution
                                           state of one system to the other system

# Examples
```julia
# Connect the left boundary of mesh 2 to our boundary such that our positive
# boundary direction will match the positive y direction of the other boundary
BoundaryConditionCoupled(2, (:begin, :i), Float64, fun)

# Connect the same two boundaries oppositely oriented
BoundaryConditionCoupled(2, (:begin, :i_backwards), Float64, fun)

# Using this as y_neg boundary will connect `our_cells[i, 1, j]` to `other_cells[j, end-i, end]`
BoundaryConditionCoupled(2, (:j, :i_backwards, :end), Float64, fun)
```

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct BoundaryConditionCoupled{NDIMS,
                                        # Store the other semi index as type parameter,
                                        # so that retrieving the other semidiscretization
                                        # is type-stable.
                                        # x-ref: https://github.com/trixi-framework/Trixi.jl/pull/1979
                                        other_semi_index, NDIMST2M1,
                                        uEltype <: Real, Indices, CouplingConverter}
    # NDIMST2M1 == NDIMS * 2 - 1
    # Buffer for boundary values: [variable, nodes_i, nodes_j, cell_i, cell_j]
    u_boundary               :: Array{uEltype, NDIMST2M1} # NDIMS * 2 - 1
    const other_orientation  :: Int
    const indices            :: Indices
    const coupling_converter :: CouplingConverter

    function BoundaryConditionCoupled(other_semi_index, indices, uEltype,
                                      coupling_converter)
        NDIMS = length(indices)
        u_boundary = Array{uEltype, NDIMS * 2 - 1}(undef, ntuple(_ -> 0, NDIMS * 2 - 1))

        if indices[1] in (:begin, :end)
            other_orientation = 1
        elseif indices[2] in (:begin, :end)
            other_orientation = 2
        else # indices[3] in (:begin, :end)
            other_orientation = 3
        end

        return new{NDIMS, other_semi_index, NDIMS * 2 - 1, uEltype, typeof(indices),
                   typeof(coupling_converter)}(u_boundary,
                                               other_orientation,
                                               indices, coupling_converter)
    end
end

function Base.eltype(boundary_condition::BoundaryConditionCoupled)
    return eltype(boundary_condition.u_boundary)
end

function (boundary_condition::BoundaryConditionCoupled)(u_inner, orientation, direction,
                                                        cell_indices,
                                                        surface_node_indices,
                                                        surface_flux_function,
                                                        equations)
    # get_node_vars(boundary_condition.u_boundary, equations, solver, surface_node_indices..., cell_indices...),
    # but we don't have a solver here
    u_boundary = SVector(ntuple(v -> boundary_condition.u_boundary[v,
                                                                   surface_node_indices...,
                                                                   cell_indices...],
                                Val(nvariables(equations))))

    # Calculate boundary flux
    if surface_flux_function isa Tuple
        # In case of conservative (index 1) and non-conservative (index 2) fluxes,
        # add the non-conservative one with a factor of 1/2.
        if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
            flux = (surface_flux_function[1](u_inner, u_boundary, orientation,
                                             equations),
                    surface_flux_function[2](u_inner, u_boundary, orientation,
                                             equations))
        else # u_boundary is "left" of boundary, u_inner is "right" of boundary
            flux = (surface_flux_function[1](u_boundary, u_inner, orientation,
                                             equations),
                    surface_flux_function[2](u_boundary, u_inner, orientation,
                                             equations))
        end
    else
        if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
            flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
        else # u_boundary is "left" of boundary, u_inner is "right" of boundary
            flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
        end
    end

    return flux
end

"""
    BoundaryConditionCoupledPMortar(other_semi_index, indices, uEltype, coupling_converter,
                                    basis_own, basis_other; entropy_variables = false)

Like [`BoundaryConditionCoupled`](@ref), but the adjacent faces may use different
surface quadrature (1:1 p-nonconforming). Neighbor traces are stored on their
native nodes. The Riemann flux is evaluated on a Gauss mortar quadrature that
differs from both faces, then L²-projected back onto this element's original
nodes via [`MortarP1to1`](@ref).

Currently implemented for 2D [`StructuredMesh`](@ref) / [`StructuredMeshView`](@ref).
Both coupled solvers should use the same `surface_flux`. The Riemann problem uses
contravariant face normals (same scaling as interior structured interfaces).
On affine Cartesian mappings the mortar is globally conservative.

# Arguments
- `other_semi_index`, `indices`, `uEltype`, `coupling_converter`: same as
  [`BoundaryConditionCoupled`](@ref)
- `basis_own`: SBP basis of this semidiscretization
- `basis_other`: SBP basis of the neighbor semidiscretization
- `entropy_variables`: if `true`, interpolate traces in entropy variables
  (flux is still computed in conservative variables and L²-projected)

!!! warning "Experimental code"
    This is an experimental feature and can change any time.
"""
mutable struct BoundaryConditionCoupledPMortar{NDIMS,
                                               other_semi_index, NDIMST2M1,
                                               uEltype <: Real, Indices,
                                               CouplingConverter, Mortar}
    # Neighbor state on the neighbor's native face nodes:
    # [variable, nodes_other, cell]
    u_boundary               :: Array{uEltype, NDIMST2M1}
    # Riemann flux L²-projected onto this element's face nodes:
    # [variable, nodes_own, cell]
    flux_projected           :: Array{uEltype, NDIMST2M1}
    const other_orientation  :: Int
    const indices            :: Indices
    own_is_left              :: Bool
    const coupling_converter :: CouplingConverter
    const mortar             :: Mortar
    const entropy_variables  :: Bool

    function BoundaryConditionCoupledPMortar(other_semi_index, indices, uEltype,
                                             coupling_converter,
                                             basis_own, basis_other;
                                             entropy_variables = false)
        NDIMS = length(indices)
        @assert NDIMS == 2 "BoundaryConditionCoupledPMortar is currently only implemented for 2D"

        u_boundary = Array{uEltype, NDIMS * 2 - 1}(undef,
                                                   ntuple(_ -> 0, NDIMS * 2 - 1))
        flux_projected = Array{uEltype, NDIMS * 2 - 1}(undef,
                                                       ntuple(_ -> 0, NDIMS * 2 - 1))

        if indices[1] in (:begin, :end)
            other_orientation = 1
        else
            other_orientation = 2
        end

        mortar = MortarP1to1(basis_own, basis_other)

        return new{NDIMS, other_semi_index, NDIMS * 2 - 1, uEltype, typeof(indices),
                   typeof(coupling_converter), typeof(mortar)}(u_boundary,
                                                               flux_projected,
                                                               other_orientation,
                                                               indices, false,
                                                               coupling_converter,
                                                               mortar,
                                                               entropy_variables)
    end
end

function Base.eltype(boundary_condition::BoundaryConditionCoupledPMortar)
    return eltype(boundary_condition.u_boundary)
end

function allocate_coupled_boundary_conditions(semi::AbstractSemidiscretization)
    n_boundaries = 2 * ndims(semi)
    mesh, equations, solver, _ = mesh_equations_solver_cache(semi)

    for direction in 1:n_boundaries
        boundary_condition = semi.boundary_conditions[direction]

        allocate_coupled_boundary_condition(boundary_condition, direction, mesh,
                                            equations,
                                            solver)
    end
end

# Don't do anything for other BCs than BoundaryConditionCoupled
function allocate_coupled_boundary_condition(boundary_condition, direction, mesh,
                                             equations,
                                             solver)
    return nothing
end

# In 2D
function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupled{2},
                                             direction, mesh, equations, dg::DGSEM)
    if direction in (1, 2)
        cell_size = size(mesh, 2)
    else
        cell_size = size(mesh, 1)
    end

    uEltype = eltype(boundary_condition)
    return boundary_condition.u_boundary = Array{uEltype, 3}(undef,
                                                             nvariables(equations),
                                                             nnodes(dg),
                                                             cell_size)
end

function allocate_coupled_boundary_condition(boundary_condition::BoundaryConditionCoupledPMortar{2},
                                             direction, mesh, equations, dg::DGSEM)
    if direction in (1, 2)
        cell_size = size(mesh, 2)
    else
        cell_size = size(mesh, 1)
    end

    uEltype = eltype(boundary_condition)
    n_own = nnodes_own(boundary_condition.mortar)
    n_other = nnodes_other(boundary_condition.mortar)
    boundary_condition.u_boundary = Array{uEltype, 3}(undef,
                                                      nvariables(equations),
                                                      n_other,
                                                      cell_size)
    boundary_condition.flux_projected = Array{uEltype, 3}(undef,
                                                          nvariables(equations),
                                                          n_own,
                                                          cell_size)
    # Even directions (x_pos, y_pos) are the left state of the Riemann problem.
    boundary_condition.own_is_left = iseven(direction)

    return nothing
end

# Don't do anything for other BCs than BoundaryConditionCoupled
function copy_to_coupled_boundary!(boundary_condition, u_ode, semi_coupled, semi)
    return nothing
end

function copy_to_coupled_boundary!(u_ode, semi_coupled, semi, i, n_boundaries,
                                   boundary_condition, boundary_conditions...)
    copy_to_coupled_boundary!(boundary_condition, u_ode, semi_coupled, semi)
    if i < n_boundaries
        copy_to_coupled_boundary!(u_ode, semi_coupled, semi, i + 1, n_boundaries,
                                  boundary_conditions...)
    end
end

function copy_to_coupled_boundary!(boundary_conditions::Union{Tuple, NamedTuple}, u_ode,
                                   semi_coupled, semi)
    return copy_to_coupled_boundary!(u_ode, semi_coupled, semi, 1,
                                     length(boundary_conditions),
                                     boundary_conditions...)
end

# In 2D
function copy_to_coupled_boundary!(boundary_condition::BoundaryConditionCoupled{2,
                                                                                other_semi_index},
                                   u_ode, semi_coupled, semi) where {other_semi_index}
    @unpack u_indices = semi_coupled
    @unpack other_orientation, indices = boundary_condition
    @unpack coupling_converter, u_boundary = boundary_condition

    mesh_own, equations_own, solver_own, cache_own = mesh_equations_solver_cache(semi)
    other_semi = semi_coupled.semis[other_semi_index]
    mesh_other, equations_other, solver_other, cache_other = mesh_equations_solver_cache(other_semi)

    node_coordinates_other = cache_other.elements.node_coordinates
    u_ode_other = get_system_u_ode(u_ode, other_semi_index, semi_coupled)
    u_other = wrap_array(u_ode_other, mesh_other, equations_other, solver_other,
                         cache_other)

    linear_indices = LinearIndices(size(mesh_other))

    if other_orientation == 1
        cells = axes(mesh_other, 2)
    else # other_orientation == 2
        cells = axes(mesh_other, 1)
    end

    # Copy solution data to the coupled boundary using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    node_index_range = eachnode(solver_other)
    i_node_start, i_node_step = index_to_start_step_2d(indices[1], node_index_range)
    j_node_start, j_node_step = index_to_start_step_2d(indices[2], node_index_range)

    i_cell_start, i_cell_step = index_to_start_step_2d(indices[1], axes(mesh_other, 1))
    j_cell_start, j_cell_step = index_to_start_step_2d(indices[2], axes(mesh_other, 2))

    # We need indices starting at 1 for the handling of `i_cell` etc.
    Base.require_one_based_indexing(cells)

    # 1:1 p-nonconforming coupling: interpolate the neighbor face from its LGL
    # nodes onto this side's LGL nodes. Same-degree faces keep a direct copy.
    nnodes_own = nnodes(solver_own)
    nnodes_other = nnodes(solver_other)
    interpolate_face = nnodes_own != nnodes_other
    vandermonde = interpolate_face ?
                  polynomial_interpolation_matrix(solver_other.basis.nodes,
                                                  solver_own.basis.nodes) :
                  nothing

    @threaded for i in eachindex(cells)
        cell = cells[i]
        i_cell = i_cell_start + (i - 1) * i_cell_step
        j_cell = j_cell_start + (i - 1) * j_cell_step

        i_node = i_node_start
        j_node = j_node_start
        element_other = linear_indices[i_cell, j_cell]

        if interpolate_face
            nvars = size(u_boundary, 1)
            u_src = zeros(eltype(u_boundary), nvars, nnodes_other)
            for k in eachnode(solver_other)
                x_other = get_node_coords(node_coordinates_other, equations_other,
                                          solver_other, i_node, j_node, element_other)
                u_node_other = get_node_vars(u_other, equations_other, solver_other,
                                             i_node, j_node, element_other)
                u_node_converted = coupling_converter(x_other, u_node_other,
                                                      equations_other, equations_own)
                for v in eachindex(u_node_converted)
                    u_src[v, k] = u_node_converted[v]
                end
                i_node += i_node_step
                j_node += j_node_step
            end
            for k_own in 1:nnodes_own
                for v in 1:nvars
                    acc = zero(eltype(u_boundary))
                    for k_src in 1:nnodes_other
                        acc += vandermonde[k_own, k_src] * u_src[v, k_src]
                    end
                    u_boundary[v, k_own, cell] = acc
                end
            end
        else
            for node in eachnode(solver_other)
                x_other = get_node_coords(node_coordinates_other, equations_other,
                                          solver_other, i_node, j_node, element_other)
                u_node_other = get_node_vars(u_other, equations_other, solver_other,
                                             i_node, j_node, element_other)
                u_node_converted = coupling_converter(x_other, u_node_other,
                                                      equations_other, equations_own)

                for v in eachindex(u_node_converted)
                    u_boundary[v, node, cell] = u_node_converted[v]
                end

                i_node += i_node_step
                j_node += j_node_step
            end
        end
    end

    return nothing
end

# Copy neighbor traces on their native quadrature (no interpolation).
function copy_to_coupled_boundary!(boundary_condition::BoundaryConditionCoupledPMortar{2,
                                                                                       other_semi_index},
                                   u_ode, semi_coupled, semi) where {other_semi_index}
    @unpack indices = boundary_condition
    @unpack coupling_converter, u_boundary = boundary_condition

    mesh_own, equations_own, solver_own, cache_own = mesh_equations_solver_cache(semi)
    other_semi = semi_coupled.semis[other_semi_index]
    mesh_other, equations_other, solver_other, cache_other = mesh_equations_solver_cache(other_semi)

    node_coordinates_other = cache_other.elements.node_coordinates
    u_ode_other = get_system_u_ode(u_ode, other_semi_index, semi_coupled)
    u_other = wrap_array(u_ode_other, mesh_other, equations_other, solver_other,
                         cache_other)

    linear_indices = LinearIndices(size(mesh_other))

    if boundary_condition.other_orientation == 1
        cells = axes(mesh_other, 2)
    else
        cells = axes(mesh_other, 1)
    end

    node_index_range = eachnode(solver_other)
    i_node_start, i_node_step = index_to_start_step_2d(indices[1], node_index_range)
    j_node_start, j_node_step = index_to_start_step_2d(indices[2], node_index_range)

    i_cell_start, i_cell_step = index_to_start_step_2d(indices[1], axes(mesh_other, 1))
    j_cell_start, j_cell_step = index_to_start_step_2d(indices[2], axes(mesh_other, 2))

    Base.require_one_based_indexing(cells)

    @threaded for i in eachindex(cells)
        cell = cells[i]
        i_cell = i_cell_start + (i - 1) * i_cell_step
        j_cell = j_cell_start + (i - 1) * j_cell_step

        i_node = i_node_start
        j_node = j_node_start
        element_other = linear_indices[i_cell, j_cell]

        for node in eachnode(solver_other)
            x_other = get_node_coords(node_coordinates_other, equations_other,
                                      solver_other, i_node, j_node, element_other)
            u_node_other = get_node_vars(u_other, equations_other, solver_other,
                                         i_node, j_node, element_other)
            u_node_converted = coupling_converter(x_other, u_node_other,
                                                  equations_other, equations_own)

            for v in eachindex(u_node_converted)
                u_boundary[v, node, cell] = u_node_converted[v]
            end

            i_node += i_node_step
            j_node += j_node_step
        end
    end

    return nothing
end

# Don't do anything for BCs that are not 1:1 p-mortars
function calc_coupled_p_mortar_fluxes!(boundary_condition, u_ode, semi_coupled, semi)
    return nothing
end

function calc_coupled_p_mortar_fluxes!(u_ode, semi_coupled, semi, i, n_boundaries,
                                       boundary_condition, boundary_conditions...)
    calc_coupled_p_mortar_fluxes!(boundary_condition, u_ode, semi_coupled, semi)
    if i < n_boundaries
        calc_coupled_p_mortar_fluxes!(u_ode, semi_coupled, semi, i + 1, n_boundaries,
                                      boundary_conditions...)
    end
end

function calc_coupled_p_mortar_fluxes!(boundary_conditions::Union{Tuple, NamedTuple},
                                       u_ode, semi_coupled, semi)
    return calc_coupled_p_mortar_fluxes!(u_ode, semi_coupled, semi, 1,
                                         length(boundary_conditions),
                                         boundary_conditions...)
end

function calc_coupled_p_mortar_fluxes!(boundary_condition::BoundaryConditionCoupledPMortar{2,
                                                                                           other_semi_index},
                                       u_ode, semi_coupled, semi) where {other_semi_index}
    @unpack mortar, u_boundary, flux_projected, other_orientation, own_is_left = boundary_condition
    @unpack entropy_variables = boundary_condition

    mesh_own, equations, solver_own, cache_own = mesh_equations_solver_cache(semi)

    own_semi_index = find_semi_index(semi, semi_coupled)
    u_ode_own = get_system_u_ode(u_ode, own_semi_index, semi_coupled)
    u_own = wrap_array(u_ode_own, mesh_own, equations, solver_own, cache_own)
    contravariant_vectors = cache_own.elements.contravariant_vectors

    surface_flux = solver_own.surface_integral.surface_flux
    if surface_flux isa Tuple
        error("BoundaryConditionCoupledPMortar does not support nonconservative terms")
    end

    own_indices = own_face_indices(other_orientation, own_is_left)
    linear_indices = LinearIndices(size(mesh_own))

    if other_orientation == 1
        cells = axes(mesh_own, 2)
    else
        cells = axes(mesh_own, 1)
    end

    node_index_range = eachnode(solver_own)
    i_node_start, i_node_step = index_to_start_step_2d(own_indices[1], node_index_range)
    j_node_start, j_node_step = index_to_start_step_2d(own_indices[2], node_index_range)
    i_cell_start, i_cell_step = index_to_start_step_2d(own_indices[1], axes(mesh_own, 1))
    j_cell_start, j_cell_step = index_to_start_step_2d(own_indices[2], axes(mesh_own, 2))

    Base.require_one_based_indexing(cells)

    n_own = nnodes_own(mortar)
    n_other = nnodes_other(mortar)
    n_mortar = nnodes_mortar(mortar)
    nvars = nvariables(equations)

    @threaded for i in eachindex(cells)
        cell = cells[i]
        i_cell = i_cell_start + (i - 1) * i_cell_step
        j_cell = j_cell_start + (i - 1) * j_cell_step

        i_node = i_node_start
        j_node = j_node_start
        element_own = linear_indices[i_cell, j_cell]

        u_face_own = zeros(eltype(flux_projected), nvars, n_own)
        u_face_other = zeros(eltype(flux_projected), nvars, n_other)
        u_mortar_own = zeros(eltype(flux_projected), nvars, n_mortar)
        u_mortar_other = zeros(eltype(flux_projected), nvars, n_mortar)
        f_mortar = zeros(eltype(flux_projected), nvars, n_mortar)
        # Contravariant normal on the own LGL face; interpolated to the mortar
        # so the Riemann flux matches interior structured-mesh scaling (f · Ja).
        n_dims = ndims(equations)
        ja_face = zeros(eltype(flux_projected), n_dims, n_own)
        ja_mortar = zeros(eltype(flux_projected), n_dims, n_mortar)

        for node in 1:n_own
            u_node = get_node_vars(u_own, equations, solver_own,
                                   i_node, j_node, element_own)
            if entropy_variables
                u_node = cons2entropy(u_node, equations)
            end
            for v in eachvariable(equations)
                u_face_own[v, node] = u_node[v]
            end
            ja = get_contravariant_vector(other_orientation, contravariant_vectors,
                                          i_node, j_node, element_own)
            for d in 1:n_dims
                ja_face[d, node] = ja[d]
            end
            i_node += i_node_step
            j_node += j_node_step
        end

        for node in 1:n_other
            u_node = SVector(ntuple(v -> u_boundary[v, node, cell], Val(nvars)))
            if entropy_variables
                u_node = cons2entropy(u_node, equations)
            end
            for v in eachvariable(equations)
                u_face_other[v, node] = u_node[v]
            end
        end

        multiply_dimensionwise_face!(u_mortar_own, mortar.vandermonde_own, u_face_own)
        multiply_dimensionwise_face!(u_mortar_other, mortar.vandermonde_other,
                                     u_face_other)
        multiply_dimensionwise_face!(ja_mortar, mortar.vandermonde_own, ja_face)

        for k in 1:n_mortar
            u_own_k = SVector(ntuple(v -> u_mortar_own[v, k], Val(nvars)))
            u_other_k = SVector(ntuple(v -> u_mortar_other[v, k], Val(nvars)))
            if entropy_variables
                u_own_k = entropy2cons(u_own_k, equations)
                u_other_k = entropy2cons(u_other_k, equations)
            end
            normal_k = SVector(ntuple(d -> ja_mortar[d, k], Val(2)))

            if own_is_left
                flux = surface_flux(u_own_k, u_other_k, normal_k, equations)
            else
                flux = surface_flux(u_other_k, u_own_k, normal_k, equations)
            end

            for v in eachvariable(equations)
                f_mortar[v, k] = flux[v]
            end
        end

        # L²-project mortar flux onto this element's original face quadrature
        for node in 1:n_own
            for v in eachvariable(equations)
                acc = zero(eltype(flux_projected))
                for k in 1:n_mortar
                    acc += mortar.project_own[node, k] * f_mortar[v, k]
                end
                flux_projected[v, node, cell] = acc
            end
        end
    end

    return nothing
end

@inline function own_face_indices(other_orientation, own_is_left)
    if other_orientation == 1
        return own_is_left ? (:end, :i_forward) : (:begin, :i_forward)
    else
        return own_is_left ? (:i_forward, :end) : (:i_forward, :begin)
    end
end

function find_semi_index(semi, semi_coupled::SemidiscretizationCoupled)
    for i in eachsystem(semi_coupled)
        if semi_coupled.semis[i] === semi
            return i
        end
    end
    error("semidiscretization not found in SemidiscretizationCoupled")
end

################################################################################
### DGSEM/structured
################################################################################

@inline function calc_boundary_flux_by_direction!(surface_flux_values, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionCoupled,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::False,
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @unpack node_coordinates, contravariant_vectors, inverse_jacobian, interfaces_u = cache.elements
    # Boundary values are for `StructuredMesh` stored in the interface datastructure
    boundaries_u = interfaces_u
    @unpack surface_flux = surface_integral

    cell_indices = get_boundary_indices(element, orientation, mesh)

    u_inner = get_node_vars(boundaries_u, equations, dg, surface_node_indices...,
                            direction, element)

    # If the mapping is orientation-reversing, the contravariant vectors' orientation
    # is reversed as well. The normal vector must be oriented in the direction
    # from `left_element` to `right_element`, or the numerical flux will be computed
    # incorrectly (downwind direction).
    sign_jacobian = sign(inverse_jacobian[node_indices..., element])

    # Contravariant vector Ja^i is the normal vector
    normal = sign_jacobian *
             get_contravariant_vector(orientation, contravariant_vectors,
                                      node_indices..., element)

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * boundary_condition(u_inner, normal, direction, cell_indices,
                              surface_node_indices, surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
    end

    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionCoupled,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::True,
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @unpack node_coordinates, contravariant_vectors, inverse_jacobian, interfaces_u = cache.elements
    # Boundary values are for `StructuredMesh` stored in the interface datastructure
    boundaries_u = interfaces_u
    @unpack surface_flux = surface_integral

    cell_indices = get_boundary_indices(element, orientation, mesh)

    u_inner = get_node_vars(boundaries_u, equations, dg, surface_node_indices...,
                            direction, element)

    # If the mapping is orientation-reversing, the contravariant vectors' orientation
    # is reversed as well. The normal vector must be oriented in the direction
    # from `left_element` to `right_element`, or the numerical flux will be computed
    # incorrectly (downwind direction).
    sign_jacobian = sign(inverse_jacobian[node_indices..., element])

    # Contravariant vector Ja^i is the normal vector
    normal = sign_jacobian *
             get_contravariant_vector(orientation, contravariant_vectors,
                                      node_indices..., element)

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux, noncons_flux = boundary_condition(u_inner, normal, direction, cell_indices,
                                            surface_node_indices, surface_flux,
                                            equations)

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = sign_jacobian *
                                                                              (flux[v] +
                                                                               0.5f0 *
                                                                               noncons_flux[v])
    end

    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionCoupledPMortar,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::False,
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @unpack inverse_jacobian = cache.elements
    cell_indices = get_boundary_indices(element, orientation, mesh)
    sign_jacobian = sign(inverse_jacobian[node_indices..., element])

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = sign_jacobian *
                                                                              boundary_condition.flux_projected[v,
                                                                                                                surface_node_indices...,
                                                                                                                cell_indices...]
    end

    return nothing
end

function get_boundary_indices(element, orientation,
                              mesh::Union{StructuredMesh{2}, StructuredMeshView{2}})
    cartesian_indices = CartesianIndices(size(mesh))
    if orientation == 1
        # Get index of element in y-direction
        cell_indices = (cartesian_indices[element][2],)
    else # orientation == 2
        # Get index of element in x-direction
        cell_indices = (cartesian_indices[element][1],)
    end

    return cell_indices
end

################################################################################
### Special elixirs
################################################################################

# Analyze convergence for SemidiscretizationCoupled
function analyze_convergence(errors_coupled, iterations,
                             semi_coupled::SemidiscretizationCoupled)
    # Extract errors: the errors are currently stored as
    # | iter 1 sys 1 var 1...n | iter 1 sys 2 var 1...n | ... | iter 2 sys 1 var 1...n | ...
    # but for calling `analyze_convergence` below, we need the following layout
    # sys n: | iter 1 var 1...n | iter 1 var 1...n | ... | iter 2 var 1...n | ...
    # That is, we need to extract and join the data for a single system
    errors = Dict{Symbol, Vector{Float64}}[]
    for i in eachsystem(semi_coupled)
        push!(errors, Dict(:l2 => Float64[], :linf => Float64[]))
    end
    offset = 0
    for iter in 1:iterations, i in eachsystem(semi_coupled)
        # Extract information on current semi
        semi = semi_coupled.semis[i]
        _, equations, _, _ = mesh_equations_solver_cache(semi)
        variablenames = varnames(cons2cons, equations)

        # Compute offset
        first = offset + 1
        last = offset + length(variablenames)
        offset += length(variablenames)

        # Append errors to appropriate storage
        append!(errors[i][:l2], errors_coupled[:l2][first:last])
        append!(errors[i][:linf], errors_coupled[:linf][first:last])
    end

    eocs = Vector{Dict{Symbol, Any}}(undef, nsystems(semi_coupled))
    errorsmatrix = Vector{Dict{Symbol, Matrix{Float64}}}(undef, nsystems(semi_coupled))
    for i in eachsystem(semi_coupled)
        # Use visual cues to separate output from multiple systems
        println()
        println("="^100)
        println("# System $i")
        println("="^100)

        # Extract information on current semi
        semi = semi_coupled.semis[i]
        _, equations, _, _ = mesh_equations_solver_cache(semi)
        variablenames = varnames(cons2cons, equations)

        eocs[i], errorsmatrix[i] = analyze_convergence(errors[i], iterations,
                                                       variablenames)
    end

    return eocs, errorsmatrix
end
end # @muladd
