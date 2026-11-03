mutable struct ParabolicContainer2D{uEltype <: Real}
    # [variables, nodes, nodes, elements]
    u_transformed::Array{uEltype, 4}
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    gradients::NTuple{2, Array{uEltype, 4}}
    # ([variables, nodes, nodes, elements],
    #  [variables, nodes, nodes, elements])
    flux_parabolic::NTuple{2, Array{uEltype, 4}}
    # SVV work arrays (filtered gradients / parabolic fluxes and the
    # intermediate result after filtering in the first direction)
    filtered_gradients::NTuple{2, Array{uEltype, 4}}
    filtered_parabolic::NTuple{2, Array{uEltype, 4}}
    tmp_gradient::NTuple{2, Array{uEltype, 4}}
    tmp_parabolic::NTuple{2, Array{uEltype, 4}}

    # internal `resize!`able storage
    _u_transformed::Vector{uEltype}
    # Use Tuple for outer, fixed-size datastructure
    _gradients::Tuple{Vector{uEltype}, Vector{uEltype}}
    _flux_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}}
    _filtered_gradients::Tuple{Vector{uEltype}, Vector{uEltype}}
    _filtered_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}}
    _tmp_gradient::Tuple{Vector{uEltype}, Vector{uEltype}}
    _tmp_parabolic::Tuple{Vector{uEltype}, Vector{uEltype}}

    function ParabolicContainer2D{uEltype}(n_vars::Integer, n_nodes::Integer,
                                           n_elements::Integer) where {uEltype <: Real}
        new_arrays() = (Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements),
                        Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements))
        new_vectors() = (Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                         Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements))
        return new(Array{uEltype, 4}(undef, n_vars, n_nodes, n_nodes, n_elements), # `u_transformed`
                   new_arrays(), # `gradients`
                   new_arrays(), # `flux_parabolic`
                   new_arrays(), # `filtered_gradients`
                   new_arrays(), # `filtered_parabolic`
                   new_arrays(), # `tmp_gradient`
                   new_arrays(), # `tmp_parabolic`
                   # `_u_transformed`
                   Vector{uEltype}(undef, n_vars * n_nodes^2 * n_elements),
                   new_vectors(), # `_gradients`
                   new_vectors(), # `_flux_parabolic`
                   new_vectors(), # `_filtered_gradients`
                   new_vectors(), # `_filtered_parabolic`
                   new_vectors(), # `_tmp_gradient`
                   new_vectors()) # `_tmp_parabolic`
    end
end

function init_parabolic_container_2d(n_vars::Integer, n_nodes::Integer,
                                     n_elements::Integer,
                                     ::Type{uEltype}) where {uEltype <: Real}
    return ParabolicContainer2D{uEltype}(n_vars, n_nodes, n_elements)
end

# Only one-dimensional `Array`s are `resize!`able in Julia.
# Hence, we use `Vector`s as internal storage and `resize!`
# them whenever needed. Then, we reuse the same memory by
# `unsafe_wrap`ping multi-dimensional `Array`s around the
# internal storage.
function Base.resize!(parabolic_container::ParabolicContainer2D, equations, dg, cache)
    capacity = nvariables(equations) * nnodes(dg)^2 * nelements(dg, cache)
    resize!(parabolic_container._u_transformed, capacity)
    for dim in 1:2
        resize!(parabolic_container._gradients[dim], capacity)
        resize!(parabolic_container._flux_parabolic[dim], capacity)
        resize!(parabolic_container._filtered_gradients[dim], capacity)
        resize!(parabolic_container._filtered_parabolic[dim], capacity)
        resize!(parabolic_container._tmp_gradient[dim], capacity)
        resize!(parabolic_container._tmp_parabolic[dim], capacity)
    end

    parabolic_container.u_transformed = unsafe_wrap(Array,
                                                    pointer(parabolic_container._u_transformed),
                                                    (nvariables(equations),
                                                     nnodes(dg), nnodes(dg),
                                                     nelements(dg, cache)))

    gradients_1 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[1]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))
    gradients_2 = unsafe_wrap(Array,
                              pointer(parabolic_container._gradients[2]),
                              (nvariables(equations),
                               nnodes(dg), nnodes(dg),
                               nelements(dg, cache)))

    parabolic_container.gradients = (gradients_1, gradients_2)

    flux_parabolic_1 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[1]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))
    flux_parabolic_2 = unsafe_wrap(Array,
                                   pointer(parabolic_container._flux_parabolic[2]),
                                   (nvariables(equations),
                                    nnodes(dg), nnodes(dg),
                                    nelements(dg, cache)))

    parabolic_container.flux_parabolic = (flux_parabolic_1, flux_parabolic_2)

    wrap(storage) = (unsafe_wrap(Array, pointer(storage[1]),
                                 (nvariables(equations), nnodes(dg), nnodes(dg),
                                  nelements(dg, cache))),
                     unsafe_wrap(Array, pointer(storage[2]),
                                 (nvariables(equations), nnodes(dg), nnodes(dg),
                                  nelements(dg, cache))))
    parabolic_container.filtered_gradients = wrap(parabolic_container._filtered_gradients)
    parabolic_container.filtered_parabolic = wrap(parabolic_container._filtered_parabolic)
    parabolic_container.tmp_gradient = wrap(parabolic_container._tmp_gradient)
    parabolic_container.tmp_parabolic = wrap(parabolic_container._tmp_parabolic)

    return nothing
end
