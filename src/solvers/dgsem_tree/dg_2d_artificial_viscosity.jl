@muladd begin
    function create_cache(mesh, artificial_viscosity::EntropyCorrectionArtificialViscosity,
                          dg::DG, cache, RealT, uEltype)
        
        coefficients = zeros(real(dg), nelements(dg, cache))
        svv_coefficients = zeros(real(dg), nelements(dg, cache))
        norm_residuals = zero(real(dg))

        ## create element filtered flux and transform gradients
        ## this does not create the whole mesh variables to save memory
        ## we reuse memory
        n_nodes = nnodes(dg)
        n_vars = nvariables(artificial_viscosity.equations_artificial_viscosity)
        n_elements = nelements(dg, cache)

        sensor = Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)
        max_coeff = Float64[]
        # Bit i (0-based) marks hanging face direction i+1 of that element.
        hanging_faces = zeros(UInt8, n_elements)
        #velocity_data = Array{uEltype, 5}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)
        cache = (; sensor, max_coeff, coefficients, svv_coefficients, norm_residuals,
                 hanging_faces)
        return cache
    end

    @inline function get_mortar_node_vars(u_mortar, equations, dg, leftright, i, mortar)
        return SVector(ntuple(@inline(v -> u_mortar[leftright, v, i, mortar]),
                              Val(nvariables(equations))))
    end

    # Quadrature weights at the nodes where `prolong2mortars!` stores traces / Riemann states.
    @inline function mortar_quadrature_weights(mortar, dg)
        if mortar_nodes(mortar) isa Val{:gauss}
            _, weights = gauss_nodes_weights(nnodes(dg), real(dg))
            return weights
        else
            return dg.basis.weights
        end
    end

    @inline function get_face_node_vars(u, equations, dg, element, direction, k)
        if direction == 1
            return get_node_vars(u, equations, dg, 1, k, element)
        elseif direction == 2
            return get_node_vars(u, equations, dg, nnodes(dg), k, element)
        elseif direction == 3
            return get_node_vars(u, equations, dg, k, 1, element)
        else
            return get_node_vars(u, equations, dg, k, nnodes(dg), element)
        end
    end

    function calc_volume_entropy_residual(du, u, element, mesh::TreeMesh{2}, equations, dg,
                                          cache, hanging_faces)

        # calculate volume integral
        volume_integral_du_entropy = zero(real(dg))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            du_node = get_node_vars(du, equations, dg, i, j, element)
            weight_ij = dg.basis.weights[i] * dg.basis.weights[j]

            # calc integral(-dv/dx_i * f(u)) -> missing factor of J
            volume_integral_du_entropy = volume_integral_du_entropy +
                                         dot(cons2entropy(u_node, equations), du_node) *
                                         weight_ij
        end

        # Conforming (and boundary) faces: LGL quadrature of ψ(u), matching LGL SAT.
        # Hanging faces are omitted here and added with mortar quadrature in
        # `add_mortar_entropy_potential!`.
        surface_integral_entropy_potential = zero(real(dg))
        for direction in (1, 2, 3, 4)
            is_hanging_face(hanging_faces, element, direction) && continue
            normal = if direction == 1
                SVector(-1.0f0, 0.0f0)
            elseif direction == 2
                SVector(1.0f0, 0.0f0)
            elseif direction == 3
                SVector(0.0f0, -1.0f0)
            else
                SVector(0.0f0, 1.0f0)
            end
            for ii in eachnode(dg)
                u_node = get_face_node_vars(u, equations, dg, element, direction, ii)
                surface_integral_entropy_potential = surface_integral_entropy_potential +
                                                     dg.basis.weights[ii] *
                                                     entropy_potential(u_node, normal,
                                                                       equations)
            end
        end

        # by default, the volume_integral contribution to du does not scale by any geometric terms
        # For TreeMesh, these geometric terms are ds/dx = 0 and dr/dx * J = 0.5 * h. Thus, to calculate 
        # the volume integral over the physical element, we need to scale by the 1D Jacobian. Similarly,
        # the surface integrals should be scaled by the 1D Jacobian as well. 
        jacobian_1d = inv(cache.elements.inverse_jacobian[element]) # O(h) 
        return (volume_integral_du_entropy + surface_integral_entropy_potential) *
               jacobian_1d
    end

    # ∫ ψ n dS on hanging faces, using the same mortar traces and weights as f*.
    # Large face: composite half-interval quadrature (factor 1/2). Do not reverse-project ψ.
    function add_mortar_entropy_potential!(entropy_residual, mesh::TreeMesh{2},
                                           equations, dg, cache)
        mortar = dg.mortar
        weights = mortar_quadrature_weights(mortar, dg)
        mortars = cache.mortars
        inverse_jacobian = cache.elements.inverse_jacobian

        for mortar_id in eachmortar(dg, cache)
            large_element = mortars.neighbor_ids[3, mortar_id]
            upper_element = mortars.neighbor_ids[2, mortar_id]
            lower_element = mortars.neighbor_ids[1, mortar_id]
            orientation = mortars.orientations[mortar_id]
            large_sides = mortars.large_sides[mortar_id]

            # large_sides = 1 means the large face on the left, 2 means on the right
            if large_sides == 1
                normal_large = orientation == 1 ? SVector(1.0f0, 0.0f0) :
                               SVector(0.0f0, 1.0f0)
                leftright_large = 1
                leftright_small = 2
            else
                normal_large = orientation == 1 ? SVector(-1.0f0, 0.0f0) :
                               SVector(0.0f0, -1.0f0)
                leftright_large = 2
                leftright_small = 1
            end
            normal_small = -normal_large

            psi_large = zero(eltype(entropy_residual))
            psi_upper = zero(eltype(entropy_residual))
            psi_lower = zero(eltype(entropy_residual))
            for j in eachnode(dg)
                u_upper_large = get_mortar_node_vars(mortars.u_upper, equations, dg,
                                                     leftright_large, j, mortar_id)
                u_lower_large = get_mortar_node_vars(mortars.u_lower, equations, dg,
                                                     leftright_large, j, mortar_id)
                u_upper_small = get_mortar_node_vars(mortars.u_upper, equations, dg,
                                                     leftright_small, j, mortar_id)
                u_lower_small = get_mortar_node_vars(mortars.u_lower, equations, dg,
                                                     leftright_small, j, mortar_id)
                psi_large = psi_large +
                            weights[j] *
                            (entropy_potential(u_upper_large, normal_large, equations) +
                             entropy_potential(u_lower_large, normal_large, equations))
                psi_upper = psi_upper +
                            weights[j] *
                            entropy_potential(u_upper_small, normal_small, equations)
                psi_lower = psi_lower +
                            weights[j] *
                            entropy_potential(u_lower_small, normal_small, equations)
            end

            # Half-interval map on the large face: ds_large = (1/2) dη per small mortar.
            entropy_residual[large_element] = entropy_residual[large_element] +
                                              0.5f0 * psi_large *
                                              inv(inverse_jacobian[large_element])
            entropy_residual[upper_element] = entropy_residual[upper_element] +
                                              psi_upper *
                                              inv(inverse_jacobian[upper_element])
            entropy_residual[lower_element] = entropy_residual[lower_element] +
                                              psi_lower *
                                              inv(inverse_jacobian[lower_element])
        end

        return nothing
    end

    function calc_entropy_residuals!(entropy_residual, du, u, mesh::TreeMesh{2},
                                     equations, dg, cache)
        hanging_faces = cache.artificial_viscosity.hanging_faces
        n_elements = nelements(dg, cache)
        if length(hanging_faces) != n_elements
            resize!(hanging_faces, n_elements)
        end

        mortar = dg.mortar
        has_mortars = mortar isa Union{LobattoLegendreMortarL2,
                                       LobattoLegendreMortarEntropy} &&
                      !isempty(eachmortar(dg, cache))
        if has_mortars
            mark_hanging_faces!(hanging_faces, dg, cache)
        else
            fill!(hanging_faces, 0x00)
        end

        @threaded for element in eachelement(dg, cache)
            entropy_residual[element] = calc_volume_entropy_residual(du, u, element,
                                                                     mesh, equations,
                                                                     dg, cache,
                                                                     hanging_faces)
        end
        if has_mortars
            add_mortar_entropy_potential!(entropy_residual, mesh, equations, dg, cache)
        end

        return nothing
    end

    function prolong2mortars_for_entropy_residual!(cache, u, mesh, equations, dg)
        mortar = dg.mortar
        if mortar isa Union{LobattoLegendreMortarL2, LobattoLegendreMortarEntropy} &&
           !isempty(eachmortar(dg, cache))
            prolong2mortars!(cache, u, mesh, equations, mortar, dg)
        end
        return nothing
    end

    function calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual,
                                     equations, mesh::TreeMesh{2}, dg, cache)
        element_v = 0.0;
        for element in eachelement(dg, cache)
            volume_jacobian_ = volume_jacobian(element, mesh, cache)

            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))
            for j in eachnode(dg), i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic[1], equations, dg, i,
                                                      j,
                                                      element)
                flux_parabolic_y_node = get_node_vars(flux_parabolic[2], equations, dg, i,
                                                      j,
                                                      element)
                gradients_x_node = get_node_vars(gradients[1], equations, dg, i, j, element)
                gradients_y_node = get_node_vars(gradients[2], equations, dg, i, j, element)
                viscous_dissipation_x = dot(flux_parabolic_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_parabolic_y_node, gradients_y_node)

                weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x +
                                               viscous_dissipation_y) * weight_ij *
                                              volume_jacobian_
            end

            # Scale viscous flux by ecav coefficient.
            # Note: we usually use "-min(0, entropy_residual)" to define the ECAV coefficient, but we
            # flip the sign to account for the fact that viscous terms are negated by convention in Trixi.jl.
            ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]),
                                                 element_viscous_dissipation)
            ecav_coefficient = entropy_residual[element] /
                                                  element_viscous_dissipation
            #ecav_coefficient = 0.0
            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            for j in eachnode(dg), i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic[1], equations, dg, i,
                                                      j,
                                                      element)
                flux_parabolic_y_node = get_node_vars(flux_parabolic[2], equations, dg, i,
                                                      j,
                                                      element)
                set_node_vars!(flux_parabolic[1], ecav_coefficient * flux_parabolic_x_node,
                               equations, dg, i, j, element)
                set_node_vars!(flux_parabolic[2], ecav_coefficient * flux_parabolic_y_node,
                               equations, dg, i, j, element)
            end
            element_v = max(element_v, element_viscous_dissipation)
        end
        push!(cache.artificial_viscosity.max_coeff, maximum(cache.artificial_viscosity.coefficients))
        return nothing
    end

    function rhs_artificial_viscosity!(du, u, t, mesh::TreeMesh{2},
                                       equations, equations_parabolic,
                                       equations_artificial_viscosity,
                                       boundary_conditions, boundary_conditions_parabolic,
                                       source_terms::Source,
                                       dg::DG, solver_parabolic, cache,
                                       cache_parabolic) where {Source}
        backend = trixi_backend(u)

        # Reset du
        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            reset_du!(du, dg, cache)
        end

        # Calculate volume integral
        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        # Mortar traces are needed for hanging-face ψ quadrature in the residual.
        prolong2mortars_for_entropy_residual!(cache, u, mesh, equations, dg)

        # calculate entropy residual
        entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
        calc_entropy_residuals!(entropy_residual, du, u, mesh, equations, dg, cache)
        # Prolong solution to interfaces
        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate interface fluxes
        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        # Prolong solution to boundaries
        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate boundary fluxes
        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                                dg.surface_integral, dg)
        end

        # # Prolong solution to mortars
        # @trixi_timeit timer() "prolong2mortars" begin
        #     # prolong2mortars!(cache, u, mesh, equations, dg.mortar, dg)
        #     prolong_entropy_projection_2_mortars!(cache, u, mesh, equations, dg.mortar, dg)
        # end

        # # Calculate mortar fluxes
        # @trixi_timeit timer() "mortar flux" begin
        #     calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
        #                       have_nonconservative_terms(equations), equations,
        #                       dg.mortar, dg.surface_integral, dg, cache)
        # end

        # Calculate surface integrals
        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        # @trixi_timeit timer() "transform variables" begin
        #     (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
        #     transform_variables!(u_transformed, u, mesh, equations_artificial_viscosity, dg,
        #                          solver_parabolic, cache)
        # end

        @trixi_timeit timer() "calculate parabolic fluxes" begin
            (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations,
                                mesh,
                                dg, cache)

        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u,
                                                                 mesh,
                                                                 equations_parabolic,
                                                                 boundary_conditions_parabolic, # TODO: hacky pass in parabolic equations
                                                                 #  equations_artificial_viscosity, BoundaryConditionDoNothing(), 
                                                                 dg, solver_parabolic,
                                                                 cache, t)

        # Apply Jacobian from mapping to reference element
        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        # Calculate source terms
        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end

    function rhs_combined!(du, u, t, mesh::TreeMesh{2},
                           equations, equations_parabolic, equations_artificial_viscosity,
                           boundary_conditions, boundary_conditions_parabolic,
                           source_terms::Source,
                           dg::DG, parabolic_scheme, cache, cache_parabolic) where {Source}
        (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
        backend = trixi_backend(u)
        # Reset du
        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end

        # ========= hyperbolic part ============

        # Calculate volume integral
        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        # Mortar traces are needed for hanging-face ψ quadrature in the residual.
        @trixi_timeit_ext backend timer() "prolong2mortars+entropy_residual" begin
            prolong2mortars_for_entropy_residual!(cache, u, mesh, equations, dg)
        end

        # calculate entropy residual
        entropy_residual = cache.artificial_viscosity.coefficients # reuse storage
        calc_entropy_residuals!(entropy_residual, du, u, mesh, equations, dg, cache)
        #push!(cache.artificial_viscosity.max_coeff, maximum(-min.(0.0, entropy_residual)))

        # Prolong solution to interfaces
        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate interface fluxes
        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        # Prolong solution to boundaries
        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(backend, cache, u, mesh, equations, dg)
        end

        # Calculate boundary fluxes
        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                                dg.surface_integral, dg)
        end

        # Mortar traces were already filled before the entropy residual.

        # Calculate mortar fluxes
        @trixi_timeit timer() "mortar flux" begin
            calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
        end

        # Calculate surface integrals
        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        # ==== shared parabolic terms ====

        # Convert conservative variables to a form more suitable for viscous flux calculations
        @trixi_timeit timer() "transform variables" begin
            transform_variables!(u_transformed, u, mesh, equations_parabolic,
                                 dg, cache)
        end
        #push!(cache.artificial_viscosity.max_coeff, maximum(u_transformed))

        # Compute the gradients of the transformed variables
        @trixi_timeit timer() "calculate gradient" begin
            calc_gradient!(gradients, u_transformed, t, mesh,
                           equations_parabolic, boundary_conditions_parabolic,
                           dg, parabolic_scheme, cache)
        end
        #push!(cache.artificial_viscosity.max_coeff, maximum(gradients[1]))

        # ========= AV specific part ============

        @trixi_timeit timer() "calculate AV parabolic fluxes" begin
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual, equations,
                                mesh,
                                dg, cache)

        # # TODO: accumulate into flux_parabolic instead
        # # accumulate the AV term
        # @trixi_timeit timer() "calc AV divergence" calc_divergence!(du, flux_parabolic, u, mesh, 
        #                                                             equations_artificial_viscosity, 
        #                                                             boundaryConditionDoNothing(), 
        #                                                             dg, parabolic_scheme, cache, t)

        # ======== physical parabolic part ==========

        # accumulate physical viscous fluxes    
        @trixi_timeit timer() "calculate viscous fluxes" begin
            accum_viscous_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                  equations_parabolic, dg, cache)
        end

        # TODO: fix BCs for equations_artificial_viscosity
        @trixi_timeit timer() "calc divergence" calc_divergence!(du, flux_parabolic, u,
                                                                 mesh,
                                                                 equations_parabolic,
                                                                 boundary_conditions_parabolic,
                                                                 dg, parabolic_scheme,
                                                                 cache, t)

        # Apply Jacobian from mapping to reference element
        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        # Calculate source terms
        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end

    function accum_viscous_fluxes!(flux_viscous,
                                   gradients, u_transformed,
                                   mesh::Union{TreeMesh{2}, P4estMesh{2}},
                                   equations_parabolic::AbstractEquationsParabolic,
                                   dg::DG, cache)
        gradients_x, gradients_y = gradients
        flux_viscous_x, flux_viscous_y = flux_viscous # output arrays

        @threaded for element in eachelement(dg, cache)
            for j in eachnode(dg), i in eachnode(dg)
                # Get solution and gradients
                u_node = get_node_vars(u_transformed, equations_parabolic, dg,
                                       i, j, element)
                gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                                 i, j, element)
                gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                                 i, j, element)

                # Calculate viscous flux and store each component for later use
                flux_viscous_node_x = flux(u_node, (gradients_1_node, gradients_2_node), 1,
                                           equations_parabolic)
                flux_viscous_node_y = flux(u_node, (gradients_1_node, gradients_2_node), 2,
                                           equations_parabolic)
                # flip sign for Trixi's parabolic convention
                add_to_node_vars!(flux_viscous_x, -flux_viscous_node_x, equations_parabolic,
                                  dg,
                                  i, j, element)
                add_to_node_vars!(flux_viscous_y, -flux_viscous_node_y, equations_parabolic,
                                  dg,
                                  i, j, element)
            end
        end

        return nothing
    end
end # @muladd
