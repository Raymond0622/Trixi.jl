@muladd begin
    function assert_no_mortars_for_ecav(mesh::P4estMesh{2}, dg::DG, cache)
        if !isempty(eachmortar(dg, cache))
            throw(ArgumentError("P4est ECAV currently supports conforming meshes only. Disable AMR or add mortar support before using nonconforming P4est meshes."))
        end

        return nothing
    end

    function calc_volume_entropy_residual(du, u, element, mesh::P4estMesh{2},
                                          equations, dg, cache)
        (; contravariant_vectors) = cache.elements

        # Volume contribution on the reference element. For P4est, the volume
        # operator already contains the contravariant metric terms.
        volume_integral_du_entropy = zero(real(dg))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            du_node = get_node_vars(du, equations, dg, i, j, element)
            weight_ij = dg.basis.weights[i] * dg.basis.weights[j]

            volume_integral_du_entropy += dot(cons2entropy(u_node, equations),
                                              du_node) * weight_ij
        end

        # Boundary entropy-potential contribution using the outward
        # J-scaled physical normals from the curved P4est geometry.
        surface_integral_entropy_potential = zero(real(dg))
        for l in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, 1, l, element)
            normal_direction = get_normal_direction(1, contravariant_vectors,
                                                    1, l, element)
            surface_integral_entropy_potential += dg.basis.weights[l] *
                                                  entropy_potential(u_node,
                                                                    normal_direction,
                                                                    equations)

            u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
            normal_direction = get_normal_direction(2, contravariant_vectors,
                                                    nnodes(dg), l, element)
            surface_integral_entropy_potential += dg.basis.weights[l] *
                                                  entropy_potential(u_node,
                                                                    normal_direction,
                                                                    equations)

            u_node = get_node_vars(u, equations, dg, l, 1, element)
            normal_direction = get_normal_direction(3, contravariant_vectors,
                                                    l, 1, element)
            surface_integral_entropy_potential += dg.basis.weights[l] *
                                                  entropy_potential(u_node,
                                                                    normal_direction,
                                                                    equations)

            u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
            normal_direction = get_normal_direction(4, contravariant_vectors,
                                                    l, nnodes(dg), element)
            surface_integral_entropy_potential += dg.basis.weights[l] *
                                                  entropy_potential(u_node,
                                                                    normal_direction,
                                                                    equations)
        end

        return volume_integral_du_entropy + surface_integral_entropy_potential
    end

    function calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual,
                                     equations, mesh::P4estMesh{2}, dg, cache)
        (; inverse_jacobian) = cache.elements

        for element in eachelement(dg, cache)
            element_viscous_dissipation = zero(real(dg))
            for j in eachnode(dg), i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic[1], equations, dg,
                                                      i, j, element)
                flux_parabolic_y_node = get_node_vars(flux_parabolic[2], equations, dg,
                                                      i, j, element)
                gradients_x_node = get_node_vars(gradients[1], equations, dg, i, j,
                                                 element)
                gradients_y_node = get_node_vars(gradients[2], equations, dg, i, j,
                                                 element)
                viscous_dissipation_x = dot(flux_parabolic_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_parabolic_y_node, gradients_y_node)

                volume_jacobian = abs(inv(inverse_jacobian[i, j, element]))
                weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
                element_viscous_dissipation += (viscous_dissipation_x +
                                                viscous_dissipation_y) *
                                               weight_ij * volume_jacobian
            end

            # Match the TreeMesh sign convention: viscous terms are negated by
            # convention in Trixi.jl, so the saved coefficient has opposite sign.
            ecav_coefficient = regularized_ratio(min(0, entropy_residual[element]),
                                                 element_viscous_dissipation)
            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient

            for j in eachnode(dg), i in eachnode(dg)
                flux_parabolic_x_node = get_node_vars(flux_parabolic[1], equations, dg,
                                                      i, j, element)
                flux_parabolic_y_node = get_node_vars(flux_parabolic[2], equations, dg,
                                                      i, j, element)
                set_node_vars!(flux_parabolic[1],
                               ecav_coefficient * flux_parabolic_x_node,
                               equations, dg, i, j, element)
                set_node_vars!(flux_parabolic[2],
                               ecav_coefficient * flux_parabolic_y_node,
                               equations, dg, i, j, element)
            end
        end

        push!(cache.artificial_viscosity.max_coeff,
              maximum(cache.artificial_viscosity.coefficients))
        return nothing
    end

    # Symmetric SVV on curved elements, see the `TreeMesh{2}` version. The
    # dissipation is measured in ⟨a, b⟩ = Σ w_i w_j |J_ij| a·b, so the adjoint of
    # F = F1D ⊗ F1D is |J|⁻¹ (F1D_star ⊗ F1D_star) |J| with the nodal |J|.
    # On `TreeMesh` J is constant per element and the |J| scalings cancel.
    function calc_ecav_svv_coefficients!(flux_parabolic, gradients, u_transformed,
                                         filtered_parabolic, filtered_gradients,
                                         tmp_gradient, tmp_parabolic,
                                         entropy_residual, equations,
                                         equations_artificial_viscosity,
                                         mesh::P4estMesh{2}, dg, cache)
        (; inverse_jacobian) = cache.elements
        flux_parabolic_x, flux_parabolic_y = flux_parabolic
        gradients_x, gradients_y = gradients
        filtered_gradients_x, filtered_gradients_y = filtered_gradients
        svv_flux_x, svv_flux_y = tmp_parabolic
        filtered_parabolic_x, filtered_parabolic_y = filtered_parabolic

        (; F1D, F1D_star) = cache
        @threaded for element in eachelement(dg, cache)
            # calculate viscous dissipation (ECAV denominator)
            element_viscous_dissipation = zero(real(dg))
            for j in eachnode(dg), i in eachnode(dg)
                flux_viscous_x_node = get_node_vars(flux_parabolic_x, equations, dg, i, j,
                                                    element)
                flux_viscous_y_node = get_node_vars(flux_parabolic_y, equations, dg, i, j,
                                                    element)
                gradients_x_node = get_node_vars(gradients_x, equations, dg, i, j, element)
                gradients_y_node = get_node_vars(gradients_y, equations, dg, i, j, element)

                viscous_dissipation_x = dot(flux_viscous_x_node, gradients_x_node)
                viscous_dissipation_y = dot(flux_viscous_y_node, gradients_y_node)

                volume_jacobian = abs(inv(inverse_jacobian[i, j, element]))
                weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
                element_viscous_dissipation = element_viscous_dissipation +
                                              (viscous_dissipation_x +
                                               viscous_dissipation_y) * weight_ij *
                                              volume_jacobian
            end

            # F∇v
            for dim in 1:2
                apply_tensor_operator!(filtered_gradients[dim], tmp_gradient[dim], F1D,
                                       gradients[dim], mesh, equations, dg, element)
            end

            # |J| K(u) F∇v and the SVV dissipation ⟨F∇v, K F∇v⟩
            svv_element_viscous_dissipation = zero(real(dg))
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations, dg, i, j, element)
                filtered_gradients_x_node = get_node_vars(filtered_gradients_x, equations,
                                                          dg, i, j, element)
                filtered_gradients_y_node = get_node_vars(filtered_gradients_y, equations,
                                                          dg, i, j, element)
                filtered_gradients_node = (filtered_gradients_x_node,
                                           filtered_gradients_y_node)
                svv_flux_x_node = flux(u_node, filtered_gradients_node, 1,
                                       equations_artificial_viscosity)
                svv_flux_y_node = flux(u_node, filtered_gradients_node, 2,
                                       equations_artificial_viscosity)

                volume_jacobian = abs(inv(inverse_jacobian[i, j, element]))
                set_node_vars!(svv_flux_x, volume_jacobian * svv_flux_x_node, equations,
                               dg, i, j, element)
                set_node_vars!(svv_flux_y, volume_jacobian * svv_flux_y_node, equations,
                               dg, i, j, element)

                svv_viscous_dissipation_x = dot(svv_flux_x_node, filtered_gradients_x_node)
                svv_viscous_dissipation_y = dot(svv_flux_y_node, filtered_gradients_y_node)

                weight_ij = dg.basis.weights[i] * dg.basis.weights[j]
                svv_element_viscous_dissipation = svv_element_viscous_dissipation +
                                                  (svv_viscous_dissipation_x +
                                                   svv_viscous_dissipation_y) *
                                                  weight_ij * volume_jacobian
            end

            # (F1D_star ⊗ F1D_star)(|J| K(u) F∇v); the |J|⁻¹ is applied below.
            # `tmp_gradient` is free again and serves as scratch.
            for dim in 1:2
                apply_tensor_operator!(filtered_parabolic[dim], tmp_gradient[dim], F1D_star,
                                       tmp_parabolic[dim], mesh, equations, dg, element)
            end

            # Match the TreeMesh sign convention: viscous terms are negated by
            # convention in Trixi.jl, so the saved coefficients have opposite sign.
            w = 1.0
            num = min(0, entropy_residual[element])
            denom = element_viscous_dissipation * element_viscous_dissipation +
                    w * svv_element_viscous_dissipation * svv_element_viscous_dissipation
            ecav_coefficient = (num * element_viscous_dissipation) / (denom + 1e-12)
            svv_coefficient = (w * num * svv_element_viscous_dissipation) / (denom + 1e-12)

            cache.artificial_viscosity.coefficients[element] = -ecav_coefficient # save output
            cache.artificial_viscosity.svv_coefficients[element] = -svv_coefficient # save output
            # Do not index `flux_parabolic` inside `@threaded`: the name shadows
            # `Trixi.flux_parabolic`, so Polyester would capture it in a closure.
            for j in eachnode(dg), i in eachnode(dg)
                svv_factor = svv_coefficient * abs(inverse_jacobian[i, j, element])
                multiply_multiply_add_to_node_vars!(flux_parabolic_x,
                                                    ecav_coefficient, svv_factor,
                                                    get_node_vars(filtered_parabolic_x,
                                                                  equations, dg, i, j,
                                                                  element),
                                                    equations, dg, i, j, element)
                multiply_multiply_add_to_node_vars!(flux_parabolic_y,
                                                    ecav_coefficient, svv_factor,
                                                    get_node_vars(filtered_parabolic_y,
                                                                  equations, dg, i, j,
                                                                  element),
                                                    equations, dg, i, j, element)
            end
        end

        push!(cache.artificial_viscosity.max_coeff,
              maximum(cache.artificial_viscosity.coefficients))
        return nothing
    end

    # Scale `flux_parabolic` by the ECAV (or ECAV + SVV) coefficients.
    function calc_av_coefficients!(flux_parabolic, gradients, u_transformed,
                                   entropy_residual, equations,
                                   equations_artificial_viscosity, mesh::P4estMesh{2},
                                   dg, cache, cache_parabolic)
        if cache.ecav_choice == :ecav_svv
            (; filtered_gradients, filtered_parabolic, tmp_gradient, tmp_parabolic) = cache_parabolic.parabolic_container
            calc_ecav_svv_coefficients!(flux_parabolic, gradients, u_transformed,
                                        filtered_parabolic, filtered_gradients,
                                        tmp_gradient, tmp_parabolic,
                                        entropy_residual, equations,
                                        equations_artificial_viscosity, mesh, dg, cache)
        else
            calc_ecav_coefficients!(flux_parabolic, gradients, entropy_residual,
                                    equations, mesh, dg, cache)
        end
        return nothing
    end

    function calc_divergence_ecav_p4est!(du, flux_parabolic, u, mesh::P4estMesh{2},
                                         equations_parabolic,
                                         boundary_conditions_parabolic, dg,
                                         parabolic_scheme, cache, t)
        assert_no_mortars_for_ecav(mesh, dg, cache)

        @trixi_timeit timer() "volume integral" begin
            calc_volume_integral!(du, flux_parabolic, mesh, equations_parabolic, dg,
                                  cache)
        end

        @trixi_timeit timer() "prolong2interfaces" begin
            prolong2interfaces!(cache, flux_parabolic, mesh, equations_parabolic, dg)
        end

        @trixi_timeit timer() "interface flux" begin
            calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 equations_parabolic, dg, parabolic_scheme, cache)
        end

        @trixi_timeit timer() "prolong2boundaries" begin
            prolong2boundaries!(cache, flux_parabolic, mesh, equations_parabolic, dg)
        end

        @trixi_timeit timer() "boundary flux" begin
            calc_boundary_flux_divergence!(cache, t, boundary_conditions_parabolic,
                                           mesh, equations_parabolic,
                                           dg.surface_integral, dg)
        end

        @trixi_timeit timer() "surface integral" begin
            calc_surface_integral!(nothing, du, u, mesh, equations_parabolic,
                                   dg.surface_integral, dg, cache)
        end

        return nothing
    end

    function rhs_artificial_viscosity!(du, u, t, mesh::P4estMesh{2},
                                       equations, equations_parabolic,
                                       equations_artificial_viscosity,
                                       boundary_conditions, boundary_conditions_parabolic,
                                       source_terms::Source,
                                       dg::DG, solver_parabolic, cache,
                                       cache_parabolic) where {Source}
        assert_no_mortars_for_ecav(mesh, dg, cache)

        backend = trixi_backend(u)
        (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container

        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end

        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        entropy_residual = cache.artificial_viscosity.coefficients
        @threaded for element in eachelement(dg, cache)
            entropy_residual[element] = calc_volume_entropy_residual(du, u, element,
                                                                     mesh, equations,
                                                                     dg, cache)
        end

        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(backend, cache, u, mesh, equations, dg)
        end

        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(backend, cache, u, mesh, equations, dg)
        end

        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                                dg.surface_integral, dg)
        end

        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        @trixi_timeit timer() "transform variables" begin
            transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
        end

        @trixi_timeit timer() "calculate gradient" begin
            calc_gradient!(backend, gradients, u_transformed, t, mesh,
                           equations_parabolic, boundary_conditions_parabolic,
                           dg, solver_parabolic, cache)
        end

        @trixi_timeit timer() "calculate parabolic fluxes" begin
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_av_coefficients!(flux_parabolic, gradients, u_transformed, entropy_residual,
                              equations, equations_artificial_viscosity, mesh, dg, cache,
                              cache_parabolic)

        @trixi_timeit timer() "calc divergence" begin
            calc_divergence_ecav_p4est!(du, flux_parabolic, u, mesh,
                                        equations_parabolic,
                                        boundary_conditions_parabolic, dg,
                                        solver_parabolic, cache, t)
        end

        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end

    function rhs_combined!(du, u, t, mesh::P4estMesh{2},
                           equations, equations_parabolic, equations_artificial_viscosity,
                           boundary_conditions, boundary_conditions_parabolic,
                           source_terms::Source,
                           dg::DG, parabolic_scheme, cache, cache_parabolic) where {Source}
        assert_no_mortars_for_ecav(mesh, dg, cache)

        (; u_transformed, flux_parabolic, gradients) = cache_parabolic.parabolic_container
        backend = trixi_backend(u)

        @trixi_timeit_ext backend timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end

        @trixi_timeit_ext backend timer() "volume integral" begin
            calc_volume_integral!(backend, du, u, mesh,
                                  have_nonconservative_terms(equations), equations,
                                  dg.volume_integral, dg, cache)
        end

        entropy_residual = cache.artificial_viscosity.coefficients
        @threaded for element in eachelement(dg, cache)
            entropy_residual[element] = calc_volume_entropy_residual(du, u, element,
                                                                     mesh, equations,
                                                                     dg, cache)
        end

        @trixi_timeit_ext backend timer() "prolong2interfaces" begin
            prolong2interfaces!(backend, cache, u, mesh, equations, dg)
        end

        @trixi_timeit_ext backend timer() "interface flux" begin
            calc_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
        end

        @trixi_timeit_ext backend timer() "prolong2boundaries" begin
            prolong2boundaries!(backend, cache, u, mesh, equations, dg)
        end

        @trixi_timeit_ext backend timer() "boundary flux" begin
            calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                                dg.surface_integral, dg)
        end

        @trixi_timeit_ext backend timer() "surface integral" begin
            calc_surface_integral!(backend, du, u, mesh, equations,
                                   dg.surface_integral, dg, cache)
        end

        @trixi_timeit timer() "transform variables" begin
            transform_variables!(u_transformed, u, mesh, equations_parabolic,
                                 dg, cache)
        end

        @trixi_timeit timer() "calculate gradient" begin
            calc_gradient!(backend, gradients, u_transformed, t, mesh,
                           equations_parabolic, boundary_conditions_parabolic,
                           dg, parabolic_scheme, cache)
        end

        @trixi_timeit timer() "calculate AV parabolic fluxes" begin
            calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                   equations_artificial_viscosity, dg, cache)
        end

        calc_av_coefficients!(flux_parabolic, gradients, u_transformed, entropy_residual,
                              equations, equations_artificial_viscosity, mesh, dg, cache,
                              cache_parabolic)

        @trixi_timeit timer() "calculate viscous fluxes" begin
            accum_viscous_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                  equations_parabolic, dg, cache)
        end

        @trixi_timeit timer() "calc divergence" begin
            calc_divergence_ecav_p4est!(du, flux_parabolic, u, mesh,
                                        equations_parabolic,
                                        boundary_conditions_parabolic, dg,
                                        parabolic_scheme, cache, t)
        end

        @trixi_timeit_ext backend timer() "Jacobian" begin
            apply_jacobian!(backend, du, mesh, equations, dg, cache)
        end

        @trixi_timeit_ext backend timer() "source terms" begin
            calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)
        end

        return nothing
    end
end # @muladd
