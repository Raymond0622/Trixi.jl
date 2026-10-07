using Plots
using Trixi

function mortar_face_directions(large_sides, orientation)
    if large_sides == 1
        return orientation == 1 ? (2, 1) : (4, 3)
    else
        return orientation == 1 ? (1, 2) : (3, 4)
    end
end

function face_segment(xmin, xmax, ymin, ymax, direction)
    if direction == 1
        return (xmin, ymin), (xmin, ymax)
    elseif direction == 2
        return (xmax, ymin), (xmax, ymax)
    elseif direction == 3
        return (xmin, ymin), (xmax, ymin)
    else
        return (xmin, ymax), (xmax, ymax)
    end
end

function on_domain_boundary(a, b; tol = 1e-12)
    # Whole edge on the outer square, not merely a corner touching ±1.
    same(x, y, val) = abs(x - val) < tol && abs(y - val) < tol
    return same(a[1], b[1], -1) || same(a[1], b[1], 1) ||
           same(a[2], b[2], -1) || same(a[2], b[2], 1)
end

function build_example_semi()
    equations = CompressibleEulerEquations2D(1.4)
    basis = LobattoLegendreBasis(3)
    solver = DGSEM(basis, flux_lax_friedrichs, VolumeIntegralWeakForm(),
                   MortarL2(basis))
    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = 1,
                    n_cells_max = 10_000, periodicity = true)
    level0 = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
    dx0 = (coordinates_max[1] - coordinates_min[1]) / 2^level0
    cells_to_refine = Int[]
    for cell_id in Trixi.leaf_cells(mesh.tree)
        x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
        ix = round(Int, (x - coordinates_min[1]) / dx0 - 0.5)
        iy = round(Int, (y - coordinates_min[2]) / dx0 - 0.5)
        if iseven(ix + iy)
            push!(cells_to_refine, cell_id)
        end
    end
    Trixi.refine!(mesh.tree, cells_to_refine)
    semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_density_wave, solver;
                                        boundary_conditions = boundary_condition_periodic)
    return semi
end

function cell_box(semi, e)
    mesh = semi.mesh
    cache = semi.cache
    cell = cache.elements.cell_ids[e]
    x, y = Trixi.cell_coordinates(mesh.tree, cell)
    h = 2.0 / 2^mesh.tree.levels[cell]
    return (x - h / 2, x + h / 2, y - h / 2, y + h / 2, Int(mesh.tree.levels[cell]))
end

function hanging_bits(semi)
    cache = semi.cache
    dg = semi.solver
    mortars = cache.mortars
    hanging = zeros(UInt8, nelements(dg, cache))
    for m in Trixi.eachmortar(dg, cache)
        large_dir, small_dir = mortar_face_directions(mortars.large_sides[m],
                                                      mortars.orientations[m])
        hanging[mortars.neighbor_ids[3, m]] |= UInt8(0x01) << (large_dir - 1)
        hanging[mortars.neighbor_ids[2, m]] |= UInt8(0x01) << (small_dir - 1)
        hanging[mortars.neighbor_ids[1, m]] |= UInt8(0x01) << (small_dir - 1)
    end
    return hanging
end

function plot_hanging_faces(semi, hanging)
    ne = nelements(semi.solver, semi.cache)
    default(fontfamily = "Computer Modern", linewidth = 1.2,
            legendfontsize = 10, tickfontsize = 11, guidefontsize = 12,
            titlefontsize = 13)

    plt = plot(xlims = (-1.22, 1.22), ylims = (-1.22, 1.58),
               aspect_ratio = :equal, legend = :top,
               xlabel = "x", ylabel = "y",
               title = "Checkerboard hanging faces (10 elements)",
               size = (740, 800), legend_columns = 2,
               grid = false, framestyle = :box,
               xticks = -1:0.5:1, yticks = -1:0.5:1)

    drew_coarse = false
    drew_fine = false
    for e in 1:ne
        xmin, xmax, ymin, ymax, lev = cell_box(semi, e)
        fillc = lev == 1 ? RGB(0.96, 0.90, 0.78) : RGB(0.86, 0.92, 0.98)
        lab = ""
        if lev == 1 && !drew_coarse
            lab = "coarse (level 1)"
            drew_coarse = true
        elseif lev == 2 && !drew_fine
            lab = "fine (level 2)"
            drew_fine = true
        end
        plot!(plt, Shape([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]);
              fillcolor = fillc, linecolor = RGB(0.55, 0.55, 0.55),
              linewidth = 0.5, label = lab)
    end

    drew_hang = false
    drew_wrap = false
    drew_conf = false
    for e in 1:ne
        xmin, xmax, ymin, ymax, _ = cell_box(semi, e)
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        annotate!(plt, xc, yc, text("e$(e)\n$(Int(hanging[e]))", 9, :black, :center))
        for d in 1:4
            a, b = face_segment(xmin, xmax, ymin, ymax, d)
            is_h = (hanging[e] & (UInt8(0x01) << (d - 1))) != 0
            if is_h
                wrap = on_domain_boundary(a, b)
                if wrap
                    lab = drew_wrap ? "" : "hanging (periodic wrap)"
                    drew_wrap = true
                    plot!(plt, [a[1], b[1]], [a[2], b[2]];
                          color = RGB(0.75, 0.22, 0.14), linewidth = 3.2,
                          linestyle = :dash, label = lab)
                else
                    lab = drew_hang ? "" : "hanging (2:1 mortar)"
                    drew_hang = true
                    plot!(plt, [a[1], b[1]], [a[2], b[2]];
                          color = RGB(0.75, 0.22, 0.14), linewidth = 3.2,
                          label = lab)
                end
            else
                lab = drew_conf ? "" : "conforming 1:1 (sibling)"
                drew_conf = true
                plot!(plt, [a[1], b[1]], [a[2], b[2]];
                      color = RGB(0.15, 0.32, 0.55), linewidth = 2.0, label = lab)
            end
        end
    end

    annotate!(plt, 0.0, 1.44,
              text("cell label: element id and hanging_faces UInt8   bits −x +x −y +y",
                   9, RGB(0.25, 0.25, 0.25)))
    return plt
end

function svg_escape(s)
    return replace(replace(s, "&" => "&amp;"), "<" => "&lt;")
end

function write_svg(path, semi, hanging)
    ne = nelements(semi.solver, semi.cache)
    # Math coords [-1.25,1.25]×[-1.25,1.7] → SVG, y-up.
    W, H = 780.0, 860.0
    mx, Mx = -1.48, 1.28
    my, My = -1.42, 1.78
    xpix(x) = (x - mx) / (Mx - mx) * W
    ypix(y) = H - (y - my) / (My - my) * H

    io = IOBuffer()
    println(io, """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="$(W)" height="$(H)" viewBox="0 0 $(W) $(H)">
  <style>
    .title { font: 600 20px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #1a1a1a; }
    .sub { font: 13px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #444; }
    .lab { font: 600 13px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #111; text-anchor: middle; }
    .bits { font: 11px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #333; text-anchor: middle; }
    .axis { font: 12px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #333; text-anchor: middle; }
    .leg { font: 13px "Source Sans 3", "Helvetica Neue", sans-serif; fill: #222; }
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="$(W/2)" y="28" text-anchor="middle">Checkerboard hanging faces</text>
  <text class="sub" x="$(W/2)" y="48" text-anchor="middle">10 elements · 8 mortars · cell label is id and hanging_faces UInt8 (bits −x +x −y +y)</text>
""")

    # cells
    for e in 1:ne
        xmin, xmax, ymin, ymax, lev = cell_box(semi, e)
        fill = lev == 1 ? "#f4e6c8" : "#d7e6f5"
        x = xpix(xmin)
        y = ypix(ymax)
        w = xpix(xmax) - xpix(xmin)
        h = ypix(ymin) - ypix(ymax)
        println(io, """  <rect x="$(round(x; digits=2))" y="$(round(y; digits=2))" width="$(round(w; digits=2))" height="$(round(h; digits=2))" fill="$fill" stroke="#8a8a8a" stroke-width="0.8"/>""")
    end

    function edge_line(a, b; color, width, dash = nothing)
        dashattr = dash === nothing ? "" : " stroke-dasharray=\"$dash\""
        println(io, """  <line x1="$(round(xpix(a[1]); digits=2))" y1="$(round(ypix(a[2]); digits=2))" x2="$(round(xpix(b[1]); digits=2))" y2="$(round(ypix(b[2]); digits=2))" stroke="$color" stroke-width="$width" stroke-linecap="square"$dashattr/>""")
    end

    # conforming then hanging so hanging is on top
    for e in 1:ne
        xmin, xmax, ymin, ymax, _ = cell_box(semi, e)
        for d in 1:4
            is_h = (hanging[e] & (UInt8(0x01) << (d - 1))) != 0
            is_h && continue
            a, b = face_segment(xmin, xmax, ymin, ymax, d)
            edge_line(a, b; color = "#244a7a", width = 2.6)
        end
    end
    for e in 1:ne
        xmin, xmax, ymin, ymax, _ = cell_box(semi, e)
        for d in 1:4
            is_h = (hanging[e] & (UInt8(0x01) << (d - 1))) != 0
            is_h || continue
            a, b = face_segment(xmin, xmax, ymin, ymax, d)
            wrap = on_domain_boundary(a, b)
            edge_line(a, b; color = "#b3261e", width = 3.6,
                      dash = wrap ? "7 5" : nothing)
        end
    end

    for e in 1:ne
        xmin, xmax, ymin, ymax, _ = cell_box(semi, e)
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        println(io, """  <text class="lab" x="$(round(xpix(xc); digits=2))" y="$(round(ypix(yc) - 4; digits=2))">e$e</text>""")
        println(io, """  <text class="bits" x="$(round(xpix(xc); digits=2))" y="$(round(ypix(yc) + 14; digits=2))">$(Int(hanging[e]))</text>""")
    end

    # axes ticks
    for xt in -1.0:0.5:1.0
        println(io, """  <text class="axis" x="$(round(xpix(xt); digits=2))" y="$(round(ypix(-1.18); digits=2))">$xt</text>""")
    end
    for yt in -1.0:0.5:1.0
        println(io, """  <text class="axis" x="$(round(xpix(-1.18); digits=2))" y="$(round(ypix(yt) + 4; digits=2))" text-anchor="end">$yt</text>""")
    end
    println(io, """  <text class="axis" x="$(round(xpix(0); digits=2))" y="$(H - 18)">x</text>""")
    println(io, """  <text class="axis" x="22" y="$(round(ypix(0) + 4; digits=2))">y</text>""")

    println(io, """  <rect x="70" y="64" width="18" height="14" fill="#d7e6f5" stroke="#8a8a8a"/>
  <text class="leg" x="94" y="76">fine (level 2)</text>
  <rect x="210" y="64" width="18" height="14" fill="#f4e6c8" stroke="#8a8a8a"/>
  <text class="leg" x="234" y="76">coarse (level 1)</text>
  <line x1="380" y1="71" x2="418" y2="71" stroke="#b3261e" stroke-width="4"/>
  <text class="leg" x="426" y="76">hanging 2:1</text>
  <line x1="70" y1="98" x2="108" y2="98" stroke="#b3261e" stroke-width="4" stroke-dasharray="7 5"/>
  <text class="leg" x="116" y="103">periodic hanging</text>
  <line x1="280" y1="98" x2="318" y2="98" stroke="#244a7a" stroke-width="3"/>
  <text class="leg" x="326" y="103">conforming 1:1 sibling</text>""")

    println(io, "</svg>")
    write(path, String(take!(io)))
    return path
end

semi = build_example_semi()
hanging = hanging_bits(semi)
outdir = @__DIR__
svg = joinpath(outdir, "hanging_faces_checkerboard.svg")
pdf = joinpath(outdir, "hanging_faces_checkerboard.pdf")
write_svg(svg, semi, hanging)
plt = plot_hanging_faces(semi, hanging)
savefig(plt, pdf)
println("wrote ", svg)
println("wrote ", pdf)
