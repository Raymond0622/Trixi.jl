# Dump mortar neighbor_ids + element boxes from live `semi` in Main.
using Trixi

isdefined(Main, :semi) || error("`semi` is not defined in Main")

s = Main.semi
mortars = s.cache.mortars
mesh = s.mesh
elems = s.cache.elements
n_el = Trixi.nelements(s.solver, s.cache)
n_m = Trixi.nmortars(mortars)
L0 = mesh.tree.length_level_0
ids = mortars.neighbor_ids

out = joinpath(@__DIR__, "mortar_neighbor_graph.json")
open(out, "w") do io
    print(io, "{\"n_elements\":", n_el, ",\"n_mortars\":", n_m, ",\"elements\":[")
    for e in 1:n_el
        e > 1 && print(io, ",")
        cell = elems.cell_ids[e]
        x, y = Trixi.cell_coordinates(mesh.tree, cell)
        lev = Int(mesh.tree.levels[cell])
        h = L0 / 2^lev
        print(io, "{\"id\":", e, ",\"x\":", x, ",\"y\":", y, ",\"h\":", h, ",\"level\":", lev, "}")
    end
    print(io, "],\"mortars\":[")
    for m in 1:n_m
        m > 1 && print(io, ",")
        print(io, "{\"mortar\":", m,
              ",\"lower\":", ids[1, m],
              ",\"upper\":", ids[2, m],
              ",\"large\":", ids[3, m],
              ",\"large_sides\":", mortars.large_sides[m],
              ",\"orientation\":", mortars.orientations[m], "}")
    end
    print(io, "]}")
end
println("wrote ", out, "  nelements=", n_el, "  nmortars=", n_m)
