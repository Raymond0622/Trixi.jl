using Trixi
using StartUpDG

rd = RefElemData(Line(), SBP(), 3)
f(x) = x^4;
nodes = f.(rd.r)
sum(rd.M * nodes)

u_nodes = f.((rd.r .+ 1)./2)
l_nodes = f.((rd.r .- 1)/ 2)

sum(rd.M * u_nodes + rd.M * l_nodes) * 0.5

