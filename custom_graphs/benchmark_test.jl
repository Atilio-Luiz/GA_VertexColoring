using Graphs
using BenchmarkTools

include("CustomGraphs.jl")

#=
g = SimpleGraph()

for v in 1:10_000
    add_vertex!(g)
end

for i in 1:9_999
    add_edge!(g, i, i+1)
end

println("=== Benchmark 01 ===")

@btime nv($g)
@btime ne($g)
@btime degree($g, 5000)
@btime collect(neighbors($g, 5000))
@btime has_edge($g, 100, 101)
@btime connected_components($g)

#----------------------------------------------
CG = CustomGraph{Int}()

for v in 1:10_000
    cg_add_vertex!(CG, v)
end

for i in 1:9_999
    cg_add_edge!(CG, i, i+1)
end

println("=== Benchmark 02 ===")

@btime cg_nv($CG)
@btime cg_ne($CG)
@btime cg_degree($CG, 5000)
@btime collect(cg_neighbors($CG, 5000))
@btime cg_has_edge($CG, 100, 101)
@btime cg_connected_components($CG)
=#

#----------------------------------------------
CG = CustomGraph{Int}()

n = 3_000

for v in 1:n
    cg_add_vertex!(CG, v)
end

for i in 1:n-1
    for j in i+1:n
        cg_add_edge!(CG, i, j)
    end
end

println("=== Benchmark 03 ===")

@btime cg_nv($CG)
@btime cg_ne($CG)
@btime cg_degree($CG, 1000)
@btime collect(cg_neighbors($CG, 1000))
@btime cg_has_edge($CG, 100, 101)
@btime cg_rem_vertex!($CG, 500)
@btime cg_connected_components($CG)



#----------------------------------------------
G = complete_graph(3_000)

println("=== Benchmark 04 ===")

@btime nv($G)
@btime ne($G)
@btime degree($G, 1000)
@btime collect(neighbors($G, 1000))
@btime has_edge($G, 100, 101)
@btime rem_vertex!($G, 500)
@btime connected_components($G)