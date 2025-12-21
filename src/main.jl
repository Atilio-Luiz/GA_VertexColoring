using Graphs

include("ga.jl")

graph = grid([100,4]) # grafo grade 

parameters = GA_Parameters(floor(nv(graph)/2), 0.8, 0.3, 0.1, 500, 5, 150, 0.5)

for seed in [333,4321,56,78,89071,23477,651,13,17]
    chr, _ = run_ga(parameters, graph, seed, greedyVertexColoring)
    println("seed: $(seed), best fitness: $(chr.fitness)")
    #= 
    for i in eachindex(statistics) 
        println("gen $i => fitness = $(statistics[i])")
    end
    println("------------------------------------")
    =#
end