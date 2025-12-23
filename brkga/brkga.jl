using Graphs
using Metaheuristics

include("../greedy/greedy.jl")

struct ColoringDecoder
    g::SimpleGraph
end

function (cd::ColoringDecoder)(x::Vector{Float64})
    order = sortperm(x)
    _, number_of_colors_used = greedyVertexColoring(cd.g, order)
    return number_of_colors_used
end

function invoke_brkga(g::SimpleGraph, pop_fraction::Int, pe::Float64, pm::Float64, rhoe::Float64)
    n = nv(g)
    pop_size::Int = ceil(n / pop_fraction)

    num_elites::Int = ceil(pop_size * pe)
    num_mutants::Int = ceil(pop_size * pm)
    num_offsprings::Int = pop_size - num_elites - num_mutants

    algorithm = BRKGA(
        num_elites = num_elites, 
        num_mutants = num_mutants, 
        num_offsprings = pop_size - num_elites - num_mutants, 
        N = num_elites + num_mutants + num_offsprings, 
        bias = rhoe
    )

    bounds = boxconstraints(lb = zeros(n), ub = ones(n))

    decoder = ColoringDecoder(g)

    result = optimize(x -> decoder(x), bounds, algorithm)

    return result, minimum(result)
end

