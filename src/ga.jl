#=
Algoritmo genético para o problema de coloração de vértices em grafos.
Autor: Atílio Gomes Luiz
Data: 21 de dezembro de 2025
=#
using Random
using Graphs

include("greedy.jl")

# Esse struct define um indivíduo (cromossomo)
struct Individual
    genome::Vector{Int}
    fitness::Float64
end

# Define uma ordem para os indivíduos com base no fitness deles
function Base.isless(a::Individual, b::Individual)
    a.fitness < b.fitness
end

# Esse struct guarda os parâmetros do algoritmo genético
struct GA_Parameters
    pop_size::Int
    crossover_rate::Float64
    mutation_rate::Float64
    elitism_rate::Float64
    max_generations::Int
    tournament_size::Int
    max_stagnation::Int
    local_search_rate::Float64
end

# Aqui eu estou definindo um tipo novo chamado "Population"
const Population = Vector{Individual}

#=
# Essa função cria uma população de indivíduos completamente aleatória, 
# não usei nenhuma heurística aqui.
=#
function init_population(parameters::GA_Parameters, genome_length::Int, rng::Random.AbstractRNG)::Population
    pop = Vector{Individual}(undef, parameters.pop_size)
    @inbounds for i in 1:parameters.pop_size
        genome = randperm(rng, genome_length)
        pop[i] = Individual(genome, NaN)
    end
    return pop
end

#=
# Essa função avalia cada indivíduo da população e calcula o fitness dele.
# O fitness de um indivíduo é dado pela coloração gulosa implementada no outro arquivo.
=#
function evaluate!(population::Population, graph::SimpleGraph, fitness_function)
    Threads.@threads for i in eachindex(population)
        @inbounds begin
            _, max_color = fitness_function(graph, population[i].genome)
            population[i] = Individual(population[i].genome, max_color)
        end
        
    end
end

#=
# Essa função implementa a seleção por k-torneio: k progenitores são selecionados 
# aleatoriamente e aquele com o menor fitness é escolhido como vencedor do torneio e retornado.
=# 
function tournament_selection(population::Population, k::Int, rng::Random.AbstractRNG)::Individual 
    k = max(2, min(k, length(population)))
    candidates = rand(rng, population, k)
    return minimum(candidates)
end

#= 
# Essa função implementa o cruzamento de um ponto, tendo o cuidado de sempre gerar uma permutação 
# dos vértices ao final. A complexidade dessa função é O(n), onde n é o tamanho do indivíduo.
=#
function one_point_order_crossover(p1::Individual, p2::Individual, rng::Random.AbstractRNG)::Individual
    n = length(p1.genome)
    cut = rand(rng, 1:n-1)

    used = falses(n) # marca genes já usados
    child = Vector{Int}(undef, n)

    # copia prefixo
    @inbounds for i in 1:cut
        g = p1.genome[i]
        child[i] = g 
        used[g] = true
    end

    # completa com ordem de p2
    idx = cut + 1
    @inbounds for g in p2.genome
        if !used[g]
            child[idx] = g 
            idx += 1
        end
    end

    return Individual(child, NaN)
end

#=
# Escolhe um vértice aleatoriamente pelo grau usando roullette wheel.
# Vértices com grau maior têm maior probabilidade, mas não exclusividade.
=#
function random_vertex_weighted_by_degree(g::SimpleGraph, rng::Random.AbstractRNG, deg::Vector{Int})::Int
    total = sum(deg)
    r = rand(rng) * total
    acc = 0.0

    @inbounds for v in 1:nv(g)
        acc += deg[v]
        if acc >= r
            return v
        end
    end

    return nv(g)  # fallback
end


#=
# Mutação: seleciona um vértice aleatoriamente e troca os seus vizinhos dois-a-dois
=#
function mutate_neighborhood_swap(ind::Individual, graph::SimpleGraph, mutation_rate::Float64, rng::Random.AbstractRNG, deg::Vector{Int})::Individual
    if rand(rng) >= mutation_rate 
        return ind 
    end

    genome = copy(ind.genome)   
    n = length(genome)

    pos = similar(genome)
    @inbounds for i in eachindex(genome)
        pos[genome[i]] = i
    end

    w = random_vertex_weighted_by_degree(graph, rng, deg)

    nbrs = collect(neighbors(graph, w))

    shuffle!(rng, nbrs)

    @inbounds for counter in 2:2:length(nbrs)
        a = pos[nbrs[counter-1]]
        b = pos[nbrs[counter]]
        genome[a], genome[b] = genome[b], genome[a]
    end

    return Individual(genome, NaN)
end


#=
Local search procedure
=#
function local_search_swap(ind::Individual, graph::SimpleGraph, fitness_function, max_iter::Int = 20)
    best_genome = copy(ind.genome)
    n = length(best_genome)
    best_fit = ind.fitness

    for _ in 1:max_iter
        i, j = rand(1:n, 2)
        i == j && continue 

        new_genome = copy(best_genome)
        new_genome[i], new_genome[j] = new_genome[j], new_genome[i]

        _, new_fit = fitness_function(graph, new_genome)

        if new_fit < best_fit
            best_genome = new_genome
            best_fit = new_fit
        end
    end
    return Individual(best_genome, best_fit)
end

#=
# Função que executa o GA 
=#
function run_ga(parameters::GA_Parameters, graph::SimpleGraph, seed::Int, fitness_function)
    rng = MersenneTwister(seed * 17)
    genome_length = nv(graph)

    # cria a população inicial e avalia o fitness de cada indivíduo
    population = init_population(parameters, genome_length, rng)
    evaluate!(population, graph, fitness_function)
    
    counter::Int = 0
    best_so_far::Individual = minimum(population)
    best_fitness_per_gen = Vector{Float64}()

    @inbounds for gen in 1:parameters.max_generations
        # tamanho da elite 
        k = max(1, ceil(Int, parameters.elitism_rate * length(population)))
        
        elites = partialsort(population, 1:k)

        for i in 1:k        
            if rand(rng) < parameters.local_search_rate
                elites[i] = local_search_swap(elites[i], graph, fitness_function)
            end
        end

        new_pop = Vector{Individual}(undef, parameters.pop_size)

        # copia elites 
        @inbounds for i in 1:k 
            new_pop[i] = elites[i]
        end

        # calcula os graus dos vértices 
        deg = degree(graph)

        # gera filhos em paralelo
        Threads.@threads for i in k+1:parameters.pop_size
            # RNG local por thread 
            rng_local = MersenneTwister(seed + Threads.threadid() + i)
            
            # seleciona dois progenitores por meio de torneio
            p1 = tournament_selection(population, parameters.tournament_size, rng_local)
            p2 = tournament_selection(population, parameters.tournament_size, rng_local)

            # decide se realiza o cruzamento ou mantém o primeiro pai na solução
            child = rand(rng_local) < parameters.crossover_rate ? 
                one_point_order_crossover(p1,p2,rng_local) : 
                Individual(copy(p1.genome), NaN)

            # chama a função de mutação
            new_pop[i] = mutate_neighborhood_swap(child, graph, parameters.mutation_rate, rng_local, deg)
        end

        # atualiza a população e avalia o fitness de cada cromossomo
        population = new_pop
        evaluate!(population, graph, fitness_function)

        # calcula o melhor indivíduo da geração e atualiza as estatísticas 
        best_in_iteration = minimum(population)
        append!(best_fitness_per_gen, best_in_iteration.fitness)

        # atualiza o melhor encontrado até agora e verifica a estagnação
        if best_in_iteration.fitness < best_so_far.fitness
            best_so_far = best_in_iteration 
            counter = 0;
        else
            counter += 1
        end
        if counter >= parameters.max_stagnation
            break
        end 
    end

    # retorna o indivíduo com a melhor coloração e 
    # retorna um vetor de estatísticas contendo o melhor fitness de cada geração
    return best_so_far, best_fitness_per_gen
end


#=
# Função que faz uma mutação que troca de dois genes do indivíduo. 
# É uma mutação bem simples. 
=#
function mutate_pair_swap(ind::Individual, mutation_rate::Float64, rng::Random.AbstractRNG)::Individual
    genome = copy(ind.genome)
    if rand(rng) < mutation_rate
        i, j = rand(rng, 1:length(genome), 2)
        genome[i], genome[j] = genome[j], genome[i]
    end
    return Individual(genome, NaN)
end



#=
# Função que executa o GA 
=#
#=
function run_ga(parameters::GA_Parameters, graph::SimpleGraph, seed::Int, fitness_function)
    rng = MersenneTwister(seed)
    genome_length = nv(graph)

    # cria a população inicial e avalia o fitness de cada indivíduo
    population = init_population(parameters, genome_length, rng)
    evaluate!(population, graph, fitness_function)
    
    counter::Int = 0
    best_so_far::Individual = minimum(population)
    best_fitness_per_gen = Vector{Float64}()

    @inbounds for gen in 1:parameters.max_generations
        new_pop = Population()

        # realiza o elitismo
        k::Int = max(1, ceil(Int, parameters.elitism_rate * length(population)))
        
        elites = partialsort(population, 1:k)
        append!(new_pop, elites)

        while length(new_pop) < parameters.pop_size
            # seleciona dois progenitores por meio de torneio
            p1 = tournament_selection(population, parameters.tournament_size, rng)
            p2 = tournament_selection(population, parameters.tournament_size, rng)

            # decide se realiza o cruzamento ou mantém o primeiro pai na solução
            child = rand(rng) < parameters.crossover_rate ? 
                one_point_order_crossover(p1,p2,rng) : Individual(copy(p1.genome), NaN)

            # chama a função de mutação
            child = mutate(child, parameters.mutation_rate, rng)
            push!(new_pop, child)
        end

        # atualiza a população e avalia o fitness de cada cromossomo
        population = new_pop
        evaluate!(population, graph, fitness_function)

        # calcula o melhor indivíduo da geração e atualiza as estatísticas 
        best_in_iteration = minimum(population)
        append!(best_fitness_per_gen, best_in_iteration.fitness)

        # atualiza o melhor encontrado até agora e verifica a estagnação
        if best_in_iteration.fitness < best_so_far.fitness
            best_so_far = best_in_iteration 
            counter = 0;
        else
            counter += 1
        end
        if counter >= parameters.max_stagnation
            break
        end 
    end

    # retorna o indivíduo com a melhor coloração e 
    # retorna um vetor de estatísticas contendo o melhor fitness de cada geração
    return best_so_far, best_fitness_per_gen
end
=#