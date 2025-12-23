using Graphs
using ArgParse
using Random
using CSV
using DataFrames

include("ga.jl")

# Recebe como entrada o caminho para o arquivo contendo a 
# lista de arestas do grafo e cria um grafo simples (1-based)
function read_simple_graph(filename::String)
    edges = Tuple{Int,Int}[]
    vertices = Set{Int}()

    for linha in eachline(filename)
        u, v = parse.(Int, split(linha))
        push!(edges, (u, v))
        push!(vertices, u)
        push!(vertices, v)
    end

    vertex_map = Dict{Int,Int}()
    for (i, v) in enumerate(sort(collect(vertices)))
        vertex_map[v] = i
    end

    g = SimpleGraph(length(vertices))

    for (u, v) in edges
        add_edge!(g, vertex_map[u], vertex_map[v])
    end

    return g
end

# Processamento da linha de comando
function parse_command_line()
    settings = ArgParseSettings()

    @add_arg_table settings begin
        "--instance"
            help = "Caminho para o arquivo de instância (.txt, .col, etc.)"
            arg_type = String
            required = true

        "--seed"
            help = "Semente do gerador de números aleatórios"
            arg_type = Int
            default = 1234

        "--pop_factor"
            help = "Denominador que determina o tamanho da população"
            arg_type = Int
            default = 2

        "--crossover_rate"
            help = "Taxa de cruzamento"
            arg_type = Float64
            default = 0.8

        "--mutation_rate"
            help = "Taxa de mutação"
            arg_type = Float64
            default = 0.1
        
        "--elitism_rate"
            help = "Taxa de elitismo"
            arg_type = Float64
            default = 0.1

        "--max_gen"
            help = "Número máximo de gerações"
            arg_type = Int
            default = 200

        "--tournament_size"
            help = "Número de progenitores a participarem do torneio"
            arg_type = Int
            default = 2
        
        "--max_stagnation"
            help = "Número máximo de gerações sem melhoria da solução global"
            arg_type = Int
            default = 100
        
        "--ls_iter"
            help = "Número máximo de iterações da busca local"
            arg_type = Int
            default = 20

        "--trials"
            help = "Número máximo de execuções independentes do algoritmo genético"
            arg_type = Int
            default = 15 

        "--output"
            help = "Caminho para o arquivo csv de saída (.csv)"
            arg_type = String
            default = "result.csv" 
    end

    return parse_args(settings)
end

# Função principal
function main()
    args = parse_command_line()

    instance            = args["instance"]
    seed                = args["seed"]
    pop_factor          = args["pop_factor"]
    crossover_rate      = args["crossover_rate"]
    mutation_rate       = args["mutation_rate"] 
    elitism_rate        = args["elitism_rate"]
    max_generations     = args["max_gen"]
    tournament_size     = args["tournament_size"]
    max_stagnation      = args["max_stagnation"]
    ls_iter             = args["ls_iter"]
    trials              = args["trials"]
    output_file         = args["output"]

    
    # Lê o grafo do arquivo
    graph = read_simple_graph(instance)
    # Configura os parâmetros do GA
    parameters = GA_Parameters(floor(nv(graph)/pop_factor), crossover_rate, mutation_rate, 
                    elitism_rate, max_generations, tournament_size, max_stagnation, ls_iter)

    
    # evita sobrescrever csv por acidente
    isfile(output_file) && error("Arquivo $output_file já existe")

    df_header = DataFrame(trial = Int[], seed = Int[], graph = String[], n = Int[], m = Int[], density = Float64[], bestFitness = Int[], time_sec = Float64)

    CSV.write(output_file, df_header) # cria e salva o cabeçalho do csv

    # --------------------------
    # Warm-up (fora da medição)
    # --------------------------
    Random.seed!(seed)
    run_ga(parameters, graph, seed, greedyVertexColoring) # warmup: para desconsiderar o custo da compilação JIT

    # Reseta RNG antes dos trials: garante que os trials começam do mesmo estado
    Random.seed!(seed)

    # ---------------------------------
    # Execuções independentes do GA
    # ---------------------------------
    for t in 1:trials
        trial_seed = seed + t 

        start = time()
        chr, _ = run_ga(parameters, graph, trial_seed, greedyVertexColoring)
        elapsed_time = time() - start

        # salva resultados no csv
        instance_name = basename(instance)
        density = (2 * ne(graph)) / (nv(graph) * (nv(graph) - 1))
        df_row = DataFrame(trial = t, seed = trial_seed, graph = instance_name, n = nv(graph), m = ne(graph), density = density, bestFitness = chr.fitness, time_sec = elapsed_time)
        CSV.write(output_file, df_row; append = true)
    end

end

main()

