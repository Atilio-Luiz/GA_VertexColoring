using Graphs
using ArgParse
using Random
using CSV
using DataFrames
using Base.Threads

include("brkga.jl")

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

        "--pe"
            help = "Taxa da elite"
            arg_type = Float64
            default = 0.8

        "--pm"
            help = "Taxa de mutantes"
            arg_type = Float64
            default = 0.1
        
        "--rhoe"
            help = "Probabilidade de selecionar um gene do progenitor elitizado"
            arg_type = Float64
            default = 0.1

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

    println("[INFO] Número de threads disponíveis: $(Threads.nthreads())")

    args = parse_command_line()

    instance            = args["instance"]
    seed                = args["seed"]
    pop_factor          = args["pop_factor"]
    pe                  = args["pe"]
    pm                  = args["pm"] 
    rhoe                = args["rhoe"]
    trials              = args["trials"]
    output_file         = args["output"]

    
    # Lê o grafo do arquivo
    graph = read_simple_graph(instance)
    
    # evita sobrescrever csv por acidente
    isfile(output_file) && error("Arquivo $output_file já existe")

    df_header = DataFrame(trial = Int[], seed = Int[], graph = String[], n = Int[], m = Int[], 
                          density = Float64[], bestFitness = Int[], time_sec = Float64)

    CSV.write(output_file, df_header) # cria e salva o cabeçalho do csv

    # invoca o brkga: warm-up
    Random.seed!(seed)
    invoke_brkga(graph, pop_factor, pe, pm, rhoe)

    # reinicializa a semente 
    Random.seed!(seed)
    # ---------------------------------
    # Execuções independentes do GA
    # ---------------------------------
    for t in 1:trials
        trial_seed = seed + t 

        start = time()
        chr, max_color = invoke_brkga(graph, pop_factor, pe, pm, rhoe)
        elapsed_time = time() - start

        println(chr)

        # salva resultados no csv
        instance_name = basename(instance)
        density = (2 * ne(graph)) / (nv(graph) * (nv(graph) - 1))
        df_row = DataFrame(trial = t, seed = trial_seed, graph = instance_name, n = nv(graph), 
                            m = ne(graph), density = density, bestFitness = max_color, time_sec = elapsed_time)
        CSV.write(output_file, df_row; append = true)
    end
end

main()




