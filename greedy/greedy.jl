using Graphs

#=
# Essa função implementa o algoritmo guloso de coloração de vértices.
# 
# Entrada: 
# - graph: grafo simples 
# - sequence: permutação dos vértices do grafo 
#
# Saída:
# - as cores dos vértices
# - a maior cor utilizada
# 
# Complexidade: O(V+E) - linear no tamanho do grafo
=#
function greedyVertexColoring(g::SimpleGraph, sequence::Vector{Int})::Tuple{Vector{Int},Int}
    n = nv(g)
    color = fill(0, n) # cria um array de tamanho n inicializado com zeros
    used = falses(n) # vetor booleano de tamanho n usado para marcar cores proibidas
    
    for v in sequence
        # marca cores dos vizinhos
        @inbounds for w in neighbors(g, v)
            if color[w] > 0
                used[color[w]] = true
            end
        end
                
        # escolhe a menor cor disponível
        c = 1
        while used[c]
            c += 1
        end
        color[v] = c 
        
        # limpa marcas
        @inbounds for w in neighbors(g, v)
            if color[w] > 0
                used[color[w]] = false
            end
        end 
    end
    return color, maximum(color)
end
