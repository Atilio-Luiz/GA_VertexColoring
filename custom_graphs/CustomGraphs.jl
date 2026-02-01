#=
Funções que implementam um grafo simples que suporta 
operações de remoção de vértices, dentre outras.
As funções foram implementadas usando como base a biblioteca Graphs.jl 
Portanto, elas usam uma implementação de grafo como lista de adjacências.

Autor: Atílio G. Luiz
Data: Janeiro de 2026
=#
using Graphs

struct CustomGraph{T}
    g::SimpleGraph{Int}
    alias_to_vertex::Dict{T, Int}
    vertex_to_alias::Vector{T}
end

"""
    CustomGraph{T}()

    Construtor: cria um grafo simples vazio
"""
function CustomGraph{T}() where T
    return CustomGraph(SimpleGraph{Int}(0), Dict{T,Int}(), T[])
end


"""
    cg_nv(CG::CustomGraph{T})

    Retorna o número de vértices do grafo
""" 
function cg_nv(CG::CustomGraph{T}) where T 
    return Graphs.nv(CG.g)
end

"""
    cg_ne(CG::CustomGraph{T})

    Retorna o número de arestas do grafo 
"""
function cg_ne(CG::CustomGraph{T}) where T 
    return Graphs.ne(CG.g)
end


"""
    cg_density(CG::CustomGraph{T}) -> Float64

    Retorna a densidade do grafo
"""
function cg_density(CG::CustomGraph{T}) where T
    n = cg_nv(CG)
    m = cg_ne(CG)
    n <= 1 && return 0.0
    return ((2.0 * m) / (n * (n - 1.0)))::Float64
end


"""
    cg_vertices(CG::CustomGraph{T}) -> Vector{T}

    Retorna um iterador para os vértices do grafo.
"""
function cg_vertices(CG::CustomGraph{T}) where T 
    return (CG.vertex_to_alias[v] for v in Graphs.vertices(CG.g))
end


"""
    cg_edges(CG::CustomGraph{T}) 

    Retorna um iterador sobre as arestas do grafo, onde cada aresta é
    representada por uma tupla `(u, v)` dos rótulos externos (aliases).
"""
function cg_edges(CG::CustomGraph{T}) where T
    return ((CG.vertex_to_alias[src(e)], CG.vertex_to_alias[dst(e)]) for e in Graphs.edges(CG.g))
end


"""
    cg_has_vertex(CG::CustomGraph{T}, v_alias::T)

    Retorna true se e somente se tem o vértice no grafo
"""
function cg_has_vertex(CG::CustomGraph{T}, v_alias::T) where T
    return haskey(CG.alias_to_vertex, v_alias)
end


"""
    cg_has_edge(CG::CustomGraph{T}, u_alias::T, v_alias::T)

    Retorna true se e somente se existe a aresta no grafo
"""
function cg_has_edge(CG::CustomGraph{T}, u_alias::T, v_alias::T) where T
    if !haskey(CG.alias_to_vertex, u_alias) || !haskey(CG.alias_to_vertex, v_alias)
        return false
    end

    u = CG.alias_to_vertex[u_alias]
    v = CG.alias_to_vertex[v_alias]
    return Graphs.has_edge(CG.g, u, v)
end


"""
    cg_neighbors(CG::CustomGraph{T}, v_alias::T)

    Retorna um iterador para os vizinhos de um vértice
"""
function cg_neighbors(CG::CustomGraph{T}, v_alias::T) where T
    haskey(CG.alias_to_vertex, v_alias) || throw(ArgumentError("vertex $v_alias does not exist"))

    return (CG.vertex_to_alias[u] for u in Graphs.neighbors(CG.g, CG.alias_to_vertex[v_alias]))
end
 
"""
    cg_degree(CG::CustomGraph{T}, v_alias::T)

    Retorna o grau de um vértice
"""
function cg_degree(CG::CustomGraph{T}, v_alias::T) where T
    haskey(CG.alias_to_vertex, v_alias) || throw(ArgumentError("vertex $v_alias does not exist")) 

    v = CG.alias_to_vertex[v_alias]
    return Graphs.degree(CG.g, v)
end


"""
    cg_max_degree(CG::CustomGraph{T})

    Retorna o grau máximo do grafo
"""
function cg_max_degree(CG::CustomGraph{T}) where T
    Graphs.nv(CG.g) == 0 && return 0
    return maximum(Graphs.degree(CG.g, v) for v in Graphs.vertices(CG.g))
end

"""
    cg_min_degree(CG::CustomGraph{T}) where T

    Retorna o grau mínimo do grafo
"""
function cg_min_degree(CG::CustomGraph{T}) where T
    Graphs.nv(CG.g) == 0 && return 0
    return minimum(Graphs.degree(CG.g, v) for v in Graphs.vertices(CG.g))
end


"""
    cg_add_vertex!(CG::CustomGraph{T}, v_alias::T)

    Adiciona vértice ao grafo
"""
function cg_add_vertex!(CG::CustomGraph{T}, v_alias::T) where T
    haskey(CG.alias_to_vertex, v_alias) && return false

    Graphs.add_vertex!(CG.g)
    v = Graphs.nv(CG.g)

    CG.alias_to_vertex[v_alias] = v
    push!(CG.vertex_to_alias, v_alias)
    return true
end

  
"""
    cg_rem_vertex!(CG::CustomGraph{T}, v_alias::T)

    Remove vértice do grafo
"""
function cg_rem_vertex!(CG::CustomGraph{T}, v_alias::T) where T
    !haskey(CG.alias_to_vertex, v_alias) && return false

    v = CG.alias_to_vertex[v_alias]
    last = Graphs.nv(CG.g)
    last_alias = CG.vertex_to_alias[last]

    Graphs.rem_vertex!(CG.g, v)

    if v != last
        CG.vertex_to_alias[v] = last_alias
        CG.alias_to_vertex[last_alias] = v
    end

    pop!(CG.vertex_to_alias)
    delete!(CG.alias_to_vertex, v_alias)
    return true
end


"""
    cg_add_edge!(CG::CustomGraph{T}, u_alias::T, v_alias::T)

    Adiciona aresta ao grafo
"""
function cg_add_edge!(CG::CustomGraph{T}, u_alias::T, v_alias::T) where T
    haskey(CG.alias_to_vertex, u_alias) || throw(ArgumentError("vertex $u_alias does not exist"))
    haskey(CG.alias_to_vertex, v_alias) || throw(ArgumentError("vertex $v_alias does not exist"))

    u = CG.alias_to_vertex[u_alias]
    v = CG.alias_to_vertex[v_alias]

    return Graphs.add_edge!(CG.g, u, v)
end


"""
    cg_rem_edge!(CG::CustomGraph{T}, u_alias::T, v_alias::T)

    Remove aresta do grafo
"""
function cg_rem_edge!(CG::CustomGraph{T}, u_alias::T, v_alias::T) where T
    if !haskey(CG.alias_to_vertex, u_alias) || !haskey(CG.alias_to_vertex, v_alias)
        return false
    end

    u = CG.alias_to_vertex[u_alias]
    v = CG.alias_to_vertex[v_alias]

    return Graphs.rem_edge!(CG.g, u, v)
end


"""
    cg_connected_components(CG::CustomGraph{T})

    Retorna uma lista das componentes conexas do grafo
""" 
function cg_connected_components(CG::CustomGraph{T}) where T
    components = Graphs.connected_components(CG.g) 
    return [[CG.vertex_to_alias[v] for v in comp] for comp in components]
end

