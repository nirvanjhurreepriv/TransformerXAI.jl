
using Statistics: mean
using LinearAlgebra: I

"""
    bfs_path(capacities, source, sink)

This helper function was taken from geeksforgeeks and is used to find an augmenting path from a source node to a sink node

Url: https://www.geeksforgeeks.org/python/python-program-for-breadth-first-search-or-bfs-for-a-graph/

# Arguments
- `capacities` : Capacities of capacity matrix of the graph
- `source` : Index of source node
- `sink` : Index of sink node

# Returns
- `Vector{Int}` : A vector containing the nodes for the augmented path
If no path exists the return will be `nothing`

"""
function bfs_path(capacities, source, sink)
    # returns the path as a vector of node indices, or nothing if no path exists
    n = size(capacities, 1)
    parent = zeros(Int, n)
    visited = falses(n)
    queue = [source]
    visited[source] = true

    while !isempty(queue)
        current = popfirst!(queue)
        if current == sink
            # reconstruct path
            path = Int[]
            while current != source
                push!(path, current)
                current = parent[current]
            end
            push!(path, source)
            return reverse(path)
        end

        for neighbor in 1:n
            if capacities[current, neighbor] > 0 && !visited[neighbor]
                visited[neighbor] = true
                parent[neighbor] = current
                push!(queue, neighbor)
            end
        end
    end 
    return nothing   
end


"""
    max_flow(capacities, source, sink)

This function computes the single max flow between a source and sink
It repeatedly searches for augmenting paths and updates the capacities until no more paths are found

# Arguments
- `capacities`: Capacity matrix of the graph
- `source` : Index of source node
- `sink` : Index of sink node

# Returns
The function returns the maximum flow between source and sink

"""
function max_flow(capacities, source, sink)
    capacities = copy(capacities)
    total_flow = 0.0f0

    while true
        path = bfs_path(capacities, source, sink)

        if path === nothing
            break
        end
        bottleneck = minimum(capacities[path[i], path[i+1]]
                            for i in 1:length(path)-1)

        for i in 1:length(path)-1
            u = path[i]
            v = path[i+1]
            capacities[u,v] -= bottleneck
            capacities[v,u] += bottleneck
        end

        total_flow += bottleneck
    end
    return total_flow
end

"""
    adjusted_attention(attention_history::Array{Float32,4}, layer::Int, n_tokens::Int)

This function computes the adjusted attention matrix for a given layer.

# Arguments
- `attention_history::Array{Float32,4}` : Attention values
- `layer::Int` : The wished layer for computation
- `n_tokens::Int` : Number of tokens in the input

# Returns
The function returns an adjusted attention matrix with each entry representing the attention from a token in the current layer to a token in a previous layer

"""
function adjusted_attention(attention_history::Array{Float32,4}, layer::Int, n_tokens::Int)
    # Paper: A = 0.5 * W_att + 0.5 * I, then re-normalize rows to sum to 1.
    # attention_history dims: [source_pos, head, layer, affected_pos]

    # Average over all heads
    W = mean(attention_history[:, :, layer, :], dims=2)[:, 1, :] # (n_tokens × n_tokens)
    #Transpose
    W = W'

    A = 0.5f0 .* W .+ 0.5f0 .* Matrix{Float32}(I, n_tokens, n_tokens)
    
    # Re-normalize each row to sum to 1
    row_sums = sum(A, dims=2)
    A ./= row_sums
    
    return A  # A[i, j] = attention from token i (this layer) to token j (previous layer)
end

"""
    build_capacity_graph(attention_history::Array{Float32,4}, n_tokens::Integer, n_layers::Integer)

This function builds the capacity graph with the help of the `adjusted_attention` function

# Arguments
- `attention_history::Array{Float32,4}` : Attention values
- `n_tokens::Int` : Number of tokens in the input
- `n_layers::Int` : Number of layers 


# Returns
The function returns a capacity graph

"""
function build_capacity_graph(attention_history::Array{Float32,4}, n_tokens::Integer, n_layers::Integer)

    n_nodes = (n_layers +1 )* n_tokens

    # Create a capacity graph with (n_layers + 1) * n_tokens nodes
    capacity = zeros(Float32, n_nodes, n_nodes)

    for layer in 1:n_layers
        A = adjusted_attention(attention_history, layer, n_tokens)
        #Debbuging
        #println("\nLayer $layer")
        #display(A)

        #println("Row sums:")
        #println(sum(A, dims=2))

        for i in 1:n_tokens 
            for j in 1:n_tokens
                u = layer * n_tokens + i          # token i at layer l
                v = (layer - 1) * n_tokens + j    # token j at layer l-1
                capacity[u, v] = clamp(A[i, j], 0f0, 1f0)
            end
        end
        
    end

    return capacity
end

"""
    attention_flow(bot::ChatBot; input_prompt::String="Once upon a time", source_pos::Int=1)

This function creates a full attention flow through all layers to all input nodes

# Arguments
- `bot::ChatBot` : A base Llama2 Chatbot containing tokenizer and model

# Keyword Arguments
- `input_prompt::String="Once upon a time` : The input text for which the flow should be calculated/generated
- `source_pos::Int=1` : Starting Position 

# Returns
- `Vector{Float32}` : A vector containing the maximum attention flow values for source node to all input tokens
- `Vector{String}` : The input tokens corresponding to the returned flow values in the vecotr

"""
function attention_flow(bot::ChatBot; input_prompt::String="Once upon a time", source_pos::Int=1)

    attention_history, output_tokens_strings = get_att_matrix(bot; input_prompt=input_prompt)
    n_tokens = length(output_tokens_strings)
    n_layers = bot.transformer.config.n_layers

    (source_pos >= 1 && source_pos <= n_tokens) || throw(ArgumentError("source_pos must be in 1:$n_tokens, got $source_pos"))

    capacity = build_capacity_graph(attention_history, n_tokens, n_layers)


    source_node = n_layers * n_tokens + source_pos

    flow_values = Vector{Float32}(undef, n_tokens)
    for j in 1:n_tokens
        sink_node = j
        cap_copy = copy(capacity)
        flow_values[j] = max_flow(cap_copy, source_node, sink_node)
    end

    return flow_values, output_tokens_strings

end


