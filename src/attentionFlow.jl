
using Statistics: mean
using LinearAlgebra: I


# BFS to find an augmenting path
# https://www.geeksforgeeks.org/python/python-program-for-breadth-first-search-or-bfs-for-a-graph/
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

# Single max-flow computation between source and sink
function max_flow(capacities, source, sink)
    # returns the flow matrix
    capacities = copy(capacities)
    total_flow = 0.0
    #max_flow = 0.0
    found_any_path = false


    while true
        path = bfs_path(capacities, source, sink)

        if path === nothing
            break
        end
        found_any_path = true
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

function adjusted_attention(attention_history::Array{Float32,4}, layer::Int, n_tokens::Int)
    # Paper: A = 0.5 * W_att + 0.5 * I, then re-normalize rows to sum to 1.
    # Average over all heads for layers below the layer of interest (single-head assumption).
    # attention_history dims: [source_pos, head, layer, affected_pos]
    n_heads = size(attention_history, 2)
    
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


function attention_flow(bot::ChatBot; input_prompt::String="Once upon a time", source_pos::Int=1)

    attention_history, output_tokens_strings = get_att_matrix(bot; input_prompt=input_prompt)
    n_tokens = length(output_tokens_strings)
    n_layers = bot.transformer.config.n_layers

    # Debugging
    #@show typeof(n_tokens)
    #@show typeof(n_layers)
    #@show n_tokens
    #@show n_layers

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


