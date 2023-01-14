import Random
import Base.Threads.@threads

const UNDOINFO_NULL = UndoInfo(0, 0, 0, 0, false, SS_EMPTY, SS_EMPTY, EMPTY, 0)
"""
    Entry

Type storing the wins, draws and number of visits for a Node.
"""
mutable struct Entry
    W::UInt64
    D::UInt64
    N::UInt64
end

function show(io::IO, x::Entry)
    println("W = ", x.W, " D = ", x.D, " N = ", x.N)
end
show(x::Entry) = show(stdout, x)

Entry() = Entry(0, 0, 0)
"""
    Node

Type used to represent nodes on the game tree. 
"""
mutable struct Node
    data::Entry
    u::UndoInfo
    m::Move
    unexplored_moves::MoveList
    children::Vector{Node}
    parent::Union{Node,Nothing}
end

Node() = Node(Entry(), UNDOINFO_NULL, MOVE_NULL, MoveList(0), Vector{Node}(), nothing)
Node(W, D, N, u, m, M, P) = Node(Entry(W, D, N), u, m, M, Vector{Node}(), P)

"""
    robust_child(P::Node)::Move

Selects the child with the highest visit count `N` and returns the `Move` associated with that child. This is the default method of choosing the action but others exist, see secure child and max child.
"""
function robust_child(P::Node)::Move
    N, idx = findmax(x -> x.data.N, P.children)
    return P.children[idx].m
end

"""
    function selection_policy(C::Node)

Calculates the Upper Confidence Bound for Trees (UCT) via

``\\mathrm{UCT} = \\frac{w_i}{n_i} + C\\sqrt{\\frac{\\ln N_i}{n_i}}``

where

``w_i`` is the number of wins associated with taking the action that results in this node i.e. the number of wins from the parent's perspecive, ``n_i`` is the number of simulations from this node, ``C = \\sqrt{2}`` is a coefficient controlling the balance between exploration and exploitation, ``N_i`` is the number of simulations of the parent node.
"""
function selection_policy(C::Node)
    # Wins are local to each node i.e. representing a win for side whose turn it is for that node rather than the parent. This means that we use (C.data.N - C.data.W) as this is wins for the parent at this node. 
    # The value is maximised in the selection
    return ((C.data.N - C.data.W) / C.data.N) +
           ((sqrt(2.0) * sqrt(log(C.parent.data.N) / C.data.N)))
end


"""
    MCTS!(b::Board, P::Node, num_rollouts::UInt64)

Monte Carle Tree Search algorithm. Returns the Node with the highest visit count via [`robust_child`](@ref).

"""
function MCTS!(b::Board, P::Node, num_rollouts::UInt64, policy)
    firstpass!(b, P, num_rollouts)
    count = 0
    finish_after_expansion = false

    while true
        if count > 300000
            if finish_after_expansion
                break
            end
        end
        # all moves have been explored i.e. completed node
        # select the node according to selection policy
        if length(P.unexplored_moves) == 0
            # findmin returns the value, index of the minimum value in P.children
            P = P.children[findmax(selection_policy, P.children)[2]]
            u = domove!(b, P.m)
            finish_after_expansion = false
        else
            # node is not fully expanded
            P = expansion!(b, P, num_rollouts, policy)
            @assert P.parent == nothing
            finish_after_expansion = true
        end
        count += 1
    end
    return robust_child(P)
end

"""
     firstpass!(b::Board, P::Node, num_rollouts::UInt64)

Given the root node of tree, where no simulation have yet occured, populate all the children of the root node with a value based on rollouts and back propagation.
"""
function firstpass!(b::Board, P::Node, num_rollouts::UInt64)
    # first pass of the children
    # only apply to root node
    @assert P.parent == nothing
    P.unexplored_moves = moves(b)
    results = Dict{PieceColor,UInt64}(WHITE => 0, BLACK => 0, COLOR_NONE => 0)
    for i in P.unexplored_moves
        win = 0
        draw = 0
        u = domove!(b, i)
        for j = 1:num_rollouts
            winning_color = rollout(b)
            results[winning_color] += 1
            if winning_color == COLOR_NONE
                draw += 1
            elseif winning_color == sidetomove(b)
                win += 1
            end
        end
        push!(P.children, Node(win, draw, num_rollouts, u, i, moves(b), P))
        P.data.N += num_rollouts
        P.data.D += draw
        undomove!(b, u)
    end
    P.data.W += results[sidetomove(b)]
    recycle!(P.unexplored_moves)
    @assert length(P.unexplored_moves) == 0
end

"""
    expansion!(b::Board, P::Node, num_rollouts::UInt64)::Node

Selects a child node at random and removes the move from the parent node's `unexplored_moves` list. Conducts rollouts from the child node until a terminal state and backpropagates the result until the root node of of the tree (`P` is not the root node). This rollout function is multi-threaded i.e. each rollout is perfomed by a different thread. The results are collected and counted (making it all thread safe).
"""
function expansion!(b::Board, P::Node, num_rollouts::UInt64, policy)::Node
    results = Dict{PieceColor,UInt64}(WHITE => 0, BLACK => 0, COLOR_NONE => 0)
    # hack to emulate pop! - randomly select move, swap with final move, decrement move list counter
    r = Random.rand(1:length(P.unexplored_moves))
    i = P.unexplored_moves[r]
    P.unexplored_moves.moves[r] = P.unexplored_moves.moves[length(P.unexplored_moves)]
    P.unexplored_moves.count -= 1

    u = domove!(b, i)

    # store the results in an array to prevent a data race
    temp_arr = Vector{PieceColor}(undef, num_rollouts)
    @threads for i = 1:num_rollouts
        temp_arr[i] = rollout(b)
    end

    for (k, v) in results
        results[k] += count(x -> x == k, temp_arr)
    end

    C = Node(results[sidetomove(b)], results[COLOR_NONE], num_rollouts, u, i, moves(b), P)
    push!(P.children, C)
    P = backpropagate!(C, b, results)
    return P
end

"""
    backpropagate!(C::Node, b::Board, results::Dict{PieceColor,UInt64})

Backpropagates the result of a rollout up the tree until the root node. The root node is defined by `root.parent == nothing`. Returns the root node.
"""
function backpropagate!(C::Node, b::Board, results::Dict{PieceColor,UInt64})
    # backpropagate scores up the tree from child node C
    # update, W, N and undo moves until at root node
    # child node C has been updated already in expansion!
    while C.parent != nothing
        undomove!(b, C.u)
        C = C.parent
        C.data.N += results[WHITE] + results[BLACK] + results[COLOR_NONE]
        C.data.D += results[COLOR_NONE]
        C.data.W += results[sidetomove(b)]
    end
    return C
end

"""
    rollout(b::Board)::PieceColor

Plays moves randomly until a terminal state is reached.
"""
function rollout(b::Board)::PieceColor
    # https://stackoverflow.com/questions/30509132/monte-carlo-tree-search-backpropagation-backup-step-why-change-perspective-o/30521675#30521675I
    x::Board = startboard() # this seems wasteful
    Chess.copyto!(x, b)
    while !isterminal(x)
        children = moves(x)
        r = Random.rand(1:length(children))
        u = domove!(x, children[r])
    end
    return isdraw(x) ? COLOR_NONE : coloropp(sidetomove(x))
end

"""
    rollout(b::Board, policy)::PieceColor

Plays moves according to a policy, a neural network, until a terminal state is reached. When a terminal state is reached the colour of the winning side is returned, `COLOR_NONE` if it is a draw
"""
function rollout(b::Board, policy)::PieceColor
    x::Board = startboard()
    Chess.copyto!(x, b)
    while !isterminal(x)
        r = predict_move(x, policy)
        u = domove!(x, r)
    end
    return isdraw(x) ? COLOR_NONE : coloropp(sidetomove(x))
end
