"""
    JuliaGo

Julia implementation of AlphaGo for chess. Implements MCTS and policy network evaluation according to the first AlphaGo/Zero paper.
"""

module JuliaGo

using Chess
using Flux
using Random
using BSON
using Flux
using CUDA

include("policy/ResNetBlock.jl")
export ResNetBlock

include("policy/input.jl")
export nn_input, nn_output, predict_move

include("policy/policy.jl")
export policy_network_train

include("MCTS.jl")
export Entry, Node, MCTS!

end # module
