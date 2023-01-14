using JuliaGo
using BSON
using Random
using Chess
using Flux
using CUDA

function main()
    Random.seed!(1)
    local b::Board
    Root = Node()
    b = startboard()
    BSON.@load "models/policy_network.bson" model opt loss
    # UCI commands
    while true
        line = strip(readline())
        if line == "quit"
            break
        elseif line == "uci"
            println("id name JuliaGo")
            println("uciok")
        elseif line == "isready"
            println("readyok")
        elseif startswith(line, "position fen")
            line = strip(chopprefix(line, "position fen"))
            if contains(line, "moves")
                fen, m = map(String, split(line, " moves "))
                b = fromfen(fen)
                for i in split(m)
                    domove!(b, String(i))
                end
            else
                b = fromfen(String(line))
            end
        elseif startswith(line, "go")
            n_rollouts::UInt64 = 10
            @time bestmove = MCTS!(b, Root, n_rollouts, model)
            println("bestmove ", tostring(bestmove))
            total_N = sum(x -> x.data.N, Root.children)
            for i in Root.children
                println(i.m, " ", i.data.W, " ", i.data.N)
                if i.m == bestmove
                    Root = i
                    Root.parent = nothing
                    break
                end
            end
            println()
            for i in Root.children
                println(i.m, " ", i.data.W, " ", i.data.N)
            end


            println(
                "sureness = ",
                Root.data.N / total_N,
                " vs ",
                1 / length(moves(b)),
                "   ",
                Root.data.N,
                " / ",
                total_N,
            )
        elseif startswith(line, "bestmove")
            s = map(String, split(line, " "))
            u = domove!(b, s[2])
            for i in Root.children
                if tostring(i.m) == s[2]
                    Root = i
                    Root.parent = nothing
                    break
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
