"""
    nn_input

Convert a board to a `(8, 8, 17) Array{Float32}`, suitable for input to the neural network. There are 12 `(8, 8)` arrays for the pieces, 4 for the castling rights and 1 for the colour of the side to move.
"""
function nn_input(b::Board)::Array{Float32,3}
    input = zeros(Float32, (8, 8, 17))
    counter = 1
    for c in (WHITE, BLACK)
        for j in (:cancastlekingside, cancastlequeenside)
            if eval(j)(b, c)
                input[:, :, counter] = ones((8, 8))
            end
            counter += 1
        end
    end
    if sidetomove(b) == WHITE
        input[:, :, counter] = ones((8, 8))
    end
    counter += 1
    for c in (WHITE, BLACK)
        for m in (:pawns, :knights, :bishops, :rooks, :queens, :kings)
            if sidetomove(b) == WHITE
                input[:, :, counter] = toarray(eval(m)(b, c))
            else
                input[:, :, counter] = reverse(toarray(eval(m)(b, c)))
            end
            counter += 1
        end
    end
    return input
end

"""
    nn_output(b::Board, next_board::Board)::Array{Float32,3}

Given a board `b` and the `next_board`, convert the move to a `(8, 8, 73)` array. The 73 values are encoded as follows: 1-7 horizontal right, 8-14 horizontal left, 15-21 vertical up, 22-28 vertical down, 29-35 north east, 36-42 north west, 42-49 south east, 50-56 south west, 57-64 = knight moves, 65-67 pawn mv NE and underpromote to bishop, knight, rook, 68-70 pawn mv NW, 71-73 pawn mv N.
"""
function nn_output(b::Board, next_board::Board)::Array{Float32,3}
    if sidetomove(b) == BLACK
        b = flip(flop(b))
        next_board = flip(flop(next_board))
    end
    # https://arxiv.org/abs/2111.09259
    # rank1 has val 8 and is the white backrow i.e. white pawns are rank2, val=7
    output = zeros(Float32, (8, 8, 73))
    f(x) = 8 - x + 1 # turn rank1 val to have val 1 and rank 8 val to have val 8
    r_from = f(rank(from(lastmove(next_board))).val)
    f_from = file(from(lastmove(next_board))).val
    r_to = f(rank(to(lastmove(next_board))).val)
    f_to = file(to(lastmove(next_board))).val

    r_diff = r_to - r_from
    f_diff = f_to - f_from

    local mv = 0
    local counter = 0

    underpromote_val = Dict(Chess.BISHOP => 1, Chess.KNIGHT => 2, Chess.ROOK => 3)
    knight_moves = Dict(
        (2, 1) => 1,
        (1, 2) => 2,
        (-1, 2) => 3,
        (-2, 1) => 4,
        (-2, -1) => 5,
        (-1, -2) => 6,
        (1, -2) => 7,
        (2, -1) => 8,
    )

    if ptype(pieceon(b, from(lastmove(next_board)))) == Chess.KNIGHT
        counter = 56
        mv = knight_moves[(r_diff, f_diff)]
    elseif ispromotion(lastmove(next_board))
        if promotion(lastmove(next_board)) == Chess.QUEEN
            # default promotion to queen
            mv = 1 # move is max 1 square
            if f_diff == 0
                counter = 14 # vertical up
            elseif f_diff > 0
                counter = 28 # capture NE
            else
                counter = 35 # capture NW
            end
        else
            # underpromotion
            mv = underpromote_val[promotion(lastmove(next_board))]
            if f_diff == 0
                counter = 70
            elseif f_diff > 0
                counter = 64
            else
                counter = 67
            end
        end
    else
        if r_diff == 0
            counter = f_diff > 0 ? 0 : 7
            mv = f_diff
        elseif f_diff == 0
            counter = r_diff > 0 ? 14 : 21
            mv = r_diff
        elseif r_diff > 0
            counter = f_diff > 0 ? 28 : 35
            mv = r_diff
        elseif r_diff < 0
            counter = f_diff > 0 ? 42 : 49
            mv = r_diff
        end
    end

    if mv == 0
        error("move wrong")
    end

    output[r_from, f_from, counter+abs(mv)] = 1.0
    return output
end

"""
    parse_pgns

Parse a list of pgns and return arrays for training the neural network. Positions in each game are selected according with probability `prob` to prevent highly correlated positions.
"""
function parse_pgns(list_of_files, prob)
    d = Dict{Array{Float32,3},Array{Float32,3}}()
    x = Vector{Array{Float32}}()
    y = Vector{Array{Float32}}()
    non_unique = 0
    for j in list_of_files
        f = open(j)
        p = Chess.PGN.PGNReader(f)
        g = try
            Chess.PGN.readgame(p)
        catch e
            close(f)
            continue
        end
        bs = boards(g)
        close(f)
        for i = 1:length(bs)-1
            if Random.rand() < prob
                temp_input = nn_input(bs[i])
                temp_output = nn_output(bs[i], bs[i+1])
                if !haskey(d, temp_input)
                    d[temp_input] = temp_output
                else
                    non_unique += 1
                    d[temp_input] .+= temp_output
                end
            end
        end
    end
    println("found ", length(d), " unique inputs")
    println("found ", non_unique, " non-unique inputs")
    out_x = zeros(Float32, (8, 8, 17, length(d)))
    out_y = zeros(Float32, (8, 8, 73, length(d)))
    counter = 1
    for (k, v) in pairs(d)
        out_x[:, :, :, counter] .= k
        out_y[:, :, :, counter] .= v / sum(v)
        @assert sum(out_y[:, :, :, counter]) ≈ 1.0
        counter += 1
    end
    out_y = reshape(out_y, (8 * 8 * 73, length(d)))
    return out_x, out_y
end

"""
    input(preloaded, ratio)

Create train and test sets from the pgns, split according to the `ratio`.
"""
function train_test_data(preloaded, file_dir, ratio)
    @assert 0.0 < ratio < 1.0
    if !preloaded
        files = Random.shuffle!(glob("*.pgn", file_dir))
        println("found ", length(files), " pgns")
        println("parsing pgns")
        x, y = parse_pgns(files)
        JLD.save("data.jld", "x", x, "y", y)
        println("saved to file data.jld")
    else
        println("loading data from file")
        d = JLD.load("data.jld")
        x = d["x"]
        y = d["y"]
    end
    split = floor(Int64, ratio * size(x, 4))
    train_x = x[:, :, :, 1:split]
    train_y = y[:, 1:split]
    test_x = x[:, :, :, split+1:end]
    test_y = y[:, split+1:end]
    println("file reading complete")
    println("train_x = ", size(train_x), "train_y = ", size(train_y))
    println("test_x = ", size(test_x), "test_y = ", size(test_y))
    return train_x, train_y, test_x, test_y
end

"""
    output_to_move

Convert the index of the highest probability element in the model output array to a move.
"""
function output_to_move(x)::Move
    rank = x[1]
    file = x[2]
    mv = x[3]

    knight_moves = Dict(
        1 => (2, 1),
        2 => (1, 2),
        3 => (-1, 2),
        4 => (-2, 1),
        5 => (-2, -1),
        6 => (-1, -2),
        7 => (1, -2),
        8 => (2, -1),
    )
    promotion_piece = PIECE_TYPE_NONE
    promotion_dict = Dict(1 => KNIGHT, 2 => BISHOP, 3 => ROOK)

    from_sq = Square((8 - rank + 1) + ((file - 1) * 8))

    if mv <= 7
        file += mv
    elseif 7 < mv <= 14
        file -= mv - 7
    elseif 14 < mv <= 21
        rank += mv - 14
    elseif 21 < mv <= 28
        rank -= mv - 21
    elseif 28 < mv <= 35
        rank += mv - 28
        file += mv - 28
    elseif 35 < mv < 42
        rank += mv - 35
        file -= mv - 35
    elseif 42 < mv <= 49
        rank -= mv - 42
        file += mv - 42
    elseif 49 < mv <= 56
        rank -= mv - 49
        file -= mv - 49
    elseif 56 < mv <= 64
        rank += knight_moves[mv-56][1]
        file += knight_moves[mv-56][2]
    elseif 64 < mv <= 67
        rank += 1
        file += 1
        promotion_piece = promotion_dict[mv-64]
    elseif 67 < mv <= 70
        rank += 1
        file -= 1
        promotion_piece = promotion_dict[mv-67]
    elseif 70 < mv <= 73
        rank += 1
        promotion_piece = promotion_dict[mv-70]
    else
        error()
    end
    to_sq = Square((8 - rank + 1) + ((file - 1) * 8))

    if promotion_piece == PIECE_TYPE_NONE
        return Move(from_sq, to_sq)
    else
        return Move(from_sq, to_sq, promotion_piece)
    end
end

"""
    predict_move

Given a board and policy, output the array of possible next moves with associated probabilty. If the policy outputs and illegal move, a legal move is selected at random.
"""
function predict_move(b::Board, model)
    n = nn_input(b)
    n = reshape(n, (size(n)..., 1))
    out = softmax(model(n))
    out = reshape(out, (8, 8, 73))
    max_val, idx = findmax(out)
    mv = output_to_move(idx)
    if mv ∉ moves(b)
        children = moves(b)
        mv = children[Random.rand(1:length(children))]
    end
    return mv
end
