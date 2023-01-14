using Glob
using Chess
using Flux
using BSON: @load, @save
using Dates: now
using CUDA
import JLD
import Random

"""
    batch_train!(loss,
    ps,
    train_x,
    train_y,
    test_x,
    test_y,
    opt,
    model,
    batchsize = 64,
    epochs = 100000,)
  
At each epoch, a random selection of the `train_x` data of size `batchsize` is trasferred to the gpu and the model trained - this is mini-batch gradient descent. The model is saved in cpu format at every 5% of the total number of epochs.
"""
function batch_train!(
    loss,
    ps,
    train_x,
    train_y,
    test_x,
    test_y,
    opt,
    model,
    batchsize = 64,
    epochs = 100000,
)
    model_num = 0
    for i = 1:epochs
        rindex = rand(1:size(train_x, 4)-batchsize+1)
        data = [(
            gpu(train_x[:, :, :, rindex:rindex+batchsize-1]),
            gpu(train_y[:, rindex:rindex+batchsize-1]),
        )]
        Flux.train!(loss, ps, data, opt)
        if i % (epochs ÷ 20) == 0
            out::Float32 = 0.0
            count = 0
            for j = 1:batchsize:size(test_x, 4)-batchsize
                out +=
                    loss(gpu(test_x[:, :, :, j:j+batchsize]), gpu(test_y[:, j:j+batchsize]))
                count += 1
            end
            out /= count
            println("\n epoch = ", i, "total loss = ", out)
            let model = cpu(model), opt = cpu(opt)
                @save "models/model-$(model_num).bson" model opt loss = out
            end
            model_num += 1
        end
    end
end

"""
    policy_network_train(preloaded_data=false, model_name="policy_network.bson")

Train a policy network on a dataset of professional chess games. The aim of the policy network is to predict the move from a state that a professional would likely play.

By default the model is transferred to the gpu.

Pass `preloaded_data=true` if the pgns have already been parsed and converted into a input for the neural network (considerable time saving).
"""
function policy_network_train(
    preloaded_data = false,
    train_test_split = 0.8,
    model_name = "policy_network.bson",
)
    train_x, train_y, test_x, test_y =
        train_test_data(preloaded, "data/pgns", train_test_split)
    model =
        Flux.Chain(
            Flux.Conv((3, 3), 17 => 256, Flux.relu, pad = SamePad()),
            ResNetBlock()...,
            ResNetBlock()...,
            ResNetBlock()...,
            ResNetBlock()...,
            ResNetBlock()...,
            ResNetBlock()...,
            Flux.Conv((1, 1), 256 => 256, Flux.relu),
            Flux.Conv((1, 1), 256 => 73),
            Flux.flatten,
        ) |> gpu

    loss(pred, target) = Flux.Losses.logitcrossentropy(model(pred), target)::Float32 |> gpu
    opt = Flux.Optimise.Adam()
    ps = Flux.params(model)
    batch_train!(loss, ps, train_x, train_y, test_x, test_y, opt, model)
    let model = cpu(model), opt = cpu(opt)
        @save model_name model opt
    end
end
