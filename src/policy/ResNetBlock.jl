function ResNetInternal()
    return Flux.Chain(
        Flux.Conv((3, 3), 256 => 256, Flux.relu, pad = SamePad()),
        Flux.BatchNorm(256),
        Flux.Conv((3, 3), 256 => 256, Flux.relu, pad = SamePad()),
        Flux.BatchNorm(256),
    )
end

"""
    ResNetBlock()

Defines a single ResNet block - two rectified batch-normalized convolutional layers with a skip connection. 
"""
function ResNetBlock()
    return Flux.Chain(Flux.SkipConnection(ResNetInternal(), +), Flux.relu)
end
