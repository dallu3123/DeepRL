from lib import DQNNet, ptan

if __name__ == "__main__":
    net = DQNNet()
    print(net)
    target_net = ptan.agent.TargetNet(net)
    print("Main net: ", net.ff.weight)
    print("Target net: ", target_net.target_model.ff.weight)

    net.ff.weight.data += 1.0

    print("Update")
    print("Main net: ", net.ff.weight)
    print("Target net: ", target_net.target_model.ff.weight)

    """
    DQNNet(
        (ff): Linear(in_features=5, out_features=3, bias=True)
    )

    Main net:  Parameter containing:
    tensor([[ 0.1501, -0.1210, -0.0843, -0.1125,  0.3710],
            [ 0.2023, -0.0283, -0.0856,  0.1138, -0.0877],
            [ 0.4144,  0.2862,  0.3170, -0.0015, -0.3392]], requires_grad=True)
    Target net:  Parameter containing:
    tensor([[ 0.1501, -0.1210, -0.0843, -0.1125,  0.3710],
            [ 0.2023, -0.0283, -0.0856,  0.1138, -0.0877],
            [ 0.4144,  0.2862,  0.3170, -0.0015, -0.3392]], requires_grad=True)

    Update
    Main net:  Parameter containing:
    tensor([[1.1501, 0.8790, 0.9157, 0.8875, 1.3710],
            [1.2023, 0.9717, 0.9144, 1.1138, 0.9123],
            [1.4144, 1.2862, 1.3170, 0.9985, 0.6608]], requires_grad=True)
    Target net:  Parameter containing:
    tensor([[ 0.1501, -0.1210, -0.0843, -0.1125,  0.3710],
            [ 0.2023, -0.0283, -0.0856,  0.1138, -0.0877],
            [ 0.4144,  0.2862,  0.3170, -0.0015, -0.3392]], requires_grad=True)
    """

    target_net.sync()
    print("Sync")
    print("Main net: ", net.ff.weight)
    print("Target net: ", target_net.target_model.ff.weight)

    """
    Sync
    Main net:  Parameter containing:
    tensor([[1.1501, 0.8790, 0.9157, 0.8875, 1.3710],
            [1.2023, 0.9717, 0.9144, 1.1138, 0.9123],
            [1.4144, 1.2862, 1.3170, 0.9985, 0.6608]], requires_grad=True)

    Target net:  Parameter containing:
    tensor([[1.1501, 0.8790, 0.9157, 0.8875, 1.3710],
            [1.2023, 0.9717, 0.9144, 1.1138, 0.9123],
            [1.4144, 1.2862, 1.3170, 0.9985, 0.6608]], requires_grad=True)
    """