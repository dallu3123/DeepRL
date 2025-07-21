import torch
import torch.nn as nn

class OnModule(nn.Module):
    def __init__(self, num_inputs, num_outputs, dropout_prob=0.3):
        super(OnModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.pipe(x)

if __name__ == "__main__":
    model = OnModule(num_inputs=2, num_outputs=3)
    print(model)

    v = torch.FloatTensor([[2, 3]])
    out = model(v)
    print(out)
    print(out.shape)

    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to('cuda'))
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)  # 'mps' or 'cpu'

''' result
OnModule(
  (pipe): Sequential(
    (0): Linear(in_features=2, out_features=5, bias=True)
    (1): ReLU()
    (2): Linear(in_features=5, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=3, bias=True)
    (5): Dropout(p=0.3, inplace=False)
    (6): Softmax(dim=1)
  )
)
tensor([[0.2558, 0.3509, 0.3933]], grad_fn=<SoftmaxBackward0>)
torch.Size([1, 3])
Cuda's availability is False
mps
'''