# Checks whether an on-device counter can be read back under PopTorch.
# Increments a registered buffer once per call, then compares it against the
# host call count.
import torch, poptorch, controls as C
N, CALLS = 64, 25

class S(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = torch.nn.Conv3d(1,1,3,padding=1,bias=False)
        self.register_buffer("grid", torch.randn(1,1,n,n,n))
        self.register_buffer("out",  torch.zeros(1,1,n,n,n))
        self.register_buffer("count", torch.zeros(1, dtype=torch.int32))
    def forward(self, tick):
        self.out.copy_(self.conv(self.grid)); self.count += 1
        return self.count.float() * tick

C.apply_controls(verbose=False)
m = S(N)
pop = poptorch.inferenceModel(m, C.poptorch_options(device_iterations=1))
tick = torch.ones(1,1)
pop.compile(tick)

pop.copyWeightsToHost(); before = float(m.count.item())
last = None
for _ in range(CALLS):
    last = pop(tick)
pop.copyWeightsToHost(); after = float(m.count.item())

print("  calls issued            :", CALLS)
print("  copyWeightsToHost delta : %.1f   <- what the official script uses" % (after-before))
print("  returned tensor value   : %.1f   <- counter as the DEVICE reports it" % float(last.flatten()[0]))
