"""
Two checks on the IPU stencil: the result against a float64 CPU reference, and
whether the sample count scales with run time.
"""
import time, torch, poptorch, controls as C
from ipu_stencil import StencilResident

class VerifyStencil(StencilResident):
    def forward(self, tick):
        self.out.copy_(self.conv(self.grid))          # identical work
        return self.out.sum().reshape(1, 1) * tick    # checksum instead of count

# A. correctness against a CPU reference
print("=== A. output correctness (N=128, fp32) ===")
C.apply_controls(verbose=False)
m = VerifyStencil(128)
with torch.no_grad():
    cpu32 = m.conv(m.grid).sum().item()
    cpu64 = torch.nn.functional.conv3d(
        m.grid.double(), m.conv.weight.double(), padding=1).sum().item()
pop = poptorch.inferenceModel(m, C.poptorch_options(device_iterations=1))
tick = torch.ones(1, 1)
t0 = time.time(); pop.compile(tick); print(f"  compile {time.time()-t0:.1f}s")
ipu = float(pop(tick).flatten()[0])
pop.detachFromDevice()
den = abs(cpu64) if cpu64 else 1.0
print(f"  cpu  float64 reference : {cpu64:.6f}")
print(f"  cpu  float32           : {cpu32:.6f}   rel err {abs(cpu32-cpu64)/den:.3e}")
print(f"  IPU  float32           : {ipu:.6f}   rel err {abs(ipu-cpu64)/den:.3e}")
print(f"  -> IPU matches CPU to {abs(ipu-cpu64)/den:.3e} relative")

# B. does the sample count scale with time
print("\n=== B. linear scaling (N=384, fp32, the official configuration) ===")
C.apply_controls(verbose=False)
m2 = StencilResident(384)
pop2 = poptorch.inferenceModel(m2, C.poptorch_options(device_iterations=1))
t0 = time.time(); pop2.compile(tick); print(f"  compile {time.time()-t0:.1f}s (excluded)")
for _ in range(10):
    pop2(tick)
print("  warmup 10 calls (excluded)")
res = {}
for secs in (45.0, 90.0):
    calls = 0; s = time.time()
    while time.time() - s < secs:
        pop2(tick); calls += 1
    el = time.time() - s
    res[secs] = (calls, el)
    print(f"  {secs:5.0f}s -> {calls:6d} calls  ({calls/el:.3f} samples/s)")
pop2.detachFromDevice()
(c1, e1), (c2, e2) = res[45.0], res[90.0]
print(f"  sample ratio  {c2/c1:.4f}   (perfect = {e2/e1:.4f})")
print(f"  rate drift    {abs((c2/e2)/(c1/e1)-1)*100:.3f}%")
