#!/usr/bin/env python3
# Largest batch that fits, per kernel and precision. Doubles until it OOMs.
# Note this allocates in isolation, so it reports a bit more than the real
# scripts survive.
import argparse, torch, gc

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--stencil-n", type=int, default=384)
p.add_argument("--triad-n", type=int, default=68_000_000)
p.add_argument("--gemm-n", type=int, default=4096)
args = p.parse_args()

dev = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(dev)
name = torch.cuda.get_device_name(dev)
total = torch.cuda.get_device_properties(dev).total_memory / 2**30
print(f"# {name}  {total:.1f} GiB")


def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)


def try_stencil(B, N, dt):
    x = torch.randn(B, 1, N, N, N, device=dev, dtype=dt)
    w = torch.randn(1, 1, 3, 3, 3, device=dev, dtype=dt)
    out = torch.nn.functional.conv3d(x, w, padding=1)
    torch.cuda.synchronize()
    del x, w, out


def try_triad(B, N, dt):
    b = torch.randn(B, N, device=dev, dtype=dt)
    c = torch.randn(B, N, device=dev, dtype=dt)
    a = torch.empty_like(b)
    torch.add(b, c, alpha=2.0, out=a)
    torch.cuda.synchronize()
    del a, b, c


def try_gemm(B, N, dt):
    if B == 1:
        A = torch.randn(N, N, device=dev, dtype=dt)
        Bm = torch.randn(N, N, device=dev, dtype=dt)
        C = torch.empty(N, N, device=dev, dtype=dt)
        torch.matmul(A, Bm, out=C)
    else:
        A = torch.randn(B, N, N, device=dev, dtype=dt)
        Bm = torch.randn(B, N, N, device=dev, dtype=dt)
        C = torch.empty(B, N, N, device=dev, dtype=dt)
        torch.bmm(A, Bm, out=C)
    torch.cuda.synchronize()
    del A, Bm, C


for kern, fn, N, cap in (("stencil", try_stencil, args.stencil_n, 512),
                         ("triad", try_triad, args.triad_n, 512),
                         ("gemm", try_gemm, args.gemm_n, 1024)):
    for dtname, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        print(f"\n--- {kern} N={N} {dtname} ---")
        B, best = 1, 0
        while B <= cap:
            clear()
            try:
                fn(B, N, dt)
                peak = torch.cuda.max_memory_allocated(dev) / 2**30
                print(f"   B={B:<5} OK    peak {peak:6.2f} GiB "
                      f"({peak/total*100:5.1f}% of card)")
                best = B
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e).lower() else str(e)[:60]
                print(f"   B={B:<5} FAIL  {msg}")
                break
            B *= 2
        print(f"   -> largest power-of-two batch: {best}")
clear()
