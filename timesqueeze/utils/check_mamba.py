"""Smoke test for the real Mamba layers (Linux / WSL2 + CUDA): forward and backward through one layer on the GPU."""
import torch


def main():
  try:
    import causal_conv1d
    import mamba_ssm
    from mamba_ssm import Mamba
  except ImportError as e:
    raise SystemExit(f"mamba-ssm / causal-conv1d are not installed ({e}). They only install on Linux/WSL2; see README.")

  print("PyTorch:", torch.__version__)
  print("CUDA:", torch.version.cuda, "| device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
  print("causal-conv1d:", causal_conv1d.__version__)
  print("mamba-ssm:", mamba_ssm.__version__)

  model = Mamba(d_model=384, d_state=128, d_conv=4, expand=4).cuda()
  x = torch.randn(2, 512, 384, device="cuda", requires_grad=True)
  y = model(x)
  y.square().mean().backward()                      # the backward kernels must work too, they are what trains
  torch.cuda.synchronize()

  with torch.autocast("cuda", dtype=torch.bfloat16):
    yb = model(x.detach())
  print("Output:", tuple(y.shape), "| device:", y.device)
  print("Finite forward:", torch.isfinite(y).all().item(), "| finite grads:",
        all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
        "| finite bf16:", torch.isfinite(yb).all().item())

  # causality: changing the input after position t0 must not change outputs before t0
  with torch.no_grad():
    x2 = x.detach().clone()
    x2[:, 300:] += 1.0
    d = (model(x.detach())[:, :300] - model(x2)[:, :300]).abs().max().item()
  print("Causal (max change before the edit):", d)


if __name__ == "__main__":
  main()
