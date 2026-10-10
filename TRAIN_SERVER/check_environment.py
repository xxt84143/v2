"""Check the interpreter and perform one real U-Net optimizer step."""
import argparse
import json
from pathlib import Path
import platform
import sys

from model_factory import build_model
from training_core import HERE, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    import monai
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check nvidia-smi and install the matching official PyTorch wheel")
    model = build_model(config["model"], 10, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    x = torch.randn((1, 10, 128, 128), device=device)
    prediction = model(x)
    if list(prediction.shape) != [1, 1, 128, 128]:
        raise ValueError("Incorrect model output shape")
    loss = prediction.float().square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite forward loss")
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("Nonfinite model gradient")
    optimizer.step()
    report = {"status": "ok", "python": sys.version, "interpreter": sys.executable,
              "platform": platform.platform(), "torch": str(torch.__version__),
              "monai": monai.__version__, "cuda_runtime": torch.version.cuda,
              "device": str(device), "forward_backward_optimizer": "passed",
              "parameters": sum(p.numel() for p in model.parameters())}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(gpu=props.name, gpu_memory_gib=props.total_memory / 1024**3,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
