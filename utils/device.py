"""
Unified Device and Hardware Management for Multi-UAV DRL Research.
Provides reliable CPU/GPU device selection, hardware diagnostics, and tensor placement.
"""

import sys
import warnings
import torch
from typing import Optional, Dict, Any, Tuple

class DeviceManager:
    _cuda_fallback_reason: Optional[str] = None

    @staticmethod
    def _probe_cuda_device(device_idx: int = 0) -> Tuple[bool, str]:
        """Verify that the CUDA device supports compiled kernels in the installed PyTorch build."""
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is False"
        try:
            dev_name = torch.cuda.get_device_name(device_idx)
            cap_major, cap_minor = torch.cuda.get_device_capability(device_idx)
            arch_tag = f"sm_{cap_major}{cap_minor}"
            arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []

            # Pre-check compiled CUDA arch list to cleanly reject legacy GPUs (e.g. Quadro P2000 sm_61 on PyTorch >=2.7)
            if arch_list:
                supported_caps = []
                for arch in arch_list:
                    if arch.startswith("sm_") or arch.startswith("compute_"):
                        digits = arch.split("_", 1)[1].rstrip("af")
                        if digits.isdigit() and len(digits) >= 2:
                            supported_caps.append((int(digits[:-1]), int(digits[-1])))
                if supported_caps and (cap_major, cap_minor) < min(supported_caps):
                    return (
                        False,
                        f"{dev_name} ({arch_tag}) is older than the minimum supported CUDA architecture "
                        f"in PyTorch {torch.__version__} (built for {', '.join(arch_list)})."
                    )

            # Execute tiny tensor matmul + activation kernel on GPU to verify runtime kernel compatibility
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                test_dev = torch.device(f"cuda:{device_idx}")
                a = torch.ones((2, 4), dtype=torch.float32, device=test_dev)
                b = torch.ones((4, 2), dtype=torch.float32, device=test_dev)
                _ = torch.tanh(a @ b).sum().item()
                for w in caught:
                    msg = str(w.message)
                    if "not compatible" in msg or "no kernel image" in msg or arch_tag in msg:
                        return False, f"{dev_name} ({arch_tag}) CUDA compatibility warning: {msg}"

            return True, f"{dev_name} ({arch_tag})"
        except Exception as e:
            return False, f"CUDA kernel execution probe failed: {e}"

    @staticmethod
    def select_device(device_str: str = "auto") -> torch.device:
        """
        Selects compute device based on preference: 'auto', 'cuda', 'mps', 'cpu'.
        Probes CUDA kernel compatibility so legacy GPUs (e.g., Quadro P2000 / sm_61 on newer PyTorch)
        cleanly fall back to CPU under 'auto' instead of crashing mid-training.
        Raises informative RuntimeError if an explicitly requested accelerator is unusable.
        """
        DeviceManager._cuda_fallback_reason = None
        device_str = device_str.lower().strip()
        if device_str == "auto":
            if torch.cuda.is_available():
                ok, reason = DeviceManager._probe_cuda_device(0)
                if ok:
                    return torch.device("cuda:0")
                DeviceManager._cuda_fallback_reason = reason
                print(
                    f"[DeviceManager] Notice: Detected CUDA GPU cannot execute kernels on this PyTorch build "
                    f"({reason}). Automatically falling back to CPU execution."
                )
                return torch.device("cpu")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif device_str == "cuda" or device_str.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA compute device was explicitly requested ('{device_str}'), but "
                    f"torch.cuda.is_available() is False. Please install a CUDA-enabled PyTorch "
                    f"build or use '--device cpu' or '--device auto'."
                )
            idx = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
            ok, reason = DeviceManager._probe_cuda_device(idx)
            if not ok:
                raise RuntimeError(
                    f"CUDA compute device was explicitly requested ('{device_str}'), but it is incompatible "
                    f"with the current PyTorch build ({reason}). "
                    f"Use '--device auto' or '--device cpu' (recommended for small-MLP MARL), or for Pascal/sm_61 "
                    f"GPUs like Quadro P2000 install PyTorch <=2.5.1 with CUDA 11.8/12.1."
                )
            return torch.device(device_str)
        elif device_str == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    f"Apple Silicon MPS device was explicitly requested ('{device_str}'), but "
                    f"torch.backends.mps.is_available() is False. Please ensure you are on macOS 12.3+ "
                    f"with an Apple Silicon chip (M1/M2/M3/M4) or use '--device cpu' or '--device auto'."
                )
            return torch.device("mps")
        elif device_str == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device specification: '{device_str}'. Choose from 'auto', 'cuda', 'mps', 'cpu'.")

    @staticmethod
    def print_diagnostics(device: torch.device, seed: Optional[int] = None, extra_config: Optional[Dict[str, Any]] = None):
        """Prints a standardized research hardware banner at startup."""
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print("=" * 78)
        print(" HARDWARE & EXPERIMENTAL EXECUTION DIAGNOSTICS")
        print("=" * 78)
        print(f"  PyTorch Version    : {torch.__version__}")
        print(f"  CUDA Available     : {torch.cuda.is_available()}")
        print(f"  Apple Silicon MPS  : {mps_avail}")
        if torch.cuda.is_available():
            print(f"  CUDA Device Count  : {torch.cuda.device_count()}")
            print(f"  CUDA Device Name   : {torch.cuda.get_device_name(0)}")
            cap_maj, cap_min = torch.cuda.get_device_capability(0)
            print(f"  CUDA Compute Cap.  : sm_{cap_maj}{cap_min}")
            print(f"  CUDA Version       : {torch.version.cuda}")
        print(f"  Active Device      : {device.type.upper()}{(':' + str(device.index)) if device.index is not None else ''}")
        if DeviceManager._cuda_fallback_reason and device.type == "cpu":
            print(f"  CUDA Fallback      : {DeviceManager._cuda_fallback_reason}")
        if device.type == "cpu":
            print(f"  Device Detail      : CPU Execution (Thread Count: {torch.get_num_threads()})")
        elif device.type == "mps":
            print(f"  Device Detail      : Apple Silicon GPU Execution (Metal Performance Shaders / MPS)")
        elif device.type == "cuda":
            print(f"  Device Detail      : CUDA GPU Execution ({torch.cuda.get_device_name(device)})")
        if seed is not None:
            print(f"  Experiment Seed    : {seed}")
        if extra_config:
            for k, v in extra_config.items():
                print(f"  Config [{k:16s}]: {v}")
        print("=" * 78)

    @staticmethod
    def verify_device(model: torch.nn.Module, expected_device: torch.device):
        """Verifies all parameters of a model are placed on the expected device."""
        for name, param in model.named_parameters():
            if param.device.type != expected_device.type:
                raise RuntimeError(
                    f"Parameter '{name}' is on device '{param.device}', expected '{expected_device}'."
                )

    @staticmethod
    def safe_save_checkpoint(state_dict: Dict[str, Any], path: str):
        """Saves checkpoint cleanly."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    @staticmethod
    def safe_load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
        """Loads checkpoint mapped directly to the target device."""
        return torch.load(path, map_location=device)
