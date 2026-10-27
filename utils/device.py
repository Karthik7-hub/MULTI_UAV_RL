"""
Unified Device and Hardware Management for Multi-UAV DRL Research.
Provides reliable CPU/GPU device selection, hardware diagnostics, and tensor placement.
"""

import sys
import torch
from typing import Optional, Dict, Any

class DeviceManager:
    @staticmethod
    def select_device(device_str: str = "auto") -> torch.device:
        """
        Selects compute device based on preference: 'auto', 'cuda', 'mps', 'cpu'.
        Provides native Apple Silicon (Metal Performance Shaders / MPS) support on macOS.
        Raises informative RuntimeError if requested accelerator is unavailable.
        """
        device_str = device_str.lower().strip()
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
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
            print(f"  CUDA Version       : {torch.version.cuda}")
        print(f"  Active Device      : {device.type.upper()}{(':' + str(device.index)) if device.index is not None else ''}")
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
