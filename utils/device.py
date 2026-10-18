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
        Selects compute device based on preference: 'auto', 'cuda', 'cpu'.
        Raises informative RuntimeError if 'cuda' is requested but unavailable.
        """
        device_str = device_str.lower().strip()
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        elif device_str == "cuda" or device_str.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA compute device was explicitly requested ('{device_str}'), but "
                    f"torch.cuda.is_available() is False. Please install a CUDA-enabled PyTorch "
                    f"build or use '--device cpu' or '--device auto'."
                )
            return torch.device(device_str)
        elif device_str == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device specification: '{device_str}'. Choose from 'auto', 'cuda', 'cpu'.")

    @staticmethod
    def print_diagnostics(device: torch.device, seed: Optional[int] = None, extra_config: Optional[Dict[str, Any]] = None):
        """Prints a standardized research hardware banner at startup."""
        print("=" * 78)
        print(" HARDWARE & EXPERIMENTAL EXECUTION DIAGNOSTICS")
        print("=" * 78)
        print(f"  PyTorch Version    : {torch.__version__}")
        print(f"  CUDA Available     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device Count  : {torch.cuda.device_count()}")
            print(f"  CUDA Device Name   : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version       : {torch.version.cuda}")
        print(f"  Active Device      : {device.type.upper()}{(':' + str(device.index)) if device.index is not None else ''}")
        if device.type == "cpu":
            print(f"  Device Detail      : CPU Execution (Thread Count: {torch.get_num_threads()})")
        else:
            print(f"  Device Detail      : GPU Execution ({torch.cuda.get_device_name(device)})")
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
