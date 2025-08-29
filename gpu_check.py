import torch
import sys
import platform
import os

def run_command(command):
    """Helper function to run a shell command and capture output."""
    try:
        output = os.popen(command).read().strip()
        return output
    except Exception as e:
        return f"Command failed: {e}"

def check_gpu_compatibility():
    """
    Checks for GPU compatibility with PyTorch and other libraries used in v1.py.
    """
    print("="*60)
    print("GPU and PyTorch Compatibility Check")
    print("="*60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print("\n" + "-"*25 + " PyTorch CUDA Info " + "-"*25)

    # 1. Check if PyTorch sees the GPU
    is_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {is_available}")

    if is_available:
        print(f"  - CUDA version PyTorch was built with: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"  - Number of GPUs found by PyTorch: {device_count}")
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("  - PyTorch cannot detect a CUDA-enabled GPU.")
        print("  - This is the primary reason the script is not using your GPU.")

    print("\n" + "-"*23 + " System NVIDIA Info " + "-"*24)

    # 2. Check system's NVIDIA driver and CUDA version using nvidia-smi
    nvidia_smi_output = run_command("nvidia-smi")

    if "command not found" in nvidia_smi_output.lower() or "failed" in nvidia_smi_output.lower():
        print("`nvidia-smi` command not found. This means:")
        print("  1. You may not have an NVIDIA GPU.")
        print("  2. The NVIDIA driver is not installed correctly or not in the system's PATH.")
    else:
        print("`nvidia-smi` output:")
        print(nvidia_smi_output)

    print("\n" + "-"*28 + " Summary " + "-"*29)

    if is_available:
        print("✅ SUCCESS: Your GPU is compatible and correctly configured for PyTorch.")
        print("The `v1.py` script should be able to use your GPU.")
        print("If it's still not being used, ensure you are running it with the correct Python environment.")
    else:
        print("❌ FAILURE: PyTorch cannot use your GPU.")
        print("\nPossible Reasons & Solutions:")
        if "command not found" in nvidia_smi_output.lower():
            print("  - NVIDIA Driver Issue: No NVIDIA driver was found. Please install the appropriate driver for your GPU.")
        else:
            print("  - Installation Mismatch: The most common issue is a mismatch between your NVIDIA driver version and the CUDA version required by your PyTorch install.")
            print("    - Check the `nvidia-smi` output above for your driver's supported CUDA version.")
            print("    - Re-install PyTorch, making sure to select the command that matches your CUDA version from the official website: https://pytorch.org/get-started/locally/")
            print("    - For example, if your driver supports CUDA 11.8, you would use a command like:")
            print("      `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`")
        print("  - Environment Issue: Ensure you have activated the correct conda/virtual environment ('voice') before running your script.")

if __name__ == "__main__":
    check_gpu_compatibility()
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# import sounddevice as sd
# print(sd.query_devices())
