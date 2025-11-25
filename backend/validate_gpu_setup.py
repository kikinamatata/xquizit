#!/usr/bin/env python3
"""
GPU Setup Validation Script for xquizit Backend

This script validates that all GPU optimization components are properly installed:
- CUDA availability and version
- PyTorch CUDA support
- Triton availability
- Flash Attention 2
- torch.compile() functionality
- Parler-TTS model loading with optimizations

Usage:
    python3 validate_gpu_setup.py

Returns exit code 0 if all critical components pass, 1 if any failures.
"""

import sys
import platform
from typing import Dict, Any, Tuple
import time


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print('=' * 70)


def print_success(text: str) -> None:
    """Print success message in green."""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """Print error message in red."""
    print(f"✗ {text}")


def print_warning(text: str) -> None:
    """Print warning message in yellow."""
    print(f"⚠ {text}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"ℹ {text}")


def is_wsl() -> bool:
    """Check if running in WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def check_wsl_config() -> Tuple[bool, Dict[str, Any]]:
    """Check WSL configuration if running in WSL."""
    if not is_wsl():
        return True, {}

    print_header("WSL Configuration Check")

    results = {}
    config_ok = True

    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                    total_gb = total_kb / (1024 * 1024)
                    results['total_memory_gb'] = total_gb

                    print(f"WSL Total Memory: {total_gb:.1f} GB")

                    if total_gb < 16:
                        print_warning(f"WSL memory is low ({total_gb:.1f}GB). Recommended: 20GB")
                        print_info("Create/update C:\\Users\\YOUR_USERNAME\\.wslconfig with:")
                        print_info("  [wsl2]")
                        print_info("  memory=20GB")
                        print_info("  pageReporting=false")
                        print_info("Then restart WSL: wsl --shutdown")
                        config_ok = False
                    else:
                        print_success(f"WSL memory: {total_gb:.1f}GB (adequate)")
                    break
    except Exception as e:
        print_warning(f"Could not check WSL memory: {e}")

    # Check if .wslconfig exists (check common paths)
    import os
    wslconfig_found = False
    possible_paths = [
        '/mnt/c/Users/*/wslconfig',
    ]

    # Try to find Windows username
    try:
        import subprocess
        result = subprocess.run(['cmd.exe', '/c', 'echo', '%USERNAME%'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            wslconfig_path = f'/mnt/c/Users/{username}/.wslconfig'
            if os.path.exists(wslconfig_path):
                wslconfig_found = True
                print_success(f".wslconfig found at: {wslconfig_path}")
    except:
        pass

    if not wslconfig_found:
        print_warning(".wslconfig not found")
        print_info("For optimal performance, create C:\\Users\\YOUR_USERNAME\\.wslconfig")

    results['wsl_detected'] = True
    results['config_ok'] = config_ok

    return config_ok, results


def validate_system_info() -> Dict[str, Any]:
    """Validate basic system information."""
    print_header("System Information")

    results = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
    }

    print(f"Python version: {results['python_version']}")
    print(f"Platform: {results['platform']}")
    print(f"Processor: {results['processor']}")

    # Check Python version
    if results['python_version'].startswith('3.12'):
        print_success("Python 3.12 detected (recommended)")
    elif results['python_version'].startswith('3.10') or results['python_version'].startswith('3.11'):
        print_success(f"Python {results['python_version']} detected (compatible)")
    else:
        print_warning(f"Python {results['python_version']} may have compatibility issues")

    return results


def validate_pytorch() -> Tuple[bool, Dict[str, Any]]:
    """Validate PyTorch installation and CUDA support."""
    print_header("PyTorch Validation")

    try:
        import torch
        results = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else None,
        }

        print(f"PyTorch version: {results['torch_version']}")
        print(f"CUDA available: {results['cuda_available']}")

        if not results['cuda_available']:
            print_error("CUDA is not available in PyTorch!")
            print_info("You may have installed CPU-only PyTorch.")
            print_info("Reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu128")
            return False, results

        print_success(f"CUDA version: {results['cuda_version']}")
        print_success(f"cuDNN version: {results['cudnn_version']}")
        print_success(f"cuDNN enabled: {results['cudnn_enabled']}")

        # Check CUDA version
        if results['cuda_version'] and results['cuda_version'].startswith('12.'):
            print_success("CUDA 12.x detected (optimal for RTX 5060 Ti)")
        else:
            print_warning(f"CUDA {results['cuda_version']} detected. Recommended: CUDA 12.8+")

        # Get GPU info
        if torch.cuda.is_available():
            results['gpu_count'] = torch.cuda.device_count()
            results['gpu_name'] = torch.cuda.get_device_name(0)
            results['gpu_capability'] = torch.cuda.get_device_capability(0)
            results['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

            print(f"\nGPU Information:")
            print(f"  Device count: {results['gpu_count']}")
            print(f"  Device 0: {results['gpu_name']}")
            print(f"  Compute capability: {results['gpu_capability'][0]}.{results['gpu_capability'][1]}")
            print(f"  Total memory: {results['gpu_memory_gb']:.2f} GB")

            # Check compute capability (5060 Ti should be 8.9 or 9.x for Blackwell)
            if results['gpu_capability'][0] >= 8:
                print_success(f"Compute capability {results['gpu_capability'][0]}.{results['gpu_capability'][1]} supports all optimizations")
            else:
                print_warning(f"Compute capability {results['gpu_capability'][0]}.{results['gpu_capability'][1]} may not support all features")

        return True, results

    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False, {}
    except Exception as e:
        print_error(f"Error validating PyTorch: {e}")
        return False, {}


def validate_triton() -> Tuple[bool, Dict[str, Any]]:
    """Validate Triton installation."""
    print_header("Triton Validation")

    try:
        import triton
        results = {
            'triton_version': triton.__version__,
        }

        print(f"Triton version: {results['triton_version']}")
        print_success("Triton is available (required for torch.compile())")

        # Try to import triton.language
        try:
            import triton.language as tl
            print_success("Triton language module available")
        except ImportError:
            print_warning("Triton language module not available (may affect advanced features)")

        return True, results

    except ImportError:
        print_error("Triton not installed!")
        print_info("Install with: pip install triton")
        print_warning("torch.compile() will NOT work without Triton")
        return False, {}
    except Exception as e:
        print_error(f"Error validating Triton: {e}")
        return False, {}


def validate_flash_attention() -> Tuple[bool, Dict[str, Any]]:
    """Validate Flash Attention 2 installation."""
    print_header("Flash Attention 2 Validation")

    try:
        import flash_attn
        results = {
            'flash_attn_version': flash_attn.__version__,
        }

        print(f"Flash Attention version: {results['flash_attn_version']}")
        print_success("Flash Attention 2 is available (1.4x speedup)")

        # Try to import flash_attn_interface
        try:
            from flash_attn import flash_attn_func
            print_success("Flash Attention functions available")
        except ImportError as e:
            print_warning(f"Some Flash Attention functions not available: {e}")

        return True, results

    except ImportError:
        print_warning("Flash Attention 2 not installed (optional)")
        print_info("You'll still get ~4x speedup from torch.compile()")
        print_info("To install: pip install flash-attn --no-build-isolation")
        return False, {}
    except Exception as e:
        print_error(f"Error validating Flash Attention: {e}")
        return False, {}


def validate_torch_compile() -> Tuple[bool, Dict[str, Any]]:
    """Validate torch.compile() functionality."""
    print_header("torch.compile() Validation")

    try:
        import torch

        if not torch.cuda.is_available():
            print_error("CUDA not available, cannot test torch.compile()")
            return False, {}

        print_info("Testing torch.compile() with simple model...")

        # Create a simple model
        @torch.compile
        def simple_model(x):
            return torch.nn.functional.relu(x * 2 + 1)

        # Test on CPU first
        x_cpu = torch.randn(100, 100)
        start = time.time()
        result_cpu = simple_model(x_cpu)
        cpu_time = time.time() - start
        print(f"  CPU execution: {cpu_time*1000:.2f}ms")

        # Test on GPU
        x_gpu = torch.randn(1000, 1000).cuda()

        # Warmup (triggers compilation)
        print_info("Warming up (first run compiles the model)...")
        start = time.time()
        _ = simple_model(x_gpu)
        torch.cuda.synchronize()
        compile_time = time.time() - start
        print(f"  First run (compile): {compile_time*1000:.2f}ms")

        # Actual timed run
        start = time.time()
        result_gpu = simple_model(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  Second run (compiled): {gpu_time*1000:.2f}ms")

        print_success("torch.compile() is working correctly")
        print_info(f"Compilation overhead: {compile_time*1000:.2f}ms (one-time cost)")

        results = {
            'compile_working': True,
            'compile_time_ms': compile_time * 1000,
            'gpu_time_ms': gpu_time * 1000,
        }

        return True, results

    except Exception as e:
        print_error(f"torch.compile() test failed: {e}")
        print_warning("This may be due to missing Triton or CUDA issues")
        return False, {}


def validate_parler_tts() -> Tuple[bool, Dict[str, Any]]:
    """Validate Parler-TTS installation (without loading full model)."""
    print_header("Parler-TTS Validation")

    try:
        # Check if transformers is available
        import transformers
        print(f"Transformers version: {transformers.__version__}")

        # Check if parler_tts imports
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            print_success("Parler-TTS package is available")

            # Check if required classes are available
            print_success("ParlerTTSForConditionalGeneration class available")

            results = {
                'parler_tts_available': True,
                'transformers_version': transformers.__version__,
            }

            print_info("Note: Not loading full model to save time and memory")
            print_info("Full model test will happen on first server startup")

            return True, results

        except ImportError:
            print_error("Parler-TTS not installed!")
            print_info("Install with: pip install git+https://github.com/huggingface/parler-tts.git")
            return False, {'parler_tts_available': False}

    except ImportError as e:
        print_error(f"Transformers not installed: {e}")
        return False, {}
    except Exception as e:
        print_error(f"Error validating Parler-TTS: {e}")
        return False, {}


def generate_report(all_results: Dict[str, Any]) -> None:
    """Generate final validation report."""
    print_header("Validation Summary")

    critical_checks = [
        ('pytorch', 'PyTorch with CUDA'),
        ('triton', 'Triton (for torch.compile)'),
        ('parler_tts', 'Parler-TTS'),
    ]

    optional_checks = [
        ('flash_attn', 'Flash Attention 2'),
        ('torch_compile', 'torch.compile() test'),
    ]

    print("\nCritical Components:")
    all_critical_passed = True
    for key, name in critical_checks:
        if all_results.get(key, False):
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
            all_critical_passed = False

    print("\nOptional Components:")
    for key, name in optional_checks:
        if all_results.get(key, False):
            print_success(f"{name}: PASSED")
        else:
            print_warning(f"{name}: NOT AVAILABLE (reduced performance)")

    print("\nPerformance Expectations:")

    if all_results.get('pytorch') and all_results.get('triton'):
        if all_results.get('flash_attn') and all_results.get('torch_compile'):
            print_success("All optimizations available: ~15-20x faster than CPU")
            print_info("  - Flash Attention 2: ~1.4x speedup")
            print_info("  - torch.compile(): ~4x speedup")
            print_info("  - Combined: ~5-6x speedup over base GPU")
            print_info("  - Expected TTFA: <200ms")
        elif all_results.get('torch_compile'):
            print_success("torch.compile() available: ~4-6x faster than CPU")
            print_info("  - Expected TTFA: ~250ms")
            print_warning("  - Install Flash Attention for additional 1.4x speedup")
        else:
            print_warning("Basic GPU support only: ~2-3x faster than CPU")
            print_info("  - Expected TTFA: ~400ms")
            print_warning("  - Install Triton for torch.compile() (4x speedup)")
    else:
        print_error("GPU optimizations not available")
        print_info("Expected performance: CPU-only (~800ms TTFA)")

    print("\nNext Steps:")
    if all_critical_passed:
        print_success("All critical components validated!")
        print_info("1. Ensure .env has: PARLER_ENABLE_COMPILE=true")
        print_info("2. Start server: uvicorn app.main:app --reload")
        print_info("3. Monitor GPU: watch -n 1 nvidia-smi")

        if not all_results.get('flash_attn'):
            print_warning("4. Consider installing Flash Attention 2 for additional speedup")
    else:
        print_error("Some critical components failed validation")
        print_info("1. Review error messages above")
        print_info("2. Re-run setup_linux.sh if needed")
        print_info("3. Check documentation in docs/GPU_SETUP_LINUX.md")

    print("\n" + "=" * 70)

    return all_critical_passed


def main() -> int:
    """Main validation function."""
    print_header("xquizit Backend GPU Setup Validation")
    print("This script validates GPU optimization components\n")

    all_results = {}

    # Run all validations
    all_results['system'] = validate_system_info()

    # Check WSL configuration if running in WSL
    if is_wsl():
        all_results['wsl_config'], wsl_results = check_wsl_config()
        print_info("Running in WSL environment - using Windows CUDA passthrough")

    all_results['pytorch'], pytorch_results = validate_pytorch()
    all_results['triton'], triton_results = validate_triton()
    all_results['flash_attn'], flash_results = validate_flash_attention()
    all_results['torch_compile'], compile_results = validate_torch_compile()
    all_results['parler_tts'], parler_results = validate_parler_tts()

    # Generate final report
    all_passed = generate_report(all_results)

    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
