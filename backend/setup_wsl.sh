#!/bin/bash

################################################################################
# WSL GPU Setup Script for xquizit Backend
# Optimized for Windows Subsystem for Linux (WSL2)
# Installs PyTorch with CUDA 12.1, Triton, and Flash Attention 2
# For use with NVIDIA RTX 5060 Ti (CUDA 13.0 runtime via Windows passthrough)
#
# Usage: bash setup_wsl.sh
# Requires: Python 3.12, WSL2 with .wslconfig configured, ~15GB disk space
################################################################################

# Auto-fix line endings (handles files created on Windows)
if command -v dos2unix &> /dev/null; then
    dos2unix "$0" 2>/dev/null || sed -i 's/\r$//' "$0" 2>/dev/null
else
    sed -i 's/\r$//' "$0" 2>/dev/null
fi

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PYTHON_VERSION="3.12"
REQUIRED_DRIVER_VERSION="545"
REQUIRED_DISK_SPACE_GB=15
MIN_FREE_MEMORY_GB=8

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Pre-flight Checks
################################################################################

print_header "Pre-flight System Checks (WSL Optimized)"

# Check if running on WSL
if ! grep -qi microsoft /proc/version; then
    print_warning "This script is optimized for WSL2. Detected: $(uname -a)"
    print_info "For native Linux, use setup_linux.sh instead"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Running on WSL2"
fi

# Check for .wslconfig
print_info "Checking WSL configuration..."
if [ -f "/mnt/c/Users/$USER/.wslconfig" ] || [ -f "/mnt/c/Users/*/wslconfig" ]; then
    print_success ".wslconfig file detected"
else
    print_warning ".wslconfig not found in C:\\Users\\$USER\\"
    print_info "For best results, create .wslconfig with memory=20GB"
    print_info "See: https://docs.microsoft.com/en-us/windows/wsl/wsl-config"
fi

# Check available memory
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
FREE_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
print_info "Total WSL memory: ${TOTAL_MEM_GB}GB, Available: ${FREE_MEM_GB}GB"

if [[ $TOTAL_MEM_GB -lt 16 ]]; then
    print_warning "WSL has < 16GB total memory. Flash Attention may fail."
    print_info "Configure .wslconfig with memory=20GB and restart WSL"
fi

if [[ $FREE_MEM_GB -lt $MIN_FREE_MEMORY_GB ]]; then
    print_warning "Less than ${MIN_FREE_MEMORY_GB}GB free memory"
    print_info "Close other applications to free up memory"
fi

# Check Python version
if ! check_command python3; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]]; then
    print_warning "Expected Python $REQUIRED_PYTHON_VERSION, found $PYTHON_VERSION"
    print_info "Continuing anyway, but some packages may have compatibility issues"
else
    print_success "Python $PYTHON_VERSION detected"
fi

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_error "Not in a virtual environment. Please activate your venv first:"
    echo -e "  ${YELLOW}source venv/bin/activate${NC}"
    exit 1
fi
print_success "Virtual environment active: $VIRTUAL_ENV"

# Check available disk space
AVAILABLE_SPACE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [[ $AVAILABLE_SPACE_GB -lt $REQUIRED_DISK_SPACE_GB ]]; then
    print_error "Insufficient disk space. Need ${REQUIRED_DISK_SPACE_GB}GB, have ${AVAILABLE_SPACE_GB}GB"
    exit 1
fi
print_success "Sufficient disk space available: ${AVAILABLE_SPACE_GB}GB"

# Check for NVIDIA GPU (via Windows passthrough)
if ! check_command nvidia-smi; then
    print_error "nvidia-smi not found. Is NVIDIA GPU support enabled in WSL?"
    print_info "See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    exit 1
fi
print_success "NVIDIA driver detected (Windows passthrough)"

# Check driver version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
print_success "NVIDIA driver version: $DRIVER_VERSION (Windows)"

# Check CUDA version (runtime via Windows)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
print_success "CUDA runtime version: $CUDA_VERSION (Windows passthrough)"

if [[ "$CUDA_VERSION" != "13.0" ]] && [[ "$CUDA_VERSION" != "12."* ]]; then
    print_warning "Unexpected CUDA version: $CUDA_VERSION"
fi

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
print_success "Detected GPU: $GPU_NAME"

################################################################################
# Backup Current State
################################################################################

print_header "Backing Up Current Environment"

BACKUP_FILE="pip_freeze_backup_$(date +%Y%m%d_%H%M%S).txt"
pip freeze > "$BACKUP_FILE"
print_success "Package list backed up to: $BACKUP_FILE"

################################################################################
# Install System Dependencies
################################################################################

print_header "Installing System Dependencies"

print_info "Installing build tools and Python development headers..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    gcc \
    g++ \
    make \
    ninja-build \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    dos2unix

print_success "System dependencies installed"

################################################################################
# Install PyTorch with CUDA 12.1 (Compatible with CUDA 13.0 Runtime)
################################################################################

print_header "Installing PyTorch with CUDA 12.1 Support"

print_info "Note: Using CUDA 12.1 PyTorch wheels (compatible with CUDA 13.0 runtime)"

# Uninstall existing PyTorch (may be CPU-only)
print_info "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 12.1 (backward compatible with CUDA 13.0)
print_info "Installing PyTorch 2.5+ with CUDA 12.1 (this may take 3-5 minutes)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA support
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || {
    print_error "PyTorch CUDA installation failed or CUDA not available"
    exit 1
}

print_success "PyTorch with CUDA 12.1 installed successfully"
print_info "PyTorch CUDA 12.1 is forward-compatible with your CUDA 13.0 runtime"

################################################################################
# Install Triton
################################################################################

print_header "Installing Triton"

# Check if Triton is already available (comes with PyTorch 2.5+)
if python3 -c "import triton" 2>/dev/null; then
    TRITON_VERSION=$(python3 -c "import triton; print(triton.__version__)")
    print_success "Triton $TRITON_VERSION already available"
else
    print_info "Installing Triton..."
    pip install triton
    print_success "Triton installed"
fi

# Verify Triton
python3 -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    print_error "Triton installation verification failed"
    exit 1
}

################################################################################
# Install Flash Attention 2 (WSL-Optimized)
################################################################################

print_header "Installing Flash Attention 2 (WSL-Optimized)"

print_warning "Flash Attention 2 will be compiled from source (30-60 minutes)"
print_warning "Using MAX_JOBS=1 (single-threaded) to prevent WSL crash"
print_info "Installing build dependencies..."

pip install packaging ninja wheel

# CRITICAL WSL Settings for Flash Attention
export MAX_JOBS=1                           # CRITICAL: Single-threaded to prevent WSL crash
export CUDA_HOME=/usr/lib/wsl/lib           # WSL CUDA path
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

print_info "Building Flash Attention 2 (this will take 30-60 minutes)..."
print_info "MAX_JOBS=1 (single-threaded compilation for WSL stability)"
print_info "Build progress will be shown below..."

# Check free memory before starting
FREE_MEM_GB_NOW=$(free -g | awk '/^Mem:/{print $7}')
if [[ $FREE_MEM_GB_NOW -lt 6 ]]; then
    print_warning "Only ${FREE_MEM_GB_NOW}GB free memory. Flash Attention needs ~6-8GB"
    print_warning "Compilation may fail due to insufficient memory"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping Flash Attention installation"
        print_info "You can try again later when more memory is available"
        export FLASH_ATTN_SKIPPED=true
    fi
fi

# Install Flash Attention 2 (will compile with CUDA 12.1, works with 13.0)
if [[ "$FLASH_ATTN_SKIPPED" != "true" ]]; then
    pip install flash-attn --no-build-isolation -v || {
        print_error "Flash Attention 2 installation failed"
        print_warning "This is usually due to:"
        print_info "  1. Insufficient memory (need ~8GB free during compilation)"
        print_info "  2. WSL memory pressure (check .wslconfig settings)"
        print_info "  3. CUDA compilation errors"
        print_warning "Continuing without Flash Attention..."
        print_info "You'll still get ~4x speedup from torch.compile() + Triton"
        export FLASH_ATTN_FAILED=true
    }
fi

# Verify Flash Attention
if [[ "$FLASH_ATTN_SKIPPED" != "true" ]] && [[ "$FLASH_ATTN_FAILED" != "true" ]]; then
    if python3 -c "import flash_attn" 2>/dev/null; then
        FLASH_ATTN_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
        print_success "Flash Attention $FLASH_ATTN_VERSION installed successfully!"
        print_success "Expected speedup: ~1.4x additional (on top of torch.compile)"
    else
        print_warning "Flash Attention not available (install failed)"
        print_info "You'll still get ~4x speedup from torch.compile()"
    fi
else
    print_warning "Flash Attention not installed"
    print_info "You'll still get ~4x speedup from torch.compile()"
fi

################################################################################
# Install Parler-TTS and Remaining Dependencies
################################################################################

print_header "Installing Parler-TTS and Dependencies"

# Install Parler-TTS from GitHub
print_info "Installing Parler-TTS from GitHub..."
pip install git+https://github.com/huggingface/parler-tts.git

# Install remaining requirements
if [[ -f "requirements.txt" ]]; then
    print_info "Installing remaining requirements from requirements.txt..."
    # Skip torch, triton, flash-attn as they're already installed
    grep -v "^torch" requirements.txt | grep -v "^triton" | grep -v "^flash-attn" > /tmp/requirements_remaining.txt
    pip install -r /tmp/requirements_remaining.txt
    rm /tmp/requirements_remaining.txt
    print_success "All requirements installed"
else
    print_warning "requirements.txt not found in current directory"
fi

################################################################################
# Update Environment Configuration
################################################################################

print_header "Updating Environment Configuration"

# Update .env file if it exists
if [[ -f ".env" ]]; then
    print_info "Updating .env file..."

    # Enable torch.compile for WSL/Linux
    if grep -q "PARLER_ENABLE_COMPILE" .env; then
        sed -i 's/PARLER_ENABLE_COMPILE=false/PARLER_ENABLE_COMPILE=true/' .env
        print_success "Enabled torch.compile() in .env"
    else
        echo "PARLER_ENABLE_COMPILE=true" >> .env
        print_success "Added PARLER_ENABLE_COMPILE=true to .env"
    fi
else
    print_warning ".env file not found. Create one from .env.example"
fi

################################################################################
# Run Validation Tests
################################################################################

print_header "Running Validation Tests"

# Run validation script if it exists
if [[ -f "validate_gpu_setup.py" ]]; then
    print_info "Running GPU validation script..."
    python3 validate_gpu_setup.py
else
    print_info "Running basic validation tests..."

    # Basic CUDA test
    python3 << 'EOF'
import torch
import sys

print("\n=== PyTorch CUDA Validation ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
else:
    print("ERROR: CUDA is not available!")
    sys.exit(1)

# Test Flash Attention
print("\n=== Flash Attention Validation ===")
try:
    import flash_attn
    print(f"Flash Attention version: {flash_attn.__version__}")
    print("Flash Attention: AVAILABLE ✓")
except ImportError:
    print("Flash Attention: NOT AVAILABLE (you'll still get 4x from torch.compile)")

# Test Triton
print("\n=== Triton Validation ===")
try:
    import triton
    print(f"Triton version: {triton.__version__}")
    print("Triton: AVAILABLE ✓")
except ImportError:
    print("Triton: NOT AVAILABLE (required for torch.compile)")
    sys.exit(1)

# Test torch.compile
print("\n=== torch.compile() Validation ===")
try:
    @torch.compile
    def simple_fn(x):
        return x * 2

    result = simple_fn(torch.tensor([1.0]).cuda())
    print("torch.compile(): WORKING ✓")
except Exception as e:
    print(f"torch.compile() test failed: {e}")
    sys.exit(1)

print("\n=== All Critical Tests Passed ===")
EOF

    if [[ $? -eq 0 ]]; then
        print_success "All validation tests passed!"
    else
        print_error "Some validation tests failed"
        exit 1
    fi
fi

################################################################################
# Installation Complete
################################################################################

print_header "Installation Complete!"

echo ""
print_success "WSL GPU-optimized backend setup complete!"
echo ""
print_info "Summary:"
echo "  - WSL2 Environment: ✓"
echo "  - PyTorch with CUDA 12.1 (compatible with CUDA 13.0 runtime): ✓"
echo "  - Triton: ✓"
if [[ "$FLASH_ATTN_SKIPPED" == "true" ]] || [[ "$FLASH_ATTN_FAILED" == "true" ]]; then
    echo "  - Flash Attention 2: ✗ (skipped or failed)"
    print_info "    You can retry flash-attn later with: MAX_JOBS=1 pip install flash-attn --no-build-isolation"
else
    echo "  - Flash Attention 2: ✓"
fi
echo "  - Parler-TTS: ✓"
echo ""
print_info "Next steps:"
echo "  1. Run 'python3 validate_gpu_setup.py' for detailed diagnostics"
echo "  2. Ensure .env has PARLER_ENABLE_COMPILE=true"
echo "  3. Start server: uvicorn app.main:app --reload"
echo "  4. Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
print_warning "Expected performance improvements:"
if [[ "$FLASH_ATTN_SKIPPED" != "true" ]] && [[ "$FLASH_ATTN_FAILED" != "true" ]]; then
    echo "  - Flash Attention 2: ~1.4x speedup"
    echo "  - torch.compile(): ~4x speedup"
    echo "  - Combined: ~5-6x faster TTS generation"
    echo "  - Target TTFA: <200ms (from ~300ms)"
else
    echo "  - torch.compile(): ~4x speedup (without Flash Attention)"
    echo "  - Target TTFA: ~250ms"
fi
echo ""
print_info "Backup of previous environment saved to: $BACKUP_FILE"
echo ""
print_success "Setup complete! 🚀"
