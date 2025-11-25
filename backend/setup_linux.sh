#!/bin/bash

################################################################################
# Linux GPU Setup Script for xquizit Backend
# Installs CUDA 12.8, PyTorch with CUDA support, Triton, and Flash Attention 2
# For use with NVIDIA RTX 5060 Ti (Blackwell architecture)
#
# Usage: bash setup_linux.sh
# Requires: Python 3.12, sudo access, ~15GB disk space
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PYTHON_VERSION="3.12"
REQUIRED_CUDA_VERSION="12.8"
REQUIRED_DRIVER_VERSION="545"
REQUIRED_DISK_SPACE_GB=15

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

print_header "Pre-flight System Checks"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is for Linux only. Detected OS: $OSTYPE"
    exit 1
fi
print_success "Running on Linux"

# Check for sudo access
if ! sudo -v; then
    print_error "This script requires sudo access for CUDA installation"
    exit 1
fi
print_success "Sudo access confirmed"

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

# Check for NVIDIA GPU
if ! check_command nvidia-smi; then
    print_error "nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi
print_success "NVIDIA driver detected"

# Check driver version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
if [[ $DRIVER_MAJOR -lt $REQUIRED_DRIVER_VERSION ]]; then
    print_error "NVIDIA driver version $DRIVER_VERSION is too old. Need >= $REQUIRED_DRIVER_VERSION for CUDA 12.8"
    print_info "Update your driver: sudo ubuntu-drivers autoinstall"
    exit 1
fi
print_success "NVIDIA driver version: $DRIVER_VERSION"

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
    software-properties-common

print_success "System dependencies installed"

################################################################################
# Install CUDA 12.8 Toolkit
################################################################################

print_header "Installing CUDA 12.8 Toolkit"

# Check if CUDA 12.8 is already installed
if check_command nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_info "Found existing CUDA version: $CUDA_VERSION"

    if [[ "$CUDA_VERSION" == "12.8" ]]; then
        print_success "CUDA 12.8 already installed, skipping installation"
    else
        print_warning "Found CUDA $CUDA_VERSION, but need 12.8"
        print_info "Installing CUDA 12.8 alongside existing version..."
    fi
else
    print_info "CUDA not found, installing CUDA 12.8..."
fi

# Detect Ubuntu/Debian version
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS_NAME=$ID
    OS_VERSION=$VERSION_ID
    print_info "Detected: $OS_NAME $OS_VERSION"
fi

# Install CUDA 12.8 repository
if [[ ! -f /etc/apt/sources.list.d/cuda*.list ]]; then
    print_info "Adding NVIDIA CUDA repository..."

    # For Ubuntu 22.04/24.04
    if [[ "$OS_NAME" == "ubuntu" ]]; then
        if [[ "$OS_VERSION" == "22.04" ]] || [[ "$OS_VERSION" == "24.04" ]]; then
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            sudo dpkg -i cuda-keyring_1.1-1_all.deb
            rm cuda-keyring_1.1-1_all.deb
            sudo apt-get update -qq
        fi
    fi
fi

# Install CUDA 12.8
if ! dpkg -l | grep -q "cuda-toolkit-12-8"; then
    print_info "Installing CUDA 12.8 toolkit (this may take 10-15 minutes)..."
    sudo apt-get install -y -qq cuda-toolkit-12-8
    print_success "CUDA 12.8 toolkit installed"
else
    print_success "CUDA 12.8 toolkit already installed"
fi

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
if [[ -f "$CUDA_HOME/bin/nvcc" ]]; then
    NVCC_VERSION=$($CUDA_HOME/bin/nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_success "CUDA $NVCC_VERSION installed at: $CUDA_HOME"
else
    print_error "CUDA installation failed. nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

################################################################################
# Install PyTorch with CUDA 12.8
################################################################################

print_header "Installing PyTorch with CUDA 12.8 Support"

# Uninstall existing PyTorch (may be CPU-only)
print_info "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 12.8
print_info "Installing PyTorch 2.5+ with CUDA 12.8 (this may take 3-5 minutes)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA support
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" || {
    print_error "PyTorch CUDA installation failed or CUDA not available"
    exit 1
}

print_success "PyTorch with CUDA 12.8 installed successfully"

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
# Install Flash Attention 2
################################################################################

print_header "Installing Flash Attention 2"

print_warning "Flash Attention 2 will be compiled from source (10-20 minutes)"
print_info "Installing build dependencies..."

pip install packaging ninja wheel

# Set compilation flags for CUDA 12.8
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM during compilation
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

print_info "Building Flash Attention 2 (this will take ~15 minutes)..."
print_info "Build progress will be shown below..."

# Install Flash Attention 2 (will compile with CUDA 12.8)
pip install flash-attn --no-build-isolation || {
    print_error "Flash Attention 2 installation failed"
    print_info "This is usually due to compilation errors. Check if you have enough RAM (need ~8GB)"
    print_info "You can continue without Flash Attention, but performance will be reduced"
    print_warning "Continuing without Flash Attention..."
}

# Verify Flash Attention
if python3 -c "import flash_attn" 2>/dev/null; then
    FLASH_ATTN_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
    print_success "Flash Attention $FLASH_ATTN_VERSION installed successfully"
else
    print_warning "Flash Attention not available (install failed or skipped)"
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
    pip install -r requirements.txt
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

    # Enable torch.compile for Linux
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

# Add CUDA paths to bashrc (optional, for persistence)
print_info "To persist CUDA environment variables, add to ~/.bashrc:"
echo -e "${YELLOW}"
echo "export CUDA_HOME=/usr/local/cuda-12.8"
echo "export PATH=\$CUDA_HOME/bin:\$PATH"
echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo -e "${NC}"

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
    print("Flash Attention: NOT AVAILABLE (optional)")

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

print("\n=== All Tests Passed ===")
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
print_success "GPU-optimized backend setup complete!"
echo ""
print_info "Summary:"
echo "  - CUDA 12.8 installed at: $CUDA_HOME"
echo "  - PyTorch with CUDA 12.8: ✓"
echo "  - Triton: ✓"
echo "  - Flash Attention 2: $(python3 -c 'import flash_attn; print("✓")' 2>/dev/null || echo '✗ (optional)')"
echo "  - Parler-TTS: ✓"
echo ""
print_info "Next steps:"
echo "  1. Run 'python3 validate_gpu_setup.py' for detailed diagnostics"
echo "  2. Ensure .env has PARLER_ENABLE_COMPILE=true"
echo "  3. Start server: uvicorn app.main:app --reload"
echo "  4. Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
print_warning "Expected performance improvements:"
echo "  - Flash Attention 2: ~1.4x speedup"
echo "  - torch.compile(): ~4x speedup"
echo "  - Combined: ~5-6x faster TTS generation"
echo "  - Target TTFA: <200ms (from ~300ms)"
echo ""
print_info "Backup of previous environment saved to: $BACKUP_FILE"
echo ""
print_success "Setup complete! 🚀"
