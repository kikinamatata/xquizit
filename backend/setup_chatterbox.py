"""
Chatterbox TTS Setup Script for Windows (Python 3.12+)
Installs chatterbox-tts with incompatible dependencies fixed.
"""

import subprocess
import sys
import os
import re
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        return False
    return True


def fix_pyproject_toml(pyproject_path):
    """Fix incompatible dependencies in pyproject.toml for Python 3.12+."""
    print(f"\n{'='*60}")
    print("  Fixing dependencies in pyproject.toml")
    print(f"{'='*60}")

    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Remove pkuseg (not needed for English, fails to build on Windows)
    if 'pkuseg' in content:
        content = re.sub(r'["\']pkuseg[^"\']*["\'],?\s*\n?', '', content)
        changes.append("Removed pkuseg dependency")

    # 2. Remove russian-text-stresser (requires Python <3.12, not needed for English)
    if 'russian-text-stresser' in content:
        content = re.sub(r'["\']russian-text-stresser[^"\']*["\'],?\s*\n?', '', content)
        changes.append("Removed russian-text-stresser dependency (Python 3.12 incompatible)")

    # 2. Fix numpy version constraint (old version incompatible with Python 3.12)
    # Replace numpy<1.26.0 or numpy==1.26.0 with numpy>=1.26.0
    content = re.sub(
        r'["\']numpy[<>=!][^"\']*["\']',
        '"numpy>=1.26.0"',
        content
    )
    if 'numpy>=1.26.0' in content and 'numpy>=1.26.0' not in original_content:
        changes.append("Updated numpy to >=1.26.0 (Python 3.12 compatible)")

    # 3. Fix scipy version if present
    content = re.sub(
        r'["\']scipy[<>=!][^"\']*["\']',
        '"scipy>=1.12.0"',
        content
    )
    if 'scipy>=1.12.0' in content and 'scipy>=1.12.0' not in original_content:
        changes.append("Updated scipy to >=1.12.0 (Python 3.12 compatible)")

    # 4. Remove any torch version pins that might conflict
    # Let the user's existing torch installation be used
    content = re.sub(
        r'["\']torch[<>=!][^"\']*["\'],?\s*\n?',
        '',
        content
    )
    if 'torch==' in original_content or 'torch<' in original_content:
        changes.append("Removed torch version pin (uses existing installation)")

    content = re.sub(
        r'["\']torchaudio[<>=!][^"\']*["\'],?\s*\n?',
        '',
        content
    )

    # Clean up any double commas or trailing commas before ]
    content = re.sub(r',(\s*,)+', ',', content)
    content = re.sub(r',(\s*\])', r'\1', content)

    if changes:
        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  Changes made:")
        for change in changes:
            print(f"    - {change}")
    else:
        print("  No changes needed")

    return len(changes) > 0


def main():
    print("="*60)
    print("  Chatterbox TTS Setup for Windows (Python 3.12+)")
    print("="*60)
    print("\nThis script will:")
    print("  1. Clone chatterbox from GitHub")
    print("  2. Fix dependencies for Python 3.12 compatibility")
    print("  3. Install chatterbox locally")
    print()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    chatterbox_dir = script_dir / "chatterbox"

    # Check if chatterbox already exists
    if chatterbox_dir.exists():
        print(f"Found existing chatterbox directory at: {chatterbox_dir}")
        response = input("Delete and re-clone? (y/N): ").strip().lower()
        if response == 'y':
            import shutil
            print("Removing existing chatterbox directory...")
            shutil.rmtree(chatterbox_dir)
        else:
            print("Using existing directory...")

    # Clone the repo if it doesn't exist
    if not chatterbox_dir.exists():
        if not run_command(
            f'git clone https://github.com/resemble-ai/chatterbox.git "{chatterbox_dir}"',
            "Cloning chatterbox repository"
        ):
            sys.exit(1)

    # Fix pyproject.toml
    pyproject_path = chatterbox_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"ERROR: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    fix_pyproject_toml(pyproject_path)

    # Install chatterbox with --no-deps first, then let pip resolve
    os.chdir(script_dir)

    print(f"\n{'='*60}")
    print("  Installing chatterbox")
    print(f"{'='*60}")

    # Install without dependencies first to avoid conflicts
    if not run_command(
        f'pip install --no-build-isolation -e "{chatterbox_dir}"',
        "Installing chatterbox (no build isolation)"
    ):
        # Try with build isolation if that fails
        print("\nRetrying with build isolation...")
        if not run_command(
            f'pip install -e "{chatterbox_dir}"',
            "Installing chatterbox (with build isolation)"
        ):
            print("\nInstallation failed.")
            print("Try manually: pip install -e ./chatterbox --no-deps")
            sys.exit(1)

    # Verify installation
    print(f"\n{'='*60}")
    print("  Verifying installation")
    print(f"{'='*60}")

    try:
        from chatterbox.tts import ChatterboxTTS
        print("\n  SUCCESS! Chatterbox TTS is installed correctly.")
        print("\n  You can now use it with:")
        print("    from chatterbox.tts import ChatterboxTTS")
        print("    model = ChatterboxTTS.from_pretrained(device='cuda')")
        print("    audio = model.generate('Hello world!')")
    except ImportError as e:
        print(f"\n  WARNING: Import test failed: {e}")
        print("  You may need to install missing dependencies manually.")
        print("  Try: pip install librosa soundfile transformers safetensors")

    print(f"\n{'='*60}")
    print("  Setup Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
