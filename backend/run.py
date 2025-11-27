"""
Quick start script for running the backend server.
Handles basic environment checks before starting.
"""

import os
import sys
import signal
import time
import argparse
from pathlib import Path


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    if not shutdown_requested:
        shutdown_requested = True
        print("\n\n" + "=" * 60)
        print("  Shutdown signal received. Stopping server...")
        print("=" * 60)
        sys.exit(0)


def check_env_file():
    """Check if .env file exists and has required variables."""
    env_path = Path(__file__).parent / ".env"

    if not env_path.exists():
        print("ERROR: .env file not found!")
        print("\nPlease create a .env file with your API keys:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Gemini API key to the .env file")
        print("\nExample .env content:")
        print("  GEMINI_API_KEY=your_gemini_api_key_here")
        return False

    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv(env_path)

    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key or gemini_key in ("your_gemini_api_key_here", "your_google_gemini_api_key_here"):
        print("ERROR: GEMINI_API_KEY not properly configured in .env file!")
        print("\nPlease set your Gemini API key in .env file:")
        print("  - Get API key: https://aistudio.google.com/app/apikey")
        return False

    print("✓ Environment configuration looks good")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "langgraph",
        "langchain_google_genai",
        "requests",
        "PyPDF2",
        "docx"
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False

    print("✓ All dependencies are installed")
    return True


def main():
    """Main entry point."""
    # No command-line arguments needed - V3 is always used
    parser = argparse.ArgumentParser(
        description="Screening Interview Chatbot - Backend Server (V3 Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The backend now exclusively uses the V3 Hybrid Modular State Machine architecture with:
  - Conversational turn handling (clarifications, thinking, acknowledgments)
  - Strategic time allocation (priority-based, no hard limits)
  - Quality-driven follow-ups (unlimited, based on coverage + confidence)
  - Intelligent topic selection (multi-factor scoring algorithm)
        """
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Screening Interview Chatbot - Backend Server")
    print("=" * 60)
    print()

    # Run checks
    if not check_dependencies():
        sys.exit(1)

    if not check_env_file():
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # V3 is always used (Hybrid Modular State Machine Architecture)
    print("✓ V3 Hybrid Modular State Machine Architecture")
    print("  🎯 Advanced conversational interview system")
    print("  - ✓ Conversational turn handling (clarifications, thinking, acknowledgments)")
    print("  - ✓ Strategic time allocation (priority-based, no hard limits)")
    print("  - ✓ Quality-driven follow-ups (unlimited, based on coverage + confidence)")
    print("  - ✓ Intelligent topic selection (multi-factor scoring)")
    print()

    # Check for development mode reload option
    enable_reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    print()
    print("Starting server...")
    print()
    port = int(os.getenv("PORT", "8000"))
    print("API will be available at:")
    print(f"  - http://localhost:{port}")
    print(f"  - API docs: http://localhost:{port}/docs")
    print(f"  - ReDoc: http://localhost:{port}/redoc")
    print()
    if enable_reload:
        print("NOTE: Auto-reload is ENABLED (UVICORN_RELOAD=true)")
        print("      This may cause issues on Windows. Use CTRL+C to stop.")
    else:
        print("NOTE: Auto-reload is DISABLED for stability on Windows")
        print("      Set UVICORN_RELOAD=true to enable auto-reload")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 60)
    print()

    # Start the server
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=enable_reload,
        log_level="info",
        timeout_graceful_shutdown=5
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("  Server stopped by user (CTRL+C)")
        print("=" * 60)
        # Give a moment for cleanup
        time.sleep(0.5)
        sys.exit(0)
    except SystemExit:
        # Allow clean exits from signal handler
        pass
    except Exception as e:
        print("\n\n" + "=" * 60)
        print(f"  ERROR: {str(e)}")
        print("=" * 60)
        sys.exit(1)
