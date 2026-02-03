#!/usr/bin/env python3
"""Demo script for gradient pathology dashboard.

Usage:
    python examples/dashboard_demo.py
    # Then navigate to http://localhost:8501
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gradient_pathology.dashboard import run_dashboard

if __name__ == "__main__":
    print("🚀 Launching Gradient Pathology Dashboard...")
    print("📊 Navigate to http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop")
    print()
    run_dashboard()
