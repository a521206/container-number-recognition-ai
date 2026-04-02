#!/usr/bin/env python3
"""Entry point for Container Number Recognition AI."""

import sys
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        from src.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        from src.cli import main
        main()
