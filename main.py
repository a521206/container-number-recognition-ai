#!/usr/bin/env python3
"""Entry point for Container Number Recognition AI."""

import sys
from dotenv import load_dotenv

# Load environment variables before any src imports so that VISION_ENDPOINT
# and VISION_KEY are available when config.py is first imported.
load_dotenv()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Deferred import: uvicorn and the FastAPI app are only needed in API
        # mode, so avoid paying the import cost (and the OCRClient init) when
        # running in CLI mode.
        import uvicorn
        from src.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run CLI
        from src.cli import main
        main()
