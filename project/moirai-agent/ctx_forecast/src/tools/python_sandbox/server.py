#!/usr/bin/env python3
"""
HTTP server that runs inside the Docker container to execute Python code.
This server creates temporary directories for each execution and cleans them up.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for temporary execution directories
BASE_TEMP_DIR = os.environ.get("SANDBOX_TEMP_DIR", "/tmp/sandbox_executions")

# Thread pool executor for running subprocess operations
_executor = ThreadPoolExecutor(max_workers=10)


def _run_subprocess_sync(code_file: str, temp_dir: str, timeout: int) -> Dict[str, Any]:
    """
    Synchronous helper function to run subprocess.
    This runs in a thread pool to avoid blocking the event loop.
    """
    try:
        process = subprocess.Popen(
            ["python", "-u", "code.py"],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        try:
            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode

            # Extract return value from stderr if present
            return_value = None
            stderr_output = stderr or ""

            if stderr:
                # Look for return value marker
                if "=== RETURN VALUE ===" in stderr:
                    try:
                        lines = stderr.split("\n")
                        marker_idx = None
                        for i, line in enumerate(lines):
                            if "=== RETURN VALUE ===" in line:
                                marker_idx = i
                                break
                        if marker_idx is not None and marker_idx + 1 < len(lines):
                            return_value_str = lines[marker_idx + 1].strip()
                            if return_value_str:
                                try:
                                    return_value = json.loads(return_value_str)
                                except:
                                    return_value = return_value_str
                        # Remove return value marker and value from stderr output
                        if marker_idx is not None:
                            stderr_lines = [
                                line
                                for i, line in enumerate(lines)
                                if i != marker_idx and i != marker_idx + 1
                            ]
                            stderr_output = "\n".join(stderr_lines).strip()
                    except:
                        pass

            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "output": stdout or "",
                "stderr": stderr_output,
                "return_value": return_value,
                "error": None if exit_code == 0 else (stderr_output or stdout),
            }
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "stderr": "",
                "return_value": None,
                "error": f"Execution timeout after {timeout} seconds",
            }
    except Exception as e:
        logger.error(f"Error in subprocess execution: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "output": "",
            "stderr": "",
            "return_value": None,
            "error": f"Execution error: {str(e)}",
        }


async def execute_python_code(code: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute Python code in a temporary directory.
    Each execution runs in its own process and thread, allowing concurrent requests.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with success, exit_code, output, and error fields
    """
    temp_dir = None
    try:
        # Create a unique temporary directory for this execution
        os.makedirs(BASE_TEMP_DIR, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="exec_", dir=BASE_TEMP_DIR)
        logger.info(f"Created temporary directory: {temp_dir}")

        # Write code to a file in the temp directory
        code_file = os.path.join(temp_dir, "code.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Run subprocess in thread pool to avoid blocking the event loop
        # This allows multiple requests to be processed concurrently
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _run_subprocess_sync, code_file, temp_dir, timeout
        )

        return result

    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return {
            "success": False,
            "exit_code": -1,
            "output": "",
            "stderr": "",
            "return_value": None,
            "error": f"Execution error: {str(e)}",
        }
    finally:
        # Always clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                # Run cleanup in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_executor, shutil.rmtree, temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")


async def handle_execute(request: web.Request) -> web.Response:
    """Handle code execution requests"""
    try:
        data = await request.json()
        code = data.get("code", "")
        timeout = data.get("timeout", 300)

        if not code.strip():
            return web.json_response(
                {
                    "success": False,
                    "exit_code": -1,
                    "output": "",
                    "stderr": "",
                    "return_value": None,
                    "error": "No code provided",
                },
                status=400,
            )

        result = await execute_python_code(code, timeout)
        return web.json_response(result)

    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return web.json_response(
            {
                "success": False,
                "exit_code": -1,
                "output": "",
                "stderr": "",
                "return_value": None,
                "error": f"Server error: {str(e)}",
            },
            status=500,
        )


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint"""
    return web.json_response({"status": "ok", "python_version": sys.version})


def create_app():
    """Create the aiohttp application"""
    app = web.Application()
    app.router.add_post("/execute", handle_execute)
    app.router.add_get("/health", handle_health)
    return app


async def main():
    """Main entry point"""
    port = int(os.environ.get("SANDBOX_PORT", "8080"))
    logger.info(f"Starting sandbox HTTP server on port {port}")

    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info(f"Sandbox server ready on port {port}")

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await runner.cleanup()
        _executor.shutdown(wait=True)
        logger.info("Thread pool executor shut down")


if __name__ == "__main__":
    asyncio.run(main())
