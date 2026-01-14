import json
import logging

import requests

logging.basicConfig(level=logging.INFO)

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

mcp = FastMCP("time_series_forecasting")

with open("src/tools/hosts.json", "r") as f:
    HOST_CONFIG = json.load(f)


@mcp.tool()
async def chronos_forecast_service(
    history_values: list[int | float], pred_length: int
) -> str:
    """
    Chronos is an advanced time-series forecasting model. It makes predictions by discovering the causal relationships between history patterns and future values. It works only when the history values are informative enough.
    Args:
        history_values: a list of observed numerical values.
        pred_length: the number of values to forecast.
    Returns:
        a list of forecast values.
    """
    try:
        app_name = HOST_CONFIG["chronos_v2"]["APP_NAME"]
        host_addr = HOST_CONFIG["chronos_v2"]["HOST_ADDR"]
        host_port = HOST_CONFIG["chronos_v2"]["HOST_PORT"]
        response = requests.post(
            f"http://{host_addr}:{host_port}/{app_name}",
            json={"history_values": history_values, "pred_length": pred_length},
        )
        forecast_median = response.json()
        forecast_median = [round(x, 4) for x in forecast_median]
        return json.dumps({"forecast": forecast_median})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def moirai_forecast_service(
    history_values: list[int | float], pred_length: int
) -> str:
    """
    Moirai is an advanced time-series forecasting model. It makes predictions by discovering the causal relationships between history patterns and future values. It works only when the history values are informative enough.
    Args:
        history_values: a list of observed numerical values.
        pred_length: the number of values to forecast.
    Returns:
        a list of forecast values.
    """
    try:
        app_name = HOST_CONFIG["moirai"]["APP_NAME"]
        host_addr = HOST_CONFIG["moirai"]["HOST_ADDR"]
        host_port = HOST_CONFIG["moirai"]["HOST_PORT"]
        response = requests.post(
            f"http://{host_addr}:{host_port}/{app_name}",
            json={"history_values": history_values, "pred_length": pred_length},
        )
        forecast_median = response.json()
        forecast_median = [round(x, 4) for x in forecast_median]
        return json.dumps({"forecast": forecast_median})
    except:
        return "Error: tool execution failed. History must be a list of numbers and pred_length must be a positive integer."


@mcp.tool()
async def python_sandbox_service(code: str) -> str:
    """
    Python Sandbox is a tool that allows you to execute Python code in a sandboxed environment.
    Args:
        code: the Python code to execute.
    Returns:
        the output of the code execution.
    """
    try:
        url = HOST_CONFIG["python_sandbox"]
        payload = {"code": code, "timeout": 300}
        response = requests.post(f"{url}/execute", json=payload, timeout=320)
        return json.dumps(
            {"success": response.json()["success"], "output": response.json()["output"]}
        )
    except:
        return "Error: tool execution failed. Code must be a valid Python code."


if __name__ == "__main__":
    mcp.run(transport="stdio")
