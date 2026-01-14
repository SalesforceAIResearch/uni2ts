"""
A Flask server that hosts time-series forecasting models.
"""

import argparse
import json
import logging
from datetime import datetime

from flask import Flask, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

with open("src/tools/hosts.json", "r") as f:
    HOST_CONFIG = json.load(f)


def main():
    """
    usage example:
        import requests
        url = f"http://{host_addr}:{host_port}/{app_name}"
        data = {"history_values": list(range(200)), "pred_length": 5}
        response = requests.post(url, json=data)
        print(response.json())
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="chronos_v2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.model_name == "chronos_v2":
        from .tsf_models import get_chronos_forecast_fn as get_forecast_fn
    elif args.model_name == "timesfm":
        from .tsf_models import get_timesfm_forecast_fn as get_forecast_fn
    elif args.model_name == "tirex":
        from .tsf_models import get_tirex_forecast_fn as get_forecast_fn
    elif args.model_name == "moirai":
        from .tsf_models import get_moirai_forecast_fn as get_forecast_fn
    else:
        raise ValueError(f"model_name {args.model_name} not supported")
    forecast_fn = get_forecast_fn(device=args.device)

    assert args.model_name in HOST_CONFIG, "model_name must be in HOST_CONFIG"
    app_name = HOST_CONFIG[args.model_name]["APP_NAME"]
    host_addr = HOST_CONFIG[args.model_name]["HOST_ADDR"]
    host_port = HOST_CONFIG[args.model_name]["HOST_PORT"]
    app = Flask(__name__)

    @app.route(f"/{app_name}", methods=["POST"])
    def predict():
        data = request.json["history_values"]
        pred_len = request.json["pred_length"]
        assert isinstance(data, list) and len(data) > 0 and pred_len > 0
        call_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output = forecast_fn(data=[data], prediction_length=[pred_len])[0]
        logging.info(
            f"============== {args.model_name} called: {call_time} =============="
        )
        return output["median"]

    app.run(host=host_addr, port=host_port, threaded=False)


# generate a test case
if __name__ == "__main__":
    main()
    # test()
