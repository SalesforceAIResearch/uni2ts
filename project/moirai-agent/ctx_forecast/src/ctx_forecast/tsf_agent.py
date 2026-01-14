import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


async def main():
    parser = argparse.ArgumentParser(description="VLLM generation script")
    parser.add_argument(
        "--task_file",
        type=str,
        default="/fsx/sfr/data/multimodal/hsyan/time_series_analysis/cik/tasks_by_model_capability/instruct_following/data.parquet",
    )
    parser.add_argument(
        "--input_mode", type=str, default="text+image", help="text, image, text+image"
    )
    parser.add_argument(
        "--config_file", type=str, default="config.json", help="Config file path"
    )
    parser.add_argument("--config_name", type=str, default="CONFIG", help="Config name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./_tmp_/gpt4o/text+image",
        help="Output dir path",
    )
    parser.add_argument("--num_parts", type=int, default=8, help="Number of parts")
    parser.add_argument("--part_idx", type=int, default=0, help="Part index")
    args = parser.parse_args()

    # prepare data
    test_dst = load_dataset("parquet", data_files={"x": args.task_file}, split="x")
    queries = {}
    for i in range(len(test_dst)):
        if i % args.num_parts != args.part_idx:
            continue
        if os.path.exists(test_dst[i]["image_path"]):
            img_path = test_dst[i]["image_path"]
        else:
            data_dir = os.path.dirname(args.task_file)
            img_path = os.path.join(data_dir, test_dst[i]["image_path"])
            assert os.path.exists(img_path), f"Image path {img_path} does not exist"

        queries[i] = {
            "idx": test_dst[i]["idx"],
            "system_prompt": test_dst[i].get(
                "system_prompt", "You are a helpful assistant."
            ),
            "history_values": test_dst[i]["history_values"],
            "history_start": test_dst[i]["history_start"],
            "history_end": test_dst[i]["history_end"],
            "history_frequency": test_dst[i]["frequency"],
            "context_info": test_dst[i]["context_info"],
            "image_path": img_path,
            "user_instruct": test_dst[i]["user_instruct"],
            "future_values": test_dst[i]["future_values"],
            "pred_length": test_dst[i]["pred_length"],
            "roi": test_dst[i].get("roi"),
            "skills": test_dst[i].get("skills"),
            "task_id": test_dst[i].get("task_id"),
        }

    # build agent
    from src.ctx_forecast.mcp_client import MCPClient

    if args.config_file.endswith(".json"):
        config = json.load(open(args.config_file))
    elif args.config_file.endswith(".py"):
        from src.utils.import_utils import import_config

        config = import_config(args.config_file, attr_name=args.config_name)
    else:
        raise ValueError(f"Unsupported config file type: {args.config_file}")
    client = MCPClient(config)

    try:
        await client.connect_to_servers()
        for k, query_data in queries.items():
            print(
                f"Procssing query {k}, time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            use_image = "image" in args.input_mode
            image_path = query_data["image_path"] if use_image else None
            history_info = {
                "history_values": query_data["history_values"],
                "history_start": query_data["history_start"],
                "history_end": query_data["history_end"],
                "history_frequency": query_data["history_frequency"],
                "context_info": query_data["context_info"],
            }
            history_info = json.dumps(history_info)
            _, response, _, system_prompt = await client.make_forecast(
                history_info=history_info,
                user_instruct=query_data["user_instruct"],
                image_path=image_path,
            )
            print(response, "*" * 100)
            queries[k].update(
                {
                    "use_image": use_image,
                    "all_response": response,
                    "response": response[-1],
                    "system_prompt": system_prompt,
                }
            )

        save_path = os.path.join(args.output_dir, args.input_mode, "results")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, f"{args.part_idx}-{args.num_parts}.json")

        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved to {output_file}")

    finally:
        # Ensure proper cleanup of MCP client resources
        await client.cleanup()


if __name__ == "__main__":
    import time

    start_time = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"========= Time taken: {end_time - start_time} seconds ==========")
