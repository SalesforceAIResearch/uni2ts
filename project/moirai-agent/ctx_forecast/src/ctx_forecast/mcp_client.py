import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from src.ctx_forecast.utils import encode_image, parse_values_from_string


class MCPClient:
    def __init__(self, config):
        # self.openai = OpenAI()
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.system_prompt = config["system_prompt"]
        self.model_name = config["llm"]["model_name"]
        self.model_params = config["llm"]["model_params_type"]

        self.exit_stack = AsyncExitStack()
        self.servers: Dict[str, Dict[str, Any]] = (
            {}
        )  # server_name -> {session, stdio, write, tools}
        self.server_configs = config["servers"]
        self.tool_to_server_map: Dict[str, str] = {}  # tool_name -> server_name

        self.max_steps = config.get("max_steps", 5)

    async def connect_to_servers(self):
        print("🔄 Connecting to MCP servers...")
        for server_name, server_config in self.server_configs.items():
            try:
                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config["args"],
                    env=server_config.get("env", None),
                )
                stdio, write = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                await session.initialize()

                response = await session.list_tools()
                self.servers[server_name] = {
                    "session": session,
                    "tools": response.tools,
                    "config": server_config,
                }
                for tool in response.tools:
                    assert (
                        tool.name not in self.tool_to_server_map
                    ), f"Conflict: Tool {tool.name} already exists in the tool_to_server_map"
                    self.tool_to_server_map[tool.name] = server_name

                print(
                    f"✅ Connected to {server_name} with tools: {[tool.name for tool in response.tools]}"
                )
            except Exception as e:
                print(f"❌ Failed to connect to {server_name} server: {e}")
                raise e
        all_tools = await self.get_all_tools()
        print(
            f"🧰 Total {len(all_tools)} tools ENABLED: {[tool['name'] for tool in all_tools]}"
        )

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get enabled tools from all connected servers"""
        all_tool_dicts = []
        all_tool_names = []
        for server_name, server_info in self.servers.items():
            tools_enabled = self.server_configs[server_name].get(
                "enabled_tools", [tool.name for tool in server_info["tools"]]
            )
            all_tool_names.extend(tools_enabled)
            for tool in server_info["tools"]:
                if tool.name in tools_enabled:
                    tool_dict = {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                    all_tool_dicts.append(tool_dict)
        return all_tool_dicts

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        target_server = self.tool_to_server_map[tool_name]
        result = await self.servers[target_server]["session"].call_tool(
            tool_name, arguments
        )
        return result

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def stop_criteria(self, reply_content: str, count: int) -> bool:
        """Stop criteria for the LLM interactions."""
        if reply_content is not None:
            parsed_values = parse_values_from_string(
                reply_content, start_tag="\\boxed{", end_tag="}", entry_sep=","
            )
        else:
            parsed_values = []
        if count >= self.max_steps or len(parsed_values) > 0:
            flag = True
        else:
            flag = False
        return flag, parsed_values

    async def _single_round_query(
        self,
        context: list = [],
        query: str = None,
        image_path: str = None,
        step: int = 0,
    ) -> tuple:
        # prepare input
        assert query or len(context) > 0, "Either query or context must be provided"
        available_tools = await self.get_all_tools()
        instructions = self.system_prompt
        context = context + [{"role": "user", "content": f"step: {step}"}]

        if query:
            if image_path and os.path.exists(image_path):
                base64_image = encode_image(image_path)
                context = context + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                            {"type": "input_text", "text": query},
                        ],
                    },
                ]
            else:
                context = context + [{"role": "user", "content": query}]

        # call the LLM
        _logs = []
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                reply = self.openai.responses.create(
                    model=self.model_name,
                    instructions=instructions,
                    input=context,
                    tools=available_tools,
                    tool_choice="auto",
                    **self.model_params,
                )
                break  # success
            except Exception as e:
                _logs.append(f"<<< step:{step} \n[LLM Error][Attempt {attempt+1}]: {e}")
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(1)

        # handle reply
        context += reply.output
        _logs.extend(
            [
                f"<<< step:{step} \n Reasoning: {rsp.summary}.\n"
                for rsp in reply.output
                if rsp.type == "reasoning"
            ]
        )
        _logs.append(f"<<< step:{step} \n[Output]: {reply.output_text}")

        # handle tool calls
        tool_calls = [rsp for rsp in reply.output if rsp.type == "function_call"]

        for tool_call in tool_calls:
            call_id = tool_call.call_id
            tool_name = tool_call.name

            try:
                tool_args = json.loads(tool_call.arguments)
                tool_result = await self.call_tool(tool_name, tool_args)

                if tool_result.content[0].type == "text":
                    tool_result_text = tool_result.content[0].text.strip()
                    context.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": tool_result_text,
                        }
                    )

                elif tool_result.content[0].type == "image":
                    tool_result_text = tool_result.content[
                        1
                    ].text  #!!!: mcp-server may return (image, save_path)
                    context.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": tool_result_text,
                        }
                    )
                    base64_image = tool_result.content[0].data
                    context.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                },
                                {
                                    "type": "input_text",
                                    "text": f"Image returned from the tool call: {call_id}",
                                },
                            ],
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported content type: {tool_result.content[0].type}"
                    )

                _logs.append(
                    f"<<< step:{step} \n[Calling]: {tool_name} --- [Args]: {tool_args}"
                )
                _logs.append(
                    f"<<< step:{step} \n[Calling]: {tool_name} --- [Return]: {tool_result_text}"
                )

            except Exception as e:
                context.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": f"Error: tool execution failed --- {e}",
                    }
                )
                _logs.append(f"<<< step:{step} \n[Error]: {tool_name} | {e}.")

        return context, _logs, reply.output_text

    async def make_forecast(
        self, history_info: str, user_instruct: str, image_path: str = None
    ) -> tuple:
        all_logs = []
        step = 0
        context = []
        query = history_info + "\n" + user_instruct
        try:
            while True:
                step += 1
                context, _logs, reply_text = await self._single_round_query(
                    context, query, image_path, step
                )
                query, image_path = (
                    None,
                    None,
                )  # query and image_path are encoded into context, no need to input in next round
                all_logs.extend(_logs)
                is_stop, parsed_values = self.stop_criteria(reply_text, step)
                if is_stop:
                    break
            return parsed_values, all_logs, context, self.system_prompt
        except Exception as e:
            return [], [], [], self.system_prompt


async def main():
    max_iterations = 5
    CONFIG = {
        "llm": {
            "model_name": "gpt-5-mini",
            "model_params_type": {
                "temperature": 1.0,
                "top_p": 1.0,
                "reasoning": {"effort": "medium", "summary": "auto"},
                "max_output_tokens": 8192,
            },
        },
        "servers": {
            "forecast": {
                "command": "python",
                "args": ["-m", "src.ctx_forecast.mcp_server"],
                "enabled_tools": ["chronos_forecast_service", "python_sandbox_service"],
            },
        },
        "max_iterations": max_iterations,
        "system_prompt": (
            "You are an intelligent assistant that can solve complex problems by thinking step-by-step and using available tools when needed. "
            "\n## Your Task "
            "\n- Solve a contextual time-series forecasting problem, where historical values and contextual information are provided. "
            "\n- The future values depends on both of the observed values and the contextual information. "
            "\n- Contextual information may specify or imply unexpected effects in the future that will affect the normal results reasoned from history values. Then, you should combine the forecasting tool's results with the contextual information to reason the future."
            "\n- Contextual information may specify or imply occasional or abnormal factors in the history that will not persist in the future. Then, identify and eliminate the misleading factors from the history, the future should be reasonsed from modified history values and relevant knowledge. "
            "\n- Contextual information may totally dominate the future, when the history values are less informative. Then, discover the underlying correlation or math structure defined by context and history values, use it for prediction making. "
            "\n- You should figure out the real cause of the future first, then make predictions accordingly. "
            "\n## How You Work "
            "\n 1. **Think First**: Analyze the problem and determine what information or actions you need "
            "\n 2. **Use Tools When Needed**: Call appropriate functions/tools to perform numerical modeling. "
            "\n 3. **Reason with Results**: Process the tool outputs and use them to inform your next steps "
            "\n 4. **Iterate**: Continue thinking and using tools until you can provide a complete answer "
            "\n ## Important Guidelines "
            "\n - Only when the original or modified history values show clear patterns, you may call forecasting tools. Otherwise, fatal numerical errors will happen. "
            "\n - Only when exact mathematical structures are inferred from the context, you may write and execute codes in the python-sandbox. Always include a print function in your codes to return valid messages. "
            f"\n - Ensure that all reasoning and tool usage is complete within a maximum of {max_iterations} steps. Each step should either advance your understanding or gather necessary information. Be systematic and thorough in your approach. A final and fully cited answer has to be output before step {max_iterations}. "
        ),
    }

    TEST_SAMPLE = {
        "history_values": "2849.5772644674,2912.3948947443,2995.999075764,2936.6241219303,3016.1410780868,2869.9795725569,2986.7066804195,3066.8884560843,2782.6679560818,2693.2375614632,2661.9296549431,1967.4012959935,1515.6707842133,1349.1352392674,1273.4706324428,1229.6741183469,1238.6434462393,1423.6177899056,1603.492107923,1823.6914640506,2479.3371368669,2681.2494683625,2877.9508821405,2907.9706603954,3070.9617595694,3190.3460817324,3100.3907701756,3086.9856865205,3139.9443718781,3079.266410639,2979.6736206964,2994.9114919193,2704.3883634122,2577.9240432792,2576.8941548079,1938.4659952122,1503.0460097508,1378.7377880074,1372.5784429005,1294.9910027543,1227.9052974185,1363.7415138545,1703.7778809029,1980.2221755895,2510.1015495852,2674.5978757034,2818.3835788392,3181.3578493884,3320.231758244,3301.0026190218,3228.627295888,3205.0224356808,3340.5956911342,3227.3649228298,3249.5330617126,3260.9762430129,2843.4419306636,2675.3670650966,2508.6797067325,1764.3496894012,1546.9605586539,1411.4180887069,1442.4173749375,1409.2828681625,1550.1255060836,1758.0326086225,2022.1134920486,2698.6547061093,2737.1199246815,2910.4430126273,2978.3682394595,3308.3525313871,3291.7884671166,3187.7517219621,3204.9236897815,3325.544296508,3324.2469721136,3251.3005642183,3193.9688433034,3264.3704816106,2895.4935364679,2791.1018989089,2800.5312059293,2151.5665842191,1728.6765955256,1627.0487085277,1545.3184469601,1545.6999761814,1490.8337485067,1614.9024431393,1826.5591844457,1982.9076631274,2678.838898887,2925.7382380416,3152.5326522676,3242.186679976,3332.8094269936,3317.0360489927,3091.2279417315,3405.6152144636,3405.4042469335,3241.2429138586,3252.6145143279,3157.4666872174,2812.0264482085,2651.8613234121,2053.2760580319,1595.2863214544,1634.674273691,1507.7070877608,1424.3763337846,1383.3273544843,1364.8133478122,1515.1600896811,1817.7497129303,2039.1988431951,2681.0986099005,2830.4428412498,2924.3255579581,3028.6919328256,3116.139835567,3068.2273681039,3028.6146003142,3085.4604851423,3160.7928891197,3112.953502076,3134.2049673202,3095.3474592766,2666.1014374546,2720.1951626144,2678.3605791413,2061.4839399257,1712.0888833569,1697.7456982891,1645.9947715961,1597.2642423739,1628.5676845407,1748.5947744574,2043.2147270388,2348.0559704712,2791.5156149276,2932.4382430333,3158.4872499641,3251.1104525584",
        "history_start": "2012-10-03 13:00:00",
        "history_end": "2012-10-09 12:00:00",
        "history_frequency": "H",
        "context_info": "This is the electricity consumption recorded in Kilowatt (kW) in city A.Suppose that there is a heat wave in city A from 2012-10-09 18:00:00 for 3 hours in city A, leading to excessive use of air conditioning, and 5 times the usual electricity being consumed. \n ",
        "roi": "5,6,7",
        # "skills":"forecasting,natural language processing,instruction following",
        # "idx":1,
        "user_instruct": "Predict 24 future values in the range from 2012-10-09 13:00:00 to 2012-10-10 12:00:00. \nThe final result must be enclosed by '\\boxed{' and '}', and where values are separated by ','.",
        "pred_length": 24,
        "future_start": "2012-10-09 13:00:00",
        "future_end": "2012-10-10 12:00:00",
        "future_values": "3306.4741429646,2965.5678242008,2906.5024781426,3090.7526672942,3227.4065834481,16754.824097567,16714.5345840893,16508.3705522012,3078.0590968614,2925.6328331622,2712.7558533895,2382.423557348,1634.0269881106,1685.3201044048,1239.3774409873,1372.3142492601,1215.4750877617,1812.973177679,1995.3306966688,2257.0234746904,2551.7606658799,3127.782724011,2845.2415446766,3385.8030127578",
        # "task_id":"ElectricityIncreaseInPredictionTask_2"
    }

    client = MCPClient(CONFIG)
    try:
        await client.connect_to_servers()
        ###->>>>>>>> ###->>>>>>>> ###->>>>>>>>
        history_info = json.dumps(
            {
                key: TEST_SAMPLE[key]
                for key in [
                    "history_values",
                    "history_start",
                    "history_end",
                    "history_frequency",
                    "context_info",
                ]
            }
        )
        user_instruct = TEST_SAMPLE["user_instruct"]
        parsed_values, all_logs, context, system_prompt = await client.make_forecast(
            history_info, user_instruct
        )
        print(f"=" * 50)
        print(parsed_values)
        print(f"=" * 50)
        print(all_logs)
        print(f"=" * 50)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())  # Interactive chat
