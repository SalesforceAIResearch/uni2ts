import re
from typing import Union

TIME_SERIES_COMPONENTS = [
    "history_values",
    "history_start",
    "history_end",
    "history_frequency",
    "history_timestamps",
    "future_values",
    "future_start",
    "future_end",
    "future_frequency",
    "future_timestamps",
    "entry_sep",
    "context_info",
]

DEFAULT_SPECIAL_TAGS = {
    "start_history_values": "<history>",
    "end_history_values": "</history>",
    "start_future_values": "\\boxed{",
    "end_future_values": "}",
    "entry_sep": ",",
    "missing_value": "N/A",
    "start_context": "<context>",
    "end_context": "</context>",
}


def format_time_series_forecasting_json(
    history_values: str,
    history_start: Union[str, None] = None,
    history_end: Union[str, None] = None,
    history_frequency: Union[str, None] = None,
    context_info: Union[str, None] = None,
    pred_length: int = 0,  # if future_values is not None, pred_length is the length of future_values
    instruct_prompt: Union[str, None] = None,
    future_values: Union[str, None] = None,
    future_start: Union[str, None] = None,
    future_end: Union[str, None] = None,
    value_format: str = "relative-precision+scientific",
    special_tags: dict = DEFAULT_SPECIAL_TAGS,
    **kwargs,  # Accept and ignore any additional arguments
) -> tuple[str, Union[str, None]]:
    """
    Format the time series prompt.

    Args:
        history_values: History values as a string
        history_start: Start time of history (optional)
        history_end: End time of history (optional)
        history_frequency: Frequency of history data (optional)
        context_info: Context information (optional)
        functionality: Functionality type, default 'forecasting'
        pred_length: Prediction length (optional)
        future_values: Future values as a string (optional)
        future_start: Start time of future (optional)
        future_end: End time of future (optional)
        thinking_process: Thinking process description (optional)
        instruct_prompt: Instruction prompt (optional)
        value_format: Value format specification
        special_tags: Special tags dictionary
        **kwargs: Additional arguments that will be ignored

    Returns:
        Tuple of (query, response) strings
    """
    # get the special tags
    tag_start_history_values = special_tags["start_history_values"]
    tag_end_history_values = special_tags["end_history_values"]
    tag_start_future_values = special_tags["start_future_values"]
    tag_end_future_values = special_tags["end_future_values"]
    entry_sep = special_tags["entry_sep"]
    missing_value = special_tags["missing_value"]
    tag_start_context = special_tags["start_context"]
    tag_end_context = special_tags["end_context"]

    # set flags
    history_time_flag = (
        history_start is not None
        and history_end is not None
        and history_frequency is not None
    )
    future_time_flag = future_start is not None and future_end is not None

    # preprocess the values
    if future_values is None:
        hist_list = history_values.split(entry_sep)
        hist_list, _ = preprocess_values(
            hist_list, value_format=value_format, missing_value=missing_value
        )
        history_values = entry_sep.join(hist_list)
    else:
        hist_list = history_values.split(entry_sep)
        n_hist_v = len(hist_list)
        future_list = future_values.split(entry_sep)
        all_values = hist_list + future_list

        all_values, _ = preprocess_values(
            all_values, value_format=value_format, missing_value=missing_value
        )
        hist_list, future_list = all_values[:n_hist_v], all_values[n_hist_v:]
        history_values = entry_sep.join(hist_list)
        future_values = entry_sep.join(future_list)

    output_json = {
        "history_values": history_values,
        "history_start": history_start,
        "history_end": history_end,
        "history_frequency": history_frequency,
        "context_info": context_info,
        "future_values": future_values,
        "future_start": future_start,
        "future_end": future_end,
    }

    user_instruct = ""
    if future_values is not None:
        pred_length = len(future_values.split(entry_sep))
    assert (
        pred_length > 0
    ), "pred_length must be provided if future_values is not provided"

    if future_time_flag:
        user_instruct += f"Predict {pred_length} future values in the range from {future_start} to {future_end}. "
    else:
        user_instruct += f"Predict {pred_length} future values. "

    if instruct_prompt is not None:
        user_instruct += f"\n{instruct_prompt}"

    output_json["pred_length"] = pred_length
    output_json["user_instruct"] = user_instruct
    return output_json


def format_time_series_prompt(
    history_values: str,
    history_start: Union[str, None] = None,
    history_end: Union[str, None] = None,
    history_frequency: Union[str, None] = None,
    context_info: Union[str, None] = None,
    functionality: str = "forecasting",
    pred_length: int = 0,  # if future_values is not None, pred_length is the length of future_values
    instruct_prompt: Union[str, None] = None,
    thinking_process: Union[str, None] = None,
    future_values: Union[str, None] = None,
    future_start: Union[str, None] = None,
    future_end: Union[str, None] = None,
    value_format: str = "relative-precision+scientific",
    special_tags: dict = DEFAULT_SPECIAL_TAGS,
    **kwargs,  # Accept and ignore any additional arguments
) -> tuple[str, Union[str, None]]:
    """
    Format the time series prompt.

    Args:
        history_values: History values as a string
        history_start: Start time of history (optional)
        history_end: End time of history (optional)
        history_frequency: Frequency of history data (optional)
        context_info: Context information (optional)
        functionality: Functionality type, default 'forecasting'
        pred_length: Prediction length (optional)
        future_values: Future values as a string (optional)
        future_start: Start time of future (optional)
        future_end: End time of future (optional)
        thinking_process: Thinking process description (optional)
        instruct_prompt: Instruction prompt (optional)
        value_format: Value format specification
        special_tags: Special tags dictionary
        **kwargs: Additional arguments that will be ignored

    Returns:
        Tuple of (query, response) strings
    """
    # get the special tags
    tag_start_history_values = special_tags["start_history_values"]
    tag_end_history_values = special_tags["end_history_values"]
    tag_start_future_values = special_tags["start_future_values"]
    tag_end_future_values = special_tags["end_future_values"]
    entry_sep = special_tags["entry_sep"]
    missing_value = special_tags["missing_value"]
    tag_start_context = special_tags["start_context"]
    tag_end_context = special_tags["end_context"]

    # set flags
    history_time_flag = (
        history_start is not None
        and history_end is not None
        and history_frequency is not None
    )
    future_time_flag = future_start is not None and future_end is not None

    # preprocess the values
    if future_values is None:
        hist_list = history_values.split(entry_sep)
        hist_list, _ = preprocess_values(
            hist_list, value_format=value_format, missing_value=missing_value
        )
        history_values = entry_sep.join(hist_list)
    else:
        hist_list = history_values.split(entry_sep)
        n_hist_v = len(hist_list)
        future_list = future_values.split(entry_sep)
        all_values = hist_list + future_list

        all_values, _ = preprocess_values(
            all_values, value_format=value_format, missing_value=missing_value
        )
        hist_list, future_list = all_values[:n_hist_v], all_values[n_hist_v:]
        history_values = entry_sep.join(hist_list)
        future_values = entry_sep.join(future_list)

    # prepare prompts
    query = f"{tag_start_history_values}{history_values}{tag_end_history_values}\n"  # e.g., [2025-01-01, 20.0];[2025-01-02, 21.0];[2025-01-03, 22.0]
    if history_time_flag:
        query += f"The history values are recorded from {history_start} to {history_end} with the frequency of {history_frequency}.\n"

    # if time_series_description is not None:
    #     # query = f"{time_series_description}\n" + query
    #     query = query + f"{time_series_description}\n"

    if context_info is not None:
        query += f"{tag_start_context}{context_info}{tag_end_context}\n"  # e.g., You are given a context about the temperature of a city.

    # forecasting
    if functionality == "forecasting":
        # instruction
        if future_values is not None:
            pred_length = len(future_values.split(entry_sep))
        assert (
            pred_length > 0
        ), "pred_length must be provided if future_values is not provided"

        if future_time_flag:
            query += f"Predict {pred_length} future values in the range from {future_start} to {future_end}.\n"
        else:
            query += f"Predict {pred_length} future values.\n"

        if instruct_prompt is not None:
            query += f"{instruct_prompt}\n"

        # prepare target prompts
        if future_values is None:
            response = None
        else:
            if thinking_process is None:
                response = ""
            else:
                response = f"{thinking_process}\n"

            response += (
                f"{tag_start_future_values}{future_values}{tag_end_future_values}\n"
            )

        return query, response

    # other functionality
    else:
        if instruct_prompt is not None:
            query += f"{instruct_prompt}\n"
        return query, None


def preprocess_values(
    value_str: list[str], value_format: str, missing_value: str
) -> tuple[list[str], float]:
    """
    Preprocess the values.
    Args:
        value_str: the list of values, e.g., ["1000", "205", "30"]
        value_format: the format of the values, e.g., "relative-precision", "relative-precision+scientific"
    Returns:
        value_str: the list of values, e.g., ["1000", "205", "30"]
        max_scale: the maximum scale of the values, e.g., 1000.0
    """

    def round_to_relative_precision(
        values: list[float], max_value: float, relative_precision: int = 4
    ) -> list[float]:
        """
        Round the values to the relative precision.
        """
        ### example: max_value: 5678.123, relative_precision: 4 -> abs_precision: 0.1
        exponent = int(f"{max_value:e}".split("e")[-1])
        abs_precision = exponent - relative_precision
        values_ = [round(v, -1 * abs_precision) for v in values]
        return values_

    def simply_scientific_format(value_str: str, scientific_sep: str = "e") -> str:
        """
        Simply the scientific format.
        e.g., 1.23000e+05 -> 1.23e5
        """
        assert scientific_sep in ["e", "E"], f"Unknown scientific_sep {scientific_sep}"
        coefficient, exponent = value_str.split(scientific_sep)
        coefficient = coefficient.rstrip("0")
        return f"{coefficient}{scientific_sep}{exponent}"

    # convert value_str to value_dict
    value_dict = {}
    keys_to_proc = []
    values_to_proc = []
    for idx, x in enumerate(value_str):
        try:
            value_dict[idx] = float(x)
            values_to_proc.append(value_dict[idx])
            keys_to_proc.append(idx)
        except:
            value_dict[idx] = missing_value  # N/A

    max_scale = abs(max(values_to_proc, key=abs))

    if value_format == "relative-precision":
        values_to_proc = round_to_relative_precision(
            values_to_proc, max_scale, relative_precision=4
        )
        for k, v in zip(keys_to_proc, values_to_proc):
            value_dict[k] = str(v)
    elif value_format == "relative-precision+scientific":
        values_to_proc = round_to_relative_precision(
            values_to_proc, max_scale, relative_precision=4
        )
        for k, v in zip(keys_to_proc, values_to_proc):
            v_float = str(v)
            v_scientific = f"{v:e}"
            v_scientific = simply_scientific_format(v_scientific, scientific_sep="e")
            if len(v_scientific) < len(v_float):
                value_dict[k] = v_scientific
            else:
                value_dict[k] = v_float
    else:
        raise NotImplementedError(f"Unknown value_format {value_format}")

    for k, v in value_dict.items():
        value_str[k] = v

    return value_str, max_scale


def parse_values_from_ts_prompt(
    ts_str: str, ts_type: str = "future", special_tags: dict = DEFAULT_SPECIAL_TAGS
) -> list[float]:
    """
    Parse values from the time series prompt.
    """
    tag_start_future_values = special_tags["start_future_values"]
    tag_end_future_values = special_tags["end_future_values"]
    tag_start_history_values = special_tags["start_history_values"]
    tag_end_history_values = special_tags["end_history_values"]
    entry_sep = special_tags["entry_sep"]

    if ts_type.lower() == "future":
        start_tag = tag_start_future_values
        end_tag = tag_end_future_values
    elif ts_type.lower() == "history":
        start_tag = tag_start_history_values
        end_tag = tag_end_history_values
    else:
        raise NotImplementedError(f"Unknown ts_type {ts_type}")

    ###->>>>>>>> escape the special characters in the start_tag and end_tag
    start_tag = start_tag.replace("\\", "\\\\")
    pattern = re.compile(r"{}(.*?){}".format(start_tag, end_tag), re.DOTALL)
    results = re.findall(pattern, ts_str)

    if not results:
        return []
    else:
        results = results[-1]  # using the last one
        entries = (
            results.split(start_tag)[-1].split(end_tag)[0].strip(" \n").split(entry_sep)
        )
        values = []
        for entry in entries:
            try:
                if entry.startswith("[") and entry.endswith("]"):
                    entry = entry[1:-1].split(",")[-1].strip()
                v = float(entry)
                values.append(v)
            except:
                continue
        return values


if __name__ == "__main__":
    history_values = "1;2;3;N/A;4,xx;10"
    history_start = "2025-01-01"
    history_end = "2025-01-03"
    history_frequency = "1 day"
    future_values = "4;5;6;N/A;7,xx;103"
    future_start = "2025-01-04"
    future_end = "2025-01-06"
    future_frequency = "1 day"
    pred_length = 3
    time_series_description = "The history values are WEAATHER"
    instruct_prompt = (
        "Directly predict future values and do not output any reasoning steps."
    )
    thinking_process = None

    prompt_history, prompt_future = format_time_series_prompt(
        history_values=history_values,
        # history_start=history_start,
        # history_end=history_end,
        # history_frequency=history_frequency,
        future_values=future_values,
        # future_start=future_start,
        # future_end=future_end,
        # future_frequency=future_frequency,
        pred_length=pred_length,
        # time_series_description=time_series_description,
        instruct_prompt=instruct_prompt,
        thinking_process=thinking_process,
    )
    print(prompt_history)
    print(prompt_future)
