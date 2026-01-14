import importlib.util


def import_config(config_path, attr_name="CONFIG"):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = getattr(config_module, attr_name)
    return config
