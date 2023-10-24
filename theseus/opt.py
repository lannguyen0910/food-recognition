"""
Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py
"""

import yaml
import json
import os
from theseus.utilities.loading import load_yaml

from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")


class InferenceArguments:
    """
    Arguments for Opts
    """

    def __init__(self, key: str = None, config_file: str = 'test.yaml') -> None:
        assert key is not None, \
            "Please choose a task: ['detection', 'segmentation', 'classification']."

        cfg_path = os.path.join('./configs', key, config_file)
        self.config = cfg_path


class Config(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, yaml_path):
        super(Config, self).__init__()

        config = load_yaml(yaml_path)
        super(Config, self).update(config)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def save_yaml(self, path):
        LOGGER.text(f"Saving config to {path}...", level=LoggerObserver.DEBUG)
        with open(path, 'w') as f:
            yaml.dump(
                dict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path):
        LOGGER.text(
            f"Loading config from {path}...", level=LoggerObserver.DEBUG)
        return cls(path)

    def __repr__(self) -> str:
        return str(json.dumps(dict(self), sort_keys=False, indent=4))


class Opts():
    def __init__(self, args):
        super(Opts, self).__init__()
        self.args = args

    def parse_args(self):
        assert self.args.config is not None, \
            "Please specify --config=configure_file_path."
        # self.args.opt = self._parse_opt(self.args.opt)

        config = Config(self.args.config)
        # config = self.override(config, args.opt)
        return config

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def override(self, global_config, overriden):
        """
        Merge config into global config.
        Args:
            config (dict): Config to be merged.
        Returns: global config
        """
        LOGGER.text("Overriding configuration...", LoggerObserver.DEBUG)
        for key, value in overriden.items():
            if "." not in key:
                if isinstance(value, dict) and key in global_config:
                    global_config[key].update(value)
                else:
                    if key in global_config.keys():
                        global_config[key] = value
                    else:
                        LOGGER.text(f"'{key}' not found in config",
                                    level=LoggerObserver.WARN)
            else:
                sub_keys = key.split('.')
                assert (
                    sub_keys[0] in global_config
                ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                    global_config.keys(), sub_keys[0])
                cur = global_config[sub_keys[0]]
                for idx, sub_key in enumerate(sub_keys[1:]):
                    if idx == len(sub_keys) - 2:
                        if sub_key in cur.keys():
                            cur[sub_key] = value
                        else:
                            LOGGER.text(
                                f"'{key}' not found in config", level=LoggerObserver.WARN)
                    else:
                        cur = cur[sub_key]
        return global_config
