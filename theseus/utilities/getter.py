from theseus.registry import Registry


def get_instance(config, registry: Registry, **kwargs):
    # ref https://github.com/vltanh/torchan/blob/master/torchan/utils/getter.py
    assert "name" in config
    config.setdefault("args", {})
    if config.get("args", None) is None:
        config["args"] = {}

    return registry.get(config['name'])(**config.get('args', {}), **kwargs)


def get_instance_recursively(config, registry: Registry, **kwargs):
    if isinstance(config, (list, tuple)):
        out = [get_instance_recursively(
            item, registry=registry, **kwargs) for item in config]
        return out
    if isinstance(config, dict):
        if 'name' in config.keys():
            if registry:
                args = get_instance_recursively(
                    config.get('args', {}), registry, **kwargs)
                if args is None:
                    return registry.get(config['name'])(**kwargs)
                if isinstance(args, list):
                    return registry.get(config['name'])(*args, **kwargs)
                elif isinstance(args, dict):
                    return registry.get(config['name'])(**args, **kwargs)
                else:
                    raise ValueError(f'Unknown type: {type(args)}')
        else:
            out = {}
            for k, v in config.items():
                out[k] = get_instance_recursively(
                    v, registry=registry, **kwargs)
            return out
        return globals()[config["name"]](**config["args"], **kwargs)
    return config


def get_function(name):
    return globals()[name]
