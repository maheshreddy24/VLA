import yaml


def load_config(path, overrides=()):
    """ Load a YAML config and apply dotted overrides, e.g. ["trainer.lr=3e-4", "data.suites=[libero_spatial]"]. """
    with open(path) as f:
        config = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = config
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return config
