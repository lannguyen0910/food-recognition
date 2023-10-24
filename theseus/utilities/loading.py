import yaml
import torch
from theseus.utilities.loggers.observer import LoggerObserver
LOGGER = LoggerObserver.getLogger("main")


def load_yaml(path):
    with open(path, 'rt') as f:
        return yaml.safe_load(f)


def load_state_dict(instance, state_dict, key=None, is_detection=False):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module):
        try:
            if is_detection and key is not None:
                instance.load_state_dict(state_dict[key].state_dict())
            else:
                if key is not None:
                    instance.load_state_dict(state_dict[key])
                else:
                    instance.load_state_dict(state_dict)

            LOGGER.text("Loaded Successfully!", level=LoggerObserver.INFO)
        except RuntimeError as e:
            LOGGER.text(
                f'Loaded Successfully. Ignoring {e}', level=LoggerObserver.WARN)
        return instance
    else:
        if key in state_dict.keys():
            return state_dict[key]
        else:
            LOGGER.text(
                f"Cannot load key={key} from state_dict", LoggerObserver.WARN)
