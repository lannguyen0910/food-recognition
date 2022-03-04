from typing import List

class BaseCollator(object):
    """Base collator function
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, batch):
        return batch

class ChainCollatorWrapper(BaseCollator):
    """Wrapper for list of collate functions
    """
    def __init__(self, pre_collate_fns: List, **kwargs):
        self.pre_collate_fns = pre_collate_fns

    def __call__(self, batch):
        for fn in self.pre_collate_fns:
            batch = fn(batch)
        return batch