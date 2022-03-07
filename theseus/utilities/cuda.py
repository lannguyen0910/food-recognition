""" CUDA / AMP utils
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


def get_devices_info(device_names="0"):

    if device_names.startswith('cuda'):
        device_names = device_names.split('cuda:')[1]
    elif device_names.startswith('cpu'):
        return "CPU"

    devices_info = ""
    for i, device_id in enumerate(device_names.split(',')):
        p = torch.cuda.get_device_properties(i)
        # bytes to MB
        devices_info += f"CUDA:{device_id} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    return devices_info
