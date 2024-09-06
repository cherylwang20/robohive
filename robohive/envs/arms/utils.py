import os
from pathlib import Path
import time
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def read_config_from_node(root_node, parent_name, child_name, dtype=int):
    # find parent
    parent_node = root_node.find(parent_name)
    if parent_node is None:
        quit("Parent %s not found" % parent_name)

    # get child data
    child_data = parent_node.get(child_name)
    if child_data is None:
        quit("Child %s not found" % child_name)

    config_val = np.array(child_data.split(), dtype=dtype)
    return config_val


def get_config_root_node(config_file_name=None, config_file_data=None):
    # get root
    if config_file_data is None:
        with open(config_file_name) as config_file_content:
            config = ET.parse(config_file_content)
        root_node = config.getroot()
    else:
        root_node = ET.fromstring(config_file_data)

    # get root data
    root_data = root_node.get("name")
    assert isinstance(root_data, str)
    root_name = np.array(root_data.split(), dtype=str)

    return root_node, root_name



def read_config_from_xml(config_file_name, parent_name, child_name, dtype=int):
    root_node, _ = get_config_root_node(config_file_name=config_file_name)
    return read_config_from_node(root_node, parent_name, child_name, dtype)
