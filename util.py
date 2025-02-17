import os
import yaml

from pydantic.v1.utils import deep_update


def load_config(cfg_path, max_inherit=30):
    """ Integrate the inheritance relations in config to a final config

    Args:
        cfg_path (str): Loaded config path
        max_inherit (int): max iteration of inheritance

    Returns:
        dst_config (Dict): final config
    """
    config_list = []
    config_dir, config_name = os.path.split(cfg_path)
    inherit_count = -1
    
    config = { 'inherit': config_name }
    while config['inherit'] is not None:
        assert os.path.exists(os.path.join(config_dir, config['inherit'])), \
            f"{os.path.join(config_dir, config['inherit'])} does not exist"

        config_name = config['inherit']
        with open(os.path.join(config_dir, config_name), 'r') as stream:
            config = yaml.safe_load(stream)
            assert 'inherit' in config.keys(), f'inherit key does not exist in {config_name}'
        config_list.insert(0, config)
        inherit_count += 1
        if inherit_count > max_inherit:
            raise RecursionError(f'Inheritation should be less than {max_inherit}')
    
    dst_config = {}
    for config in config_list:
        dst_config = deep_update(dst_config, config)

    del dst_config['inherit']
    from omegaconf import OmegaConf
    conf = OmegaConf.create(dst_config)
    return conf
