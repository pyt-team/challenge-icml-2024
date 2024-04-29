import pprint

import omegaconf
import rootutils

rootutils.setup_root("./", indicator=".project-root", pythonpath=True)
root_folder = rootutils.find_root()


def load_dataset_config(dataset_name: str) -> omegaconf.DictConfig:
    r"""Load the dataset configuration.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    omegaconf.DictConfig
        Dataset configuration.
    """
    dataset_config_path = f"{root_folder}/configs/datasets/{dataset_name}.yaml"
    dataset_config = omegaconf.OmegaConf.load(dataset_config_path)
    # Print configuration
    pprint.pp(dict(dataset_config.copy()))
    return dataset_config


def load_transform_config(
    transform_type: str, transform_id: str
) -> omegaconf.DictConfig:
    r"""Load the transform configuration.

    Parameters
    ----------
    transform_name : str
        Name of the transform.
    transform_id : str
        Identifier of the transform. If the transform is a topological lifting,
        it should include the type of the lifting and the identifier separated by a '/'
        (e.g. graph2cell/cycle_lifting).

    Returns
    -------
    omegaconf.DictConfig
        Transform configuration.
    """
    transform_config_path = (
        f"{root_folder}/configs/transforms/{transform_type}/{transform_id}.yaml"
    )
    transform_config = omegaconf.OmegaConf.load(transform_config_path)
    # Print configuration
    pprint.pp(dict(transform_config.copy()))
    return transform_config
