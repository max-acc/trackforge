"""
Module containing the base class for configuration data.
"""
import yaml

class Config:
    """
    Base configuration loader.

    Loads configuration from a YAML file and provides access to the raw configuration dictionary. All specialized config
    classes inherit from this base class.
    """
    def __init__(self, config_path: str):
        """
        Initialize configuration by loading a YAML file.

        :param config_path: Path to configuration file.
        """
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def get_config_name(self) -> str:
        """
        Returns the config name of the included component

        :return:    The config name as defined in the YAML config.
        """
        return self.config['default']['name']


class BaseConfig(Config):
    """
    Base project-level configuration containing common settings.

    Provides access to project metadata and global parameters.
    """

    def get_project_name(self) -> str:
        """
        Return the name of the project.

        :return:    The project name as defined in the YAML config.
        """
        return self.config['project_name']

    def get_seed(self) -> int:
        """
        Return the random seed for reproducibility.

        :return:    Integer seed value.
        """
        return int(self.config['seed'])
