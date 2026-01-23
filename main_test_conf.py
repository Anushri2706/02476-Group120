import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configshydra", config_name="config")
def my_app(cfg: DictConfig):
    # This prints the config in a clean YAML format
    print(OmegaConf.to_yaml(cfg))
    
    # ... rest of your code

if __name__ == "__main__":
    my_app()
