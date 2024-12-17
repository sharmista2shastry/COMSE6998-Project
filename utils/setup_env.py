import os
import wandb


def setup_environment():
    os.environ["FORCE_CUDA"] = "1"
    wandb_api_key = os.getenv("WANDB_API_KEY", "default")
    wandb_host = os.getenv("WANDB_HOST", "default")

    if 'default' in [wandb_api_key, wandb_host]:
        print('WARNING: Wandb credentials missing!')

    # Setting the values for this program
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_HOST"] = wandb_host
    wandb.login()
