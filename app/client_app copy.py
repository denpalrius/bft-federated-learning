from flwr.client import ClientApp

from app.client.client_manager import ClientManager
from app.utils.wandb_mod import get_wandb_mod


def client_fn(context):
    manager = ClientManager(context)
    return manager.create_client()


def create_client_app():
    return ClientApp(
        client_fn=client_fn,
        mods=[
            get_wandb_mod("W&Bs Mod"),
        ],
    )


# Initialize client app
app = create_client_app()
