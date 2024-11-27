from flwr.client import ClientApp
from app.client.client_manager import ClientManager


def client_fn(context):
    manager = ClientManager(context)
    return manager.create_client()


def create_client_app():
    return ClientApp(client_fn)


# Initialize client app
app = create_client_app()
