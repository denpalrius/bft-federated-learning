""" ServerApp for Flower """

from flwr.server import ServerApp
from flwr.server import ServerApp
from app.server.server_manager import ServerManager


def server_fn(context):
    manager = ServerManager(context)
    return manager.create_components()

def create_server_app():
    server_app = ServerApp(server_fn=server_fn)
    return server_app


# Initialize server app
app = create_server_app()
