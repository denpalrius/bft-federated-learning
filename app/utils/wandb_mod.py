import time

import wandb
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common import ConfigsRecord
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_wandb_mod(name: str) -> Mod:
    def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
        """Flower Mod that logs the metrics dictionary returned by the client's fit
        function to Weights & Biases."""

        print("======= W&B Mod =======")
        print("Message Type: ", msg.metadata)
        
        server_round = int(msg.metadata.group_id)

        print(f"Server Round: {server_round}")
        
        if msg.metadata.message_type == MessageType.TRAIN:
            run_id = msg.metadata.run_id
            group_name = f"Run ID: {run_id}"

            node_id = str(msg.metadata.dst_node_id)
            run_name = f"Node ID: {node_id}"

            wandb.init(
                project=name,
                group=group_name,
                name=run_name,
                id=f"{run_id}_{node_id}",
                resume="allow",
                reinit=True,
            )

        start_time = time.time()

        reply = app(msg, context)

        time_diff = time.time() - start_time

        # if the `ClientApp` just processed a "fit" message, let's log some metrics to W&B
        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
            metrics = reply.content.configs_records

            results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))

            results_to_log["fit_time"] = time_diff

            wandb.log(results_to_log, step=int(server_round), commit=True)

        return reply

    return wandb_mod
