from dataclasses import dataclass
from typing import Any

from enums.message_type import MessageType

@dataclass
class BFTMessage:
    msg_type: MessageType
    view_number: int
    sequence_number: int
    client_id: int
    content: Any
    timestamp: float
    signature: str  # In practice, use proper cryptographic signatures
