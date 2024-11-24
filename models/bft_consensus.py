from collections import defaultdict
from enums.bft_message import BFTMessage
from enums.message_type import MessageType
from typing import Any
import time
from collections import defaultdict


class PBFTConsensus:
    def __init__(self, total_nodes: int, max_faulty: int):
        self.total_nodes = total_nodes
        self.max_faulty = max_faulty
        self.sequence_number = 0
        self.view_number = 0
        self.prepare_messages = defaultdict(dict)
        self.commit_messages = defaultdict(dict)
        self.prepared_messages = set()
        self.committed_messages = set()
    
    def create_message(self, msg_type: MessageType, client_id: int, content: Any) -> BFTMessage:
        self.sequence_number += 1
        return BFTMessage(
            msg_type=msg_type,
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            client_id=client_id,
            content=content,
            timestamp=time.time(),
            signature=f"sig_{client_id}_{self.sequence_number}"  # Simplified signature
        )
    
    def validate_message(self, message: BFTMessage) -> bool:
        # In practice, implement proper message validation including signature verification
        return True
    
    def process_message(self, message: BFTMessage) -> bool:
        if not self.validate_message(message):
            return False
            
        if message.msg_type == MessageType.PRE_PREPARE:
            # Primary sends pre-prepare message
            return True
            
        elif message.msg_type == MessageType.PREPARE:
            self.prepare_messages[message.sequence_number][message.client_id] = message
            if len(self.prepare_messages[message.sequence_number]) >= 2 * self.max_faulty + 1:
                self.prepared_messages.add(message.sequence_number)
                return True
                
        elif message.msg_type == MessageType.COMMIT:
            self.commit_messages[message.sequence_number][message.client_id] = message
            if len(self.commit_messages[message.sequence_number]) >= 2 * self.max_faulty + 1:
                self.committed_messages.add(message.sequence_number)
                return True
                
        return False