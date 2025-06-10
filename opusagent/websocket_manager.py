"""
WebSocket connection manager for AudioCodes VoiceAI Connect integration.

This module implements the server-side handling of the AudioCodes Bot API WebSocket protocol,
providing the infrastructure to:
- Accept and manage WebSocket connections
- Route incoming messages to appropriate handler functions
- Maintain conversation state
- Respond with properly formatted messages according to the AudioCodes Bot API

The WebSocketManager class is the central component that orchestrates all WebSocket
communications between the AudioCodes VoiceAI Connect platform and the voice bot implementation.
#! This module is not being used.
"""

import json
import logging
import socket
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastapi import WebSocket
from pydantic import ValidationError

from opusagent.config.constants import LOGGER_NAME
from opusagent.handlers.activity_handlers import handle_activities
from opusagent.handlers.session_handlers import (
    handle_connection_validate,
    handle_session_end,
    handle_session_initiate,
    handle_session_resume,
)
from opusagent.handlers.stream_handlers import (
    handle_user_stream_chunk,
    handle_user_stream_start,
    handle_user_stream_stop,
)
from opusagent.models.audiocodes_api import (
    ActivitiesMessage,
    ConnectionValidateMessage,
    IncomingMessage,
    OutgoingMessage,
    SessionEndMessage,
    SessionInitiateMessage,
    SessionResumeMessage,
    UserStreamChunkMessage,
    UserStreamStartMessage,
    UserStreamStopMessage,
)
from opusagent.models.conversation import ConversationManager

logger = logging.getLogger(LOGGER_NAME)

# Type hint for handler functions
HandlerFunc = Callable[
    [Dict[str, Any], WebSocket, ConversationManager],
    Awaitable[Optional[Union[Dict[str, Any], OutgoingMessage]]],
]


class WebSocketManager:
    """Manages WebSocket connections and routes messages to appropriate handlers for AudioCodes VoiceAI Connect API.

    This class implements the server-side of the AudioCodes Bot API WebSocket protocol, handling:
    - Session lifecycle (initiate, resume, end)
    - Audio streaming (start, chunk, stop)
    - Activity handling
    - Connection validation

    Each message type is routed to a specific handler function based on the message's "type" field.
    """

    def __init__(self):
        self.conversation_manager = ConversationManager()

        # Define handlers dictionary
        self.handlers: Dict[str, HandlerFunc] = {
            "session.initiate": handle_session_initiate,
            "session.resume": handle_session_resume,
            "userStream.start": handle_user_stream_start,
            "userStream.chunk": handle_user_stream_chunk,
            "userStream.stop": handle_user_stream_stop,
            "activities": handle_activities,
            "session.end": handle_session_end,
            "connection.validate": handle_connection_validate,
        }

    async def _optimize_socket(self, websocket: WebSocket) -> None:
        """
        Optimize the WebSocket's underlying TCP socket for low-latency transmission.

        Args:
            websocket: The FastAPI WebSocket connection
        """
        try:
            # Access the underlying socket via the transport
            client = websocket.client
            if hasattr(client, "sock") and client.sock is not None:
                # Disable Nagle's algorithm to send packets immediately
                client.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("Optimized socket: TCP_NODELAY enabled for low latency")
        except Exception as e:
            logger.warning(f"Could not optimize socket: {e}")

    async def handle_websocket(self, websocket: WebSocket):
        """Handle a WebSocket connection throughout its lifecycle.

        Args:
            websocket (WebSocket): The FastAPI WebSocket connection object

        This method:
        1. Accepts the WebSocket connection
        2. Processes incoming messages in a loop
        3. Routes each message to the appropriate handler based on type
        4. Handles responses and error conditions
        5. Performs cleanup when connection ends

        The connection remains active for the entire duration of the conversation
        until either the client disconnects or a session.end message is received.
        """
        await websocket.accept()
        # Optimize socket for low latency after acceptance
        await self._optimize_socket(websocket)
        logger.info("WebSocket connection established and optimized for low latency")
        conversation_id = None

        try:
            while True:
                # Receive the message
                data = await websocket.receive_text()
                message_dict = json.loads(data)

                # Extract the message type
                message_type = message_dict.get("type")

                # Fast path for audio chunks to minimize processing overhead
                if message_type == "userStream.chunk":
                    # Direct handling for audio chunks to minimize latency
                    await handle_user_stream_chunk(
                        message_dict, websocket, self.conversation_manager
                    )
                    continue

                # Extract the conversation ID if available
                conversation_id = message_dict.get("conversationId")

                logger.info(
                    f"Received message type: {message_type}"
                    + (
                        f" for conversation: {conversation_id}"
                        if conversation_id
                        else ""
                    )
                )

                # Process the message using the appropriate handler
                if message_type in self.handlers:
                    handler = self.handlers[message_type]
                    response = await handler(
                        message_dict, websocket, self.conversation_manager
                    )

                    # Special handling for session.end
                    if message_type == "session.end":
                        break  # End the WebSocket connection

                    # If the handler returned a response and hasn't sent it yet, send it now
                    if response and message_type not in [
                        "session.initiate"
                    ]:  # session.initiate handler sends its own response
                        # Check if response is a Pydantic model or a dict
                        if hasattr(response, "json"):
                            await websocket.send_text(response.json())
                        else:
                            await websocket.send_text(json.dumps(response))
                        logger.info(f"Sent response for {message_type}")
                else:
                    logger.warning(f"Unhandled message type received: {message_type}")

        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        finally:
            # Clean up the conversation if it exists
            if conversation_id:
                self.conversation_manager.remove_conversation(conversation_id)
                logger.info(f"Conversation removed during cleanup: {conversation_id}")
            await websocket.close()
            logger.info("WebSocket connection closed")
