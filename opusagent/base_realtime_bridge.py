"""Base class for real-time bridges between telephony systems and OpenAI Realtime API.

This module provides a base class that handles common functionality for bridges between
telephony systems (like Twilio or AudioCodes) and the OpenAI Realtime API. It manages
WebSocket connections, audio streaming, session management, and event processing.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from opusagent.config.logging_config import configure_logging
from opusagent.function_handler import FunctionHandler
from opusagent.models.openai_api import (
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    LogEventType,
    ResponseAudioDeltaEvent,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
    ServerEventType,
    SessionConfig,
    SessionUpdateEvent,
)

# Configure logging
logger = configure_logging("base_realtime_bridge")

class BaseRealtimeBridge:
    """Base class for handling bidirectional communication between telephony systems and OpenAI Realtime API.

    This class provides common functionality for managing WebSocket connections between
    telephony systems and the OpenAI Realtime API, handling audio streaming, session
    management, and event processing in both directions.

    Attributes:
        telephony_websocket (WebSocket): FastAPI WebSocket connection for telephony
        realtime_websocket (websockets.WebSocketClientProtocol): WebSocket connection to OpenAI Realtime API
        session_initialized (bool): Whether the OpenAI Realtime API session has been initialized
        speech_detected (bool): Whether speech is currently being detected
        _closed (bool): Flag indicating whether the bridge connections are closed
        audio_chunks_sent (int): Number of audio chunks sent to the OpenAI Realtime API
        total_audio_bytes_sent (int): Total number of bytes sent to the OpenAI Realtime API
        input_transcript_buffer (list): Buffer for accumulating input audio transcriptions
        output_transcript_buffer (list): Buffer for accumulating output audio transcriptions
        function_handler (FunctionHandler): Handler for managing function calls from the OpenAI Realtime API
        response_active (bool): Whether a response is currently being generated
        pending_user_input (Optional[Dict]): Queue for user input during active response
        response_id_tracker (Optional[str]): Current response ID being processed
    """

    def __init__(
        self,
        telephony_websocket: WebSocket,
        realtime_websocket: websockets.WebSocketClientProtocol,
    ):
        """Initialize the bridge with WebSocket connections.

        Args:
            telephony_websocket (WebSocket): FastAPI WebSocket connection for telephony
            realtime_websocket (websockets.WebSocketClientProtocol): WebSocket connection to OpenAI Realtime API
        """
        self.telephony_websocket = telephony_websocket
        self.realtime_websocket = realtime_websocket
        self.session_initialized = False
        self.speech_detected = False
        self._closed = False

        # Audio buffer tracking for debugging
        self.audio_chunks_sent = 0
        self.total_audio_bytes_sent = 0

        # Transcript buffers for logging full transcripts
        self.input_transcript_buffer = []  # User → AI
        self.output_transcript_buffer = []  # AI → User

        # Response state tracking to prevent race conditions
        self.response_active = False  # Track if response is being generated
        self.pending_user_input = None  # Queue for user input during active response
        self.response_id_tracker = None  # Track current response ID

        # Initialize function handler
        self.function_handler = FunctionHandler(realtime_websocket)

        # Create event handler mappings for realtime events
        self.realtime_event_handlers = {
            # Session events
            ServerEventType.SESSION_UPDATED: self.handle_session_update,
            ServerEventType.SESSION_CREATED: self.handle_session_update,
            # Conversation events
            ServerEventType.CONVERSATION_ITEM_CREATED: lambda x: logger.info(
                "Conversation item created"
            ),
            # Speech detection events
            ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self.handle_speech_detection,
            ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self.handle_speech_detection,
            ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED: self.handle_speech_detection,
            # Response events
            ServerEventType.RESPONSE_CREATED: self.handle_response_created,
            ServerEventType.RESPONSE_AUDIO_DELTA: self.handle_audio_response_delta,
            ServerEventType.RESPONSE_AUDIO_DONE: self.handle_audio_response_completion,
            ServerEventType.RESPONSE_TEXT_DELTA: self.handle_text_and_transcript,
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA: self.handle_audio_transcript_delta,
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE: self.handle_audio_transcript_done,
            ServerEventType.RESPONSE_DONE: self.handle_response_completion,
            # Output item and content part events
            ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED: self.handle_output_item_added,
            ServerEventType.RESPONSE_CONTENT_PART_ADDED: self.handle_content_part_added,
            ServerEventType.RESPONSE_CONTENT_PART_DONE: self.handle_content_part_done,
            ServerEventType.RESPONSE_OUTPUT_ITEM_DONE: self.handle_output_item_done,
            # Input audio transcription events
            ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA: self.handle_input_audio_transcription_delta,
            ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self.handle_input_audio_transcription_completed,
            # Function call events
            ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA: self.function_handler.handle_function_call_arguments_delta,
            ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: self.function_handler.handle_function_call_arguments_done,
        }

    async def close(self):
        """Safely close both WebSocket connections.

        This method ensures both the telephony and OpenAI Realtime API WebSocket connections
        are properly closed, handling any exceptions that may occur during the process.
        """
        if not self._closed:
            self._closed = True
            try:
                if (
                    self.realtime_websocket
                    and self.realtime_websocket.close_code is None
                ):
                    await self.realtime_websocket.close()
            except Exception as e:
                logger.error(f"Error closing OpenAI connection: {e}")
            try:
                if self.telephony_websocket and not self._is_websocket_closed():
                    await self.telephony_websocket.close()
            except Exception as e:
                logger.error(f"Error closing telephony connection: {e}")

    def _is_websocket_closed(self):
        """Check if telephony WebSocket is closed.

        Returns True if the WebSocket is closed or in an unusable state.
        """
        try:
            from starlette.websockets import WebSocketState
            return (
                not self.telephony_websocket
                or self.telephony_websocket.client_state == WebSocketState.DISCONNECTED
            )
        except ImportError:
            # Fallback check without WebSocketState
            return not self.telephony_websocket

    async def handle_session_update(self, response_dict):
        """Handle session update events from the OpenAI Realtime API."""
        response_type = response_dict["type"]

        if response_type == ServerEventType.SESSION_UPDATED:
            logger.info("OpenAI session updated successfully")
        elif response_type == ServerEventType.SESSION_CREATED:
            logger.info("OpenAI session created successfully")
            self.session_initialized = True
            try:
                await self.send_initial_conversation_item()
            except Exception as e:
                logger.error(f"Error sending initial conversation item: {e}")

    async def handle_speech_detection(self, response_dict):
        """Handle speech detection events from the OpenAI Realtime API."""
        response_type = response_dict["type"]

        if response_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("Speech started detected")
            self.speech_detected = True
        elif response_type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("Speech stopped detected")
            self.speech_detected = False
        elif response_type == ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED:
            logger.info("Audio buffer committed confirmed by OpenAI")

    async def handle_response_created(self, response_dict):
        """Handle response created events from the OpenAI Realtime API."""
        self.response_active = True
        response_data = response_dict.get("response", {})
        self.response_id_tracker = response_data.get("id")
        logger.info(f"Response generation started: {self.response_id_tracker}")
        
        if self.pending_user_input:
            logger.info(f"Note: Pending user input exists while starting new response")

    async def handle_response_completion(self, response_dict):
        """Handle response completion events from the OpenAI Realtime API."""
        response_done = ResponseDoneEvent(**response_dict)
        self.response_active = False
        self.response_id_tracker = None
        response_id = response_done.response.get("id") if response_done.response else None
        logger.info(f"Response generation completed: {response_id}")

        if self.pending_user_input:
            logger.info("Processing queued user input after response completion")
            try:
                await self._create_response()
                logger.info("Successfully processed queued user input")
            except Exception as e:
                logger.error(f"Error processing queued user input: {e}")
            finally:
                self.pending_user_input = None

    async def handle_text_and_transcript(self, response_dict):
        """Handle text and transcript events from OpenAI."""
        response_type = response_dict["type"]

        if response_type == ServerEventType.RESPONSE_TEXT_DELTA:
            text_delta = ResponseTextDeltaEvent(**response_dict)
            logger.info(f"Text delta received: {text_delta.delta}")

    async def handle_audio_transcript_delta(self, response_dict):
        """Handle audio transcript delta events from OpenAI."""
        delta = response_dict.get("delta", "")
        if delta:
            self.output_transcript_buffer.append(delta)
        logger.debug(f"Audio transcript delta: {delta}")

    async def handle_audio_transcript_done(self, response_dict):
        """Handle audio transcript completion events from OpenAI."""
        full_transcript = "".join(self.output_transcript_buffer)
        logger.info(f"Full AI transcript: {full_transcript}")
        self.output_transcript_buffer.clear()

    async def handle_input_audio_transcription_delta(self, response_dict):
        """Handle input audio transcription delta events from OpenAI."""
        delta = response_dict.get("delta", "")
        if delta:
            self.input_transcript_buffer.append(delta)
        logger.debug(f"Input audio transcription delta: {delta}")

    async def handle_input_audio_transcription_completed(self, response_dict):
        """Handle input audio transcription completion events from OpenAI."""
        full_transcript = "".join(self.input_transcript_buffer)
        logger.info(f"Full user transcript: {full_transcript}")
        self.input_transcript_buffer.clear()

    async def handle_output_item_added(self, response_dict):
        """Handle output item added events from OpenAI."""
        item = response_dict.get("item", {})
        logger.info(f"Output item added: {item}")

        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            function_name = item.get("name")
            item_id = item.get("id")

            if call_id and function_name:
                if call_id not in self.function_handler.active_function_calls:
                    self.function_handler.active_function_calls[call_id] = {
                        "arguments_buffer": "",
                        "item_id": item_id,
                        "output_index": response_dict.get("output_index", 0),
                        "response_id": response_dict.get("response_id"),
                        "function_name": function_name,
                    }
                else:
                    self.function_handler.active_function_calls[call_id][
                        "function_name"
                    ] = function_name

                logger.info(
                    f"Function call captured: {function_name} (call_id: {call_id})"
                )

    async def handle_content_part_added(self, response_dict):
        """Handle content part added events from OpenAI."""
        logger.info(f"Content part added: {response_dict.get('part', {})}")

    async def handle_content_part_done(self, response_dict):
        """Handle content part completion events from OpenAI."""
        logger.info("Content part completed")

    async def handle_output_item_done(self, response_dict):
        """Handle output item completion events from OpenAI."""
        logger.info("Output item completed")

    async def handle_log_event(self, response_dict):
        """Handle log events from OpenAI."""
        response_type = response_dict["type"]

        if response_type == "error":
            error_code = response_dict.get("code", "unknown")
            error_message = response_dict.get("message", "No message provided")
            logger.error(f"OpenAI Error: {error_code} - {error_message}")
        elif response_type == "input_audio_buffer.speech_started":
            logger.info("OpenAI detected speech started")
        elif response_type == "input_audio_buffer.speech_stopped": 
            logger.info("OpenAI detected speech stopped")
        elif response_type == "input_audio_buffer.committed":
            logger.info("OpenAI confirmed audio buffer committed")
        elif response_type == "response.done":
            logger.info("OpenAI response completed")
        else:
            logger.debug(f"OpenAI log event: {response_type}")

    async def receive_from_realtime(self):
        """Receive and process events from the OpenAI Realtime API."""
        try:
            async for openai_message in self.realtime_websocket:
                if self._closed:
                    break

                response_dict = json.loads(openai_message)
                response_type = response_dict["type"]
                
                # Log all OpenAI events for debugging
                if response_type in ["response.audio.delta", "response.audio_transcript.delta"]:
                    logger.debug(f"Received OpenAI message type: {response_type}")
                else:
                    logger.info(f"Received OpenAI message type: {response_type}")

                # Handle log events first
                if response_type in [event.value for event in LogEventType]:
                    await self.handle_log_event(response_dict)
                    continue

                # Dispatch to appropriate handler
                handler = self.realtime_event_handlers.get(response_type)
                if handler:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(response_dict)
                        else:
                            handler(response_dict)
                    except Exception as e:
                        logger.error(f"Error in handler for {response_type}: {e}")
                else:
                    logger.warning(f"Unknown OpenAI event type: {response_type}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
            await self.close()
        except Exception as e:
            logger.error(f"Error in receive_from_realtime: {e}")
            if not self._closed:
                await self.close()

    async def _create_response(self):
        """Create a new response request to OpenAI Realtime API."""
        try:
            response_create = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "output_audio_format": "pcm16",
                    "temperature": 0.8,
                    "max_output_tokens": 4096,
                    "voice": "alloy",
                },
            }
            await self.realtime_websocket.send(json.dumps(response_create))
            logger.info("Response creation triggered")
        except Exception as e:
            logger.error(f"Error creating response: {e}")

    async def initialize_openai_session(self, system_message: str, model: str = "gpt-4o-realtime-preview-2024-10-01"):
        """Initialize the OpenAI Realtime API session.

        Args:
            system_message (str): The system message to use for the session
            model (str): The model to use for the session
        """
        session_config = SessionConfig(
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            voice="alloy",
            instructions=system_message,
            modalities=["text", "audio"],
            temperature=0.8,
            model=model,
            tools=[
                {
                    "type": "function",
                    "name": "call_intent",
                    "description": "Get the user's intent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": [
                                    "card_replacement",
                                    "account_inquiry",
                                    "other",
                                ],
                            },
                        },
                        "required": ["intent"],
                    },
                },
            ],
            input_audio_noise_reduction={"type": "near_field"},
            input_audio_transcription={"model": "whisper-1"},
            max_response_output_tokens=4096,
            tool_choice="auto",
        )

        session_update = SessionUpdateEvent(
            type="session.update", session=session_config
        )
        logger.info("Initializing OpenAI session")
        await self.realtime_websocket.send(session_update.model_dump_json())

    async def send_initial_conversation_item(self):
        """Send initial conversation item to start the AI interaction."""
        try:
            initial_conversation = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a customer service agent for Bank of Peril. Start by saying 'Hello! How can I help you today?'",
                        }
                    ],
                },
            }

            await self.realtime_websocket.send(json.dumps(initial_conversation))
            await asyncio.sleep(1)

            # Trigger initial response
            await self._create_response()
            logger.info("Initial conversation flow initiated")

        except Exception as e:
            logger.error(f"Error sending initial conversation item: {e}")

    # Abstract methods that must be implemented by subclasses
    async def handle_audio_response_delta(self, response_dict):
        """Handle audio response delta events from OpenAI.
        
        This method must be implemented by subclasses to handle the specific
        audio format and delivery mechanism of the telephony system.
        """
        raise NotImplementedError("Subclasses must implement handle_audio_response_delta")

    async def handle_audio_response_completion(self, response_dict):
        """Handle audio response completion events from OpenAI.
        
        This method must be implemented by subclasses to handle the specific
        audio completion mechanism of the telephony system.
        """
        raise NotImplementedError("Subclasses must implement handle_audio_response_completion")

    async def receive_from_telephony(self):
        """Receive and process messages from the telephony WebSocket.
        
        This method must be implemented by subclasses to handle the specific
        message format and event types of the telephony system.
        """
        raise NotImplementedError("Subclasses must implement receive_from_telephony") 