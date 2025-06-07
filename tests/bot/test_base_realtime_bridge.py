"""Unit tests for the Base Realtime Bridge module.

These tests validate the BaseRealtimeBridge class functionality including
WebSocket communication, event handling, audio processing, and session management.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from opusagent.base_realtime_bridge import BaseRealtimeBridge
from opusagent.models.openai_api import (
    ServerEventType,
    ResponseDoneEvent,
    ResponseTextDeltaEvent,
)

class TestBaseRealtimeBridge:
    """Tests for the BaseRealtimeBridge class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Use MagicMock without spec to allow __bool__ override
        self.telephony_websocket = MagicMock()
        # Make async methods return AsyncMocks
        self.telephony_websocket.send_json = AsyncMock()
        self.telephony_websocket.iter_text = AsyncMock()
        self.telephony_websocket.close = AsyncMock()
        # Make the mock truthy
        self.telephony_websocket.__bool__ = MagicMock(return_value=True)

        self.realtime_websocket = AsyncMock()
        self.realtime_websocket.close_code = None

        # Create bridge instance
        self.bridge = BaseRealtimeBridge(
            telephony_websocket=self.telephony_websocket,
            realtime_websocket=self.realtime_websocket,
        )

    def test_initialization(self):
        """Test that the bridge initializes correctly."""
        assert self.bridge.telephony_websocket == self.telephony_websocket
        assert self.bridge.realtime_websocket == self.realtime_websocket
        assert self.bridge.session_initialized is False
        assert self.bridge.speech_detected is False
        assert self.bridge._closed is False
        assert self.bridge.audio_chunks_sent == 0
        assert self.bridge.total_audio_bytes_sent == 0
        assert self.bridge.input_transcript_buffer == []
        assert self.bridge.output_transcript_buffer == []
        assert self.bridge.response_active is False
        assert self.bridge.pending_user_input is None
        assert self.bridge.response_id_tracker is None
        assert self.bridge.function_handler is not None

    def test_event_handler_mappings(self):
        """Test that event handler mappings are properly configured."""
        # Test OpenAI event handler mappings
        assert ServerEventType.SESSION_UPDATED in self.bridge.realtime_event_handlers
        assert ServerEventType.SESSION_CREATED in self.bridge.realtime_event_handlers
        assert ServerEventType.RESPONSE_AUDIO_DELTA in self.bridge.realtime_event_handlers
        assert ServerEventType.RESPONSE_DONE in self.bridge.realtime_event_handlers
        assert ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED in self.bridge.realtime_event_handlers
        assert ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED in self.bridge.realtime_event_handlers
        assert ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED in self.bridge.realtime_event_handlers

    @pytest.mark.asyncio
    async def test_close_method(self):
        """Test that the close method properly closes both WebSocket connections."""
        # Mock the WebSocketState to properly test the close method
        with patch("starlette.websockets.WebSocketState") as mock_state:
            # Set up the mock so that DISCONNECTED comparison returns False
            mock_state.DISCONNECTED = "disconnected"
            self.telephony_websocket.client_state = "connected"  # Not DISCONNECTED

            # Test normal close
            await self.bridge.close()

            assert self.bridge._closed is True
            self.realtime_websocket.close.assert_called_once()
            self.telephony_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_session_update(self):
        """Test handling of session update events."""
        # Test session updated
        response_dict = {"type": ServerEventType.SESSION_UPDATED}
        await self.bridge.handle_session_update(response_dict)
        assert self.bridge.session_initialized is False

        # Test session created
        response_dict = {"type": ServerEventType.SESSION_CREATED}
        with patch.object(self.bridge, "send_initial_conversation_item") as mock_send:
            await self.bridge.handle_session_update(response_dict)
            assert self.bridge.session_initialized is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_speech_detection(self):
        """Test handling of speech detection events."""
        # Test speech started
        response_dict = {"type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED}
        await self.bridge.handle_speech_detection(response_dict)
        assert self.bridge.speech_detected is True

        # Test speech stopped
        response_dict = {"type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED}
        await self.bridge.handle_speech_detection(response_dict)
        assert self.bridge.speech_detected is False

        # Test buffer committed
        response_dict = {"type": ServerEventType.INPUT_AUDIO_BUFFER_COMMITTED}
        await self.bridge.handle_speech_detection(response_dict)

    @pytest.mark.asyncio
    async def test_handle_response_created(self):
        """Test handling of response created events."""
        response_dict = {
            "type": ServerEventType.RESPONSE_CREATED,
            "response": {"id": "test-response-id"}
        }
        await self.bridge.handle_response_created(response_dict)
        assert self.bridge.response_active is True
        assert self.bridge.response_id_tracker == "test-response-id"

    @pytest.mark.asyncio
    async def test_handle_response_completion(self):
        """Test handling of response completion events."""
        # Set up pending input
        self.bridge.pending_user_input = {"audio_committed": True}
        self.bridge.response_active = True
        self.bridge.response_id_tracker = "test-response-id"

        response_dict = {
            "type": ServerEventType.RESPONSE_DONE,
            "response": {"id": "test-response-id"}
        }

        with patch.object(self.bridge, "_create_response") as mock_create:
            await self.bridge.handle_response_completion(response_dict)
            assert self.bridge.response_active is False
            assert self.bridge.response_id_tracker is None
            assert self.bridge.pending_user_input is None
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_text_and_transcript(self):
        """Test handling of text and transcript events."""
        response_dict = {
            "type": ServerEventType.RESPONSE_TEXT_DELTA,
            "delta": "test text"
        }
        
        # Create a mock object with the delta attribute
        mock_event = MagicMock()
        mock_event.delta = "test text"
        
        # Mock the ResponseTextDeltaEvent class
        with patch('opusagent.base_realtime_bridge.ResponseTextDeltaEvent', return_value=mock_event) as mock_class:
            await self.bridge.handle_text_and_transcript(response_dict)
            mock_class.assert_called_once_with(**response_dict)

    @pytest.mark.asyncio
    async def test_handle_audio_transcript_delta(self):
        """Test handling of audio transcript delta events."""
        response_dict = {"delta": "test transcript"}
        await self.bridge.handle_audio_transcript_delta(response_dict)
        assert self.bridge.output_transcript_buffer == ["test transcript"]

    @pytest.mark.asyncio
    async def test_handle_audio_transcript_done(self):
        """Test handling of audio transcript completion events."""
        self.bridge.output_transcript_buffer = ["test", " transcript"]
        await self.bridge.handle_audio_transcript_done({})
        assert self.bridge.output_transcript_buffer == []

    @pytest.mark.asyncio
    async def test_handle_input_audio_transcription_delta(self):
        """Test handling of input audio transcription delta events."""
        response_dict = {"delta": "test input"}
        await self.bridge.handle_input_audio_transcription_delta(response_dict)
        assert self.bridge.input_transcript_buffer == ["test input"]

    @pytest.mark.asyncio
    async def test_handle_input_audio_transcription_completed(self):
        """Test handling of input audio transcription completion events."""
        self.bridge.input_transcript_buffer = ["test", " input"]
        await self.bridge.handle_input_audio_transcription_completed({})
        assert self.bridge.input_transcript_buffer == []

    @pytest.mark.asyncio
    async def test_receive_from_realtime(self):
        """Test receiving and processing messages from OpenAI Realtime API."""
        # Create test messages
        messages = [
            json.dumps({"type": ServerEventType.SESSION_UPDATED}),
            json.dumps({"type": ServerEventType.RESPONSE_DONE})
        ]

        # Create a proper async iterator
        async def mock_iter():
            for msg in messages:
                yield msg

        # Set up the mock so that the websocket itself is an async iterable
        self.realtime_websocket.__aiter__ = lambda self=self: mock_iter()

        # Run the method
        await self.bridge.receive_from_realtime()

        # Verify no errors occurred
        assert self.bridge._closed is False

    @pytest.mark.asyncio
    async def test_receive_from_realtime_disconnect(self):
        """Test handling of OpenAI WebSocket disconnect."""
        self.realtime_websocket.__aiter__.side_effect = Exception("Test error")

        with patch.object(self.bridge, "close") as mock_close:
            await self.bridge.receive_from_realtime()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_response(self):
        """Test creating a new response."""
        self.bridge.realtime_websocket.close_code = None
        self.bridge._closed = False

        await self.bridge._create_response()

        # Verify response.create was sent
        self.realtime_websocket.send.assert_called_once()
        sent_json = self.realtime_websocket.send.call_args[0][0]
        sent_data = json.loads(sent_json)

        assert sent_data["type"] == "response.create"
        assert sent_data["response"]["modalities"] == ["text", "audio"]
        assert sent_data["response"]["output_audio_format"] == "pcm16"
        assert sent_data["response"]["voice"] == "alloy" 