"""
Unit tests for the Twilio Realtime Bridge module.

These tests validate the TwilioRealtimeBridge class functionality including
WebSocket communication, event handling, audio processing, and session management.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from opusagent.twilio_realtime_bridge import TwilioRealtimeBridge
from opusagent.models.twilio_api import (
    TwilioEventType,
    ConnectedMessage,
    StartMessage,
    MediaMessage,
    StopMessage,
    DTMFMessage,
    MarkMessage,
)

# Test SIDs - Using clearly fake test values
TEST_ACCOUNT_SID = "ACtest1234567890abcdef1234567890abcdef"
TEST_CALL_SID = "CAtest1234567890abcdef1234567890abcdef"
TEST_STREAM_SID = "MStest1234567890abcdef1234567890abcdef"

class TestTwilioRealtimeBridge:
    """Tests for the TwilioRealtimeBridge class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Use MagicMock without spec to allow __bool__ override
        self.twilio_websocket = MagicMock()
        # Make async methods return AsyncMocks
        self.twilio_websocket.send_json = AsyncMock()
        self.twilio_websocket.iter_text = AsyncMock()
        self.twilio_websocket.close = AsyncMock()
        # Make the mock truthy
        self.twilio_websocket.__bool__ = MagicMock(return_value=True)

        self.realtime_websocket = AsyncMock()
        self.realtime_websocket.close_code = None

        # Create bridge instance
        self.bridge = TwilioRealtimeBridge(
            twilio_websocket=self.twilio_websocket,
            realtime_websocket=self.realtime_websocket,
        )

    def test_initialization(self):
        """Test that the bridge initializes correctly."""
        assert self.bridge.twilio_websocket == self.twilio_websocket
        assert self.bridge.realtime_websocket == self.realtime_websocket
        assert self.bridge.stream_sid is None
        assert self.bridge.account_sid is None
        assert self.bridge.call_sid is None
        assert self.bridge.media_format is None
        assert self.bridge.session_initialized is False
        assert self.bridge.speech_detected is False
        assert self.bridge._closed is False
        assert self.bridge.audio_chunks_sent == 0
        assert self.bridge.total_audio_bytes_sent == 0
        assert self.bridge.audio_buffer == []
        assert self.bridge.input_transcript_buffer == []
        assert self.bridge.output_transcript_buffer == []
        assert self.bridge.mark_counter == 0
        assert self.bridge.function_handler is not None

    def test_event_handler_mappings(self):
        """Test that event handler mappings are properly configured."""
        # Test Twilio event handler mappings
        assert TwilioEventType.CONNECTED in self.bridge.twilio_event_handlers
        assert TwilioEventType.START in self.bridge.twilio_event_handlers
        assert TwilioEventType.MEDIA in self.bridge.twilio_event_handlers
        assert TwilioEventType.STOP in self.bridge.twilio_event_handlers
        assert TwilioEventType.DTMF in self.bridge.twilio_event_handlers
        assert TwilioEventType.MARK in self.bridge.twilio_event_handlers

    @pytest.mark.asyncio
    async def test_handle_connected(self):
        """Test handling of Twilio connected message."""
        data = {
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        }

        await self.bridge.handle_connected(data)
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_handle_start(self):
        """Test handling of Twilio start message."""
        data = {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": TEST_STREAM_SID,
            "start": {
                "streamSid": TEST_STREAM_SID,
                "accountSid": TEST_ACCOUNT_SID,
                "callSid": TEST_CALL_SID,
                "tracks": ["inbound", "outbound"],
                "customParameters": {},
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
            },
        }

        with patch.object(self.bridge, "initialize_openai_session") as mock_init:
            await self.bridge.handle_start(data)

            assert self.bridge.stream_sid == TEST_STREAM_SID
            assert self.bridge.account_sid == TEST_ACCOUNT_SID
            assert self.bridge.call_sid == TEST_CALL_SID
            assert self.bridge.media_format == "audio/x-mulaw"
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_media(self):
        """Test handling of Twilio media message with audio data."""
        # Create test audio data
        test_audio = b"test audio data"
        audio_b64 = base64.b64encode(test_audio).decode("utf-8")

        data = {
            "event": "media",
            "sequenceNumber": "2",
            "streamSid": TEST_STREAM_SID,
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "1000",
                "payload": audio_b64,
            },
        }

        with patch.object(self.bridge, "_convert_mulaw_to_pcm16") as mock_convert:
            mock_convert.return_value = b"converted pcm16 data"

            await self.bridge.handle_media(data)

            # Should buffer the audio
            assert len(self.bridge.audio_buffer) == 1
            assert self.bridge.audio_buffer[0] == test_audio

    @pytest.mark.asyncio
    async def test_handle_stop(self):
        """Test handling of Twilio stop message."""
        data = {
            "event": "stop",
            "sequenceNumber": "5",
            "streamSid": TEST_STREAM_SID,
            "stop": {
                "accountSid": TEST_ACCOUNT_SID,
                "callSid": TEST_CALL_SID,
            },
        }

        # Add some audio to buffer
        self.bridge.audio_buffer = [b"test"]

        with patch.object(self.bridge, "_commit_audio_buffer") as mock_commit:
            with patch.object(self.bridge, "close") as mock_close:
                await self.bridge.handle_stop(data)

                mock_commit.assert_called_once()
                mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_dtmf(self):
        """Test handling of Twilio DTMF message."""
        data = {
            "event": "dtmf",
            "streamSid": TEST_STREAM_SID,
            "sequenceNumber": "3",
            "dtmf": {"track": "inbound_track", "digit": "5"},
        }

        await self.bridge.handle_dtmf(data)
        # Should not raise exceptions and log the DTMF digit

    @pytest.mark.asyncio
    async def test_handle_mark(self):
        """Test handling of Twilio mark message."""
        data = {
            "event": "mark",
            "sequenceNumber": "4",
            "streamSid": TEST_STREAM_SID,
            "mark": {"name": "test_mark"},
        }

        await self.bridge.handle_mark(data)
        # Should not raise exceptions and log the mark completion

    @pytest.mark.asyncio
    async def test_handle_audio_response_delta(self):
        """Test handling of audio response delta from OpenAI."""
        self.bridge.stream_sid = TEST_STREAM_SID

        # Ensure bridge is not closed
        self.bridge._closed = False

        # Create test PCM16 audio data
        test_pcm16 = b"test pcm16 audio"
        pcm16_b64 = base64.b64encode(test_pcm16).decode("utf-8")

        response_dict = {
            "type": "response.audio.delta",
            "response_id": "resp_123",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": pcm16_b64,
        }

        # Mock both the websocket closed check and the conversion method
        with patch.object(self.bridge, "_is_websocket_closed", return_value=False):
            with patch.object(self.bridge, "_convert_pcm16_to_mulaw") as mock_convert:
                mock_convert.return_value = b"converted mulaw"

                await self.bridge.handle_audio_response_delta(response_dict)

                mock_convert.assert_called_once_with(test_pcm16)
                self.twilio_websocket.send_json.assert_called_once()

    def test_get_twilio_event_type(self):
        """Test conversion of string event types to TwilioEventType enum."""
        # Valid event type
        event_type = self.bridge._get_twilio_event_type("connected")
        assert event_type == TwilioEventType.CONNECTED

        # Invalid event type
        event_type = self.bridge._get_twilio_event_type("invalid")
        assert event_type is None

    @pytest.mark.asyncio
    async def test_receive_from_twilio(self):
        """Test receiving and processing messages from Twilio."""
        messages = [
            json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
            json.dumps(
                {
                    "event": "start",
                    "sequenceNumber": "1",
                    "streamSid": TEST_STREAM_SID,
                    "start": {
                        "streamSid": TEST_STREAM_SID,
                        "accountSid": TEST_ACCOUNT_SID,
                        "callSid": TEST_CALL_SID,
                        "tracks": ["inbound"],
                        "customParameters": {},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "channels": 1,
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "event": "stop",
                    "sequenceNumber": "2",
                    "streamSid": TEST_STREAM_SID,
                    "stop": {
                        "accountSid": TEST_ACCOUNT_SID,
                        "callSid": TEST_CALL_SID,
                    },
                }
            ),
        ]

        # Mock the async iterator directly
        async def async_iterator():
            for msg in messages:
                yield msg

        self.twilio_websocket.iter_text.return_value = async_iterator()

        # Since we can't easily mock the handlers without interfering with the actual execution,
        # let's just test that it processes without error and calls close on stop
        with patch.object(self.bridge, "close") as mock_close:
            await self.bridge.receive_from_twilio()
            # Should be called when processing stop event
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_from_twilio_disconnect(self):
        """Test handling of Twilio WebSocket disconnect."""
        self.twilio_websocket.iter_text.side_effect = WebSocketDisconnect

        with patch.object(self.bridge, "close") as mock_close:
            await self.bridge.receive_from_twilio()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_receive_from_twilio_exception(self):
        """Test handling of exceptions during Twilio message processing."""
        self.twilio_websocket.iter_text.side_effect = Exception("Test error")

        with patch.object(self.bridge, "close") as mock_close:
            await self.bridge.receive_from_twilio()
            mock_close.assert_called_once()

# Integration test class
class TestTwilioRealtimeBridgeIntegration:
    """Integration tests for TwilioRealtimeBridge."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.twilio_websocket = MagicMock()
        # Make async methods return AsyncMocks
        self.twilio_websocket.send_json = AsyncMock()
        self.twilio_websocket.iter_text = AsyncMock()
        self.twilio_websocket.close = AsyncMock()
        # Make the mock truthy so it doesn't fail the boolean check in handle_audio_response_delta
        self.twilio_websocket.__bool__ = MagicMock(return_value=True)

        self.realtime_websocket = AsyncMock()
        self.realtime_websocket.close_code = None

        self.bridge = TwilioRealtimeBridge(
            twilio_websocket=self.twilio_websocket,
            realtime_websocket=self.realtime_websocket,
        )

    @pytest.mark.asyncio
    async def test_full_call_flow(self):
        """Test a complete call flow from start to finish."""
        # 1. Connected
        await self.bridge.handle_connected(
            {"event": "connected", "protocol": "Call", "version": "1.0.0"}
        )

        # 2. Start
        with patch.object(self.bridge, "initialize_openai_session"):
            await self.bridge.handle_start(
                {
                    "event": "start",
                    "sequenceNumber": "1",
                    "streamSid": TEST_STREAM_SID,
                    "start": {
                        "streamSid": TEST_STREAM_SID,
                        "accountSid": TEST_ACCOUNT_SID,
                        "callSid": TEST_CALL_SID,
                        "tracks": ["inbound"],
                        "customParameters": {},
                        "mediaFormat": {
                            "encoding": "audio/x-mulaw",
                            "sampleRate": 8000,
                            "channels": 1,
                        },
                    },
                }
            )

        assert self.bridge.stream_sid == TEST_STREAM_SID

        # 3. Media processing
        test_audio = b"test"
        await self.bridge.handle_media(
            {
                "event": "media",
                "sequenceNumber": "2",
                "streamSid": TEST_STREAM_SID,
                "media": {
                    "track": "inbound",
                    "chunk": "1",
                    "timestamp": "1000",
                    "payload": base64.b64encode(test_audio).decode("utf-8"),
                },
            }
        )

        assert len(self.bridge.audio_buffer) == 1

        # 4. Stop
        with patch.object(self.bridge, "close"):
            await self.bridge.handle_stop(
                {
                    "event": "stop",
                    "sequenceNumber": "3",
                    "streamSid": TEST_STREAM_SID,
                    "stop": {
                        "accountSid": TEST_ACCOUNT_SID,
                        "callSid": TEST_CALL_SID,
                    },
                }
            )

    @pytest.mark.asyncio
    async def test_bidirectional_audio_flow(self):
        """Test bidirectional audio flow between Twilio and OpenAI."""
        self.bridge.stream_sid = TEST_STREAM_SID

        # Simulate audio from Twilio
        test_audio = b"test_audio"
        with patch.object(self.bridge, "_convert_mulaw_to_pcm16") as mock_mulaw_convert:
            mock_mulaw_convert.return_value = b"pcm16_data"

            # Fill buffer to trigger processing
            self.bridge.audio_buffer = [test_audio] * 10

            await self.bridge.handle_media(
                {
                    "event": "media",
                    "sequenceNumber": "2",
                    "streamSid": TEST_STREAM_SID,
                    "media": {
                        "track": "inbound",
                        "chunk": "1",
                        "timestamp": "1000",
                        "payload": base64.b64encode(test_audio).decode("utf-8"),
                    },
                }
            )

        # Simulate audio response from OpenAI
        pcm16_data = b"response_audio"
        with patch.object(self.bridge, "_convert_pcm16_to_mulaw") as mock_pcm16_convert:
            mock_pcm16_convert.return_value = b"mulaw_response"

            await self.bridge.handle_audio_response_delta(
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(pcm16_data).decode("utf-8"),
                }
            )

            mock_pcm16_convert.assert_called_once_with(pcm16_data)
            self.twilio_websocket.send_json.assert_called_once()
