"""Unit tests for the Telephony Realtime Bridge module.

These tests validate the TelephonyRealtimeBridge class functionality including
WebSocket communication, event handling, audio processing, and session management.
"""

import asyncio
import base64
import json
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from opusagent.models.openai_api import ServerEventType
from opusagent.telephony_realtime_bridge import TelephonyRealtimeBridge
from opusagent.models.audiocodes_api import (
    TelephonyEventType,
    SessionInitiateMessage,
    UserStreamStartedResponse,
    PlayStreamStartMessage,
    PlayStreamChunkMessage,
    PlayStreamStopMessage,
)

# Test conversation ID
TEST_CONVERSATION_ID = "test-conversation-id"

@pytest.fixture
def mock_websocket():
    mock = AsyncMock()
    mock.send_json = AsyncMock()
    mock.iter_text = AsyncMock()
    mock.client_state = types.SimpleNamespace(DISCONNECTED=False)
    return mock


@pytest.fixture
def mock_openai_ws():
    mock = AsyncMock()
    mock.close_code = None
    mock.send = AsyncMock()
    return mock


@pytest.fixture
def bridge(mock_websocket, mock_openai_ws):
    return TelephonyRealtimeBridge(
        telephony_websocket=mock_websocket, realtime_websocket=mock_openai_ws
    )


def test_initialization(bridge):
    """Test that the bridge initializes correctly."""
    assert bridge.conversation_id is None
    assert bridge.media_format is None
    assert bridge.active_stream_id is None
    assert bridge.session_initialized is False
    assert bridge.speech_detected is False
    assert bridge._closed is False
    assert bridge.audio_chunks_sent == 0
    assert bridge.total_audio_bytes_sent == 0
    assert bridge.input_transcript_buffer == []
    assert bridge.output_transcript_buffer == []
    assert bridge.response_active is False
    assert bridge.pending_user_input is None
    assert bridge.response_id_tracker is None
    assert bridge.function_handler is not None


def test_event_handler_mappings(bridge):
    """Test that event handler mappings are properly configured."""
    # Test Telephony event handler mappings
    assert TelephonyEventType.SESSION_INITIATE in bridge.telephony_event_handlers
    assert TelephonyEventType.USER_STREAM_START in bridge.telephony_event_handlers
    assert TelephonyEventType.USER_STREAM_CHUNK in bridge.telephony_event_handlers
    assert TelephonyEventType.USER_STREAM_STOP in bridge.telephony_event_handlers
    assert TelephonyEventType.SESSION_END in bridge.telephony_event_handlers


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.initialize_session")
@patch("opusagent.telephony_realtime_bridge.SessionAcceptedResponse")
@patch("uuid.uuid4")
async def test_receive_from_telephony_session_initiate(
    mock_uuid, mock_session_response, mock_init_session, bridge
):
    # Setup mock data
    session_initiate = {
        "type": "session.initiate",
        "conversationId": "test-conversation-id",
        "expectAudioMessages": True,
        "botName": "test-bot",
        "caller": "+1234567890",
        "supportedMediaFormats": ["raw/lpcm16"],
    }

    # Set up the mock to return the message
    async def mock_iter():
        yield json.dumps(session_initiate)

    bridge.telephony_websocket.iter_text = mock_iter
    mock_init_session.return_value = None
    mock_uuid.return_value = "test-uuid"

    # Mock the session response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "type": "session.accepted",
        "conversationId": "test-conversation-id",
        "mediaFormat": "raw/lpcm16",
    }
    mock_session_response.return_value = mock_response

    # Mock the session creation event
    session_created_event = {
        "type": "session.created",
        "session": {
            "id": "test-session-id",
            "config": {
                "model": "gpt-4o-realtime-preview-2024-10-01",
                "voice": "alloy"
            }
        }
    }

    # Set up the realtime websocket to return the session created event
    async def mock_realtime_iter():
        yield json.dumps(session_created_event)

    bridge.realtime_websocket.__aiter__ = lambda: mock_realtime_iter()
    bridge.realtime_websocket.recv = AsyncMock(return_value=json.dumps(session_created_event))
    bridge.realtime_websocket.send = AsyncMock()

    # Mock the session update event
    session_update_event = {
        "type": "session.update",
        "session": {
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "voice": "alloy",
            "modalities": ["text", "audio"],
            "model": "gpt-4o-realtime-preview-2024-10-01"
        }
    }

    # Set up the mock to handle session update
    async def mock_send(data):
        if isinstance(data, str):
            data = json.loads(data)
        if data.get("type") == "session.update":
            # Simulate receiving session created event after update
            await asyncio.sleep(0.1)  # Small delay to simulate network
            bridge.session_initialized = True

    bridge.realtime_websocket.send.side_effect = mock_send

    # Run the method
    await bridge.receive_from_telephony()

    # Verify conversation_id was set
    assert bridge.conversation_id == "test-conversation-id"
    # Verify session was initialized
    assert bridge.session_initialized is True

    # Verify session response was sent
    bridge.telephony_websocket.send_json.assert_called_once_with(
        mock_response.model_dump()
    )


@pytest.mark.asyncio
async def test_handle_session_initiate(bridge):
    """Test handling of session initiate message."""
    # Setup mock data
    session_initiate = {
        "type": "session.initiate",
        "conversationId": TEST_CONVERSATION_ID,
        "expectAudioMessages": True,
        "botName": "test-bot",
        "caller": "+1234567890",
        "supportedMediaFormats": ["raw/lpcm16"],
    }

    with patch.object(bridge, "initialize_openai_session") as mock_init:
        await bridge.handle_session_initiate(session_initiate)

        # Verify conversation_id was set
        assert bridge.conversation_id == TEST_CONVERSATION_ID
        # Verify session initialization was called
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_handle_user_stream_start(bridge):
    """Test handling of user stream start message."""
    bridge.conversation_id = TEST_CONVERSATION_ID

    # Call the handler
    await bridge.handle_user_stream_start({})

    # Verify userStream.started message was sent
    bridge.telephony_websocket.send_json.assert_called_once()
    sent_message = bridge.telephony_websocket.send_json.call_args[0][0]
    assert sent_message["type"] == "userStream.started"
    assert sent_message["conversationId"] == TEST_CONVERSATION_ID


@pytest.mark.asyncio
async def test_handle_user_stream_chunk(bridge):
    """Test handling of user stream chunk message."""
    # Create test audio data
    test_audio = b"test audio data"
    audio_b64 = base64.b64encode(test_audio).decode("utf-8")

    data = {
        "type": "userStream.chunk",
        "audioChunk": audio_b64
    }

    bridge.realtime_websocket.close_code = None
    bridge._closed = False

    await bridge.handle_user_stream_chunk(data)

    # Verify audio was sent to OpenAI
    bridge.realtime_websocket.send.assert_called_once()
    sent_json = bridge.realtime_websocket.send.call_args[0][0]
    sent_data = json.loads(sent_json)
    assert sent_data["type"] == "input_audio_buffer.append"
    assert sent_data["audio"] == audio_b64


@pytest.mark.asyncio
async def test_handle_user_stream_stop(bridge):
    """Test handling of user stream stop message."""
    bridge.realtime_websocket.close_code = None
    bridge._closed = False

    await bridge.handle_user_stream_stop({})

    # Verify audio buffer was committed and response.create was sent
    calls = bridge.realtime_websocket.send.call_args_list
    assert any('input_audio_buffer.commit' in str(call) for call in calls)
    assert any('response.create' in str(call) for call in calls)


@pytest.mark.asyncio
async def test_handle_session_end(bridge):
    """Test handling of session end message."""
    bridge.conversation_id = TEST_CONVERSATION_ID

    session_end = {
        "type": "session.end",
        "conversationId": TEST_CONVERSATION_ID,
        "reasonCode": "client-disconnected",
        "reason": "Client Side"
    }

    with patch.object(bridge, "close") as mock_close:
        await bridge.handle_session_end(session_end)
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_audio_response_delta(bridge):
    """Test handling of audio response delta from OpenAI."""
    bridge.conversation_id = TEST_CONVERSATION_ID
    bridge._closed = False

    # Create test PCM16 audio data
    test_pcm16 = b"test pcm16 audio"
    pcm16_b64 = base64.b64encode(test_pcm16).decode("utf-8")

    response_dict = {
        "type": "response.audio.delta",
        "response_id": "resp-1",
        "item_id": "item-1",
        "output_index": 0,
        "content_index": 0,
        "delta": pcm16_b64
    }

    # Mock the websocket closed check
    with patch.object(bridge, "_is_websocket_closed", return_value=False):
        await bridge.handle_audio_response_delta(response_dict)

        # Verify audio was sent to telephony
        bridge.telephony_websocket.send_json.assert_called()
        sent_message = bridge.telephony_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "playStream.chunk"
        assert sent_message["conversationId"] == TEST_CONVERSATION_ID
        assert sent_message["audioChunk"] == pcm16_b64


@pytest.mark.asyncio
async def test_handle_audio_response_completion(bridge):
    """Test handling of audio response completion."""
    bridge.conversation_id = TEST_CONVERSATION_ID
    bridge.active_stream_id = "test-stream-id"

    await bridge.handle_audio_response_completion({})

    # Verify stream was stopped
    bridge.telephony_websocket.send_json.assert_called_once()
    sent_message = bridge.telephony_websocket.send_json.call_args[0][0]
    assert sent_message["type"] == "playStream.stop"
    assert sent_message["conversationId"] == TEST_CONVERSATION_ID
    assert sent_message["streamId"] == "test-stream-id"
    assert bridge.active_stream_id is None


@pytest.mark.asyncio
async def test_receive_from_telephony(bridge):
    """Test receiving and processing messages from telephony."""
    # Create test messages
    messages = [
        json.dumps({
            "type": "session.initiate",
            "conversationId": TEST_CONVERSATION_ID,
            "supportedMediaFormats": ["raw/lpcm16"]
        }),
        json.dumps({
            "type": "session.end",
            "conversationId": TEST_CONVERSATION_ID
        })
    ]

    # Mock the async iterator
    async def async_iterator():
        for msg in messages:
            yield msg

    bridge.telephony_websocket.iter_text.return_value = async_iterator()

    # Run the method
    await bridge.receive_from_telephony()

    # Verify bridge was closed
    assert bridge._closed is True


@pytest.mark.asyncio
async def test_receive_from_telephony_disconnect(bridge):
    """Test handling of telephony WebSocket disconnect."""
    bridge.telephony_websocket.iter_text.side_effect = WebSocketDisconnect()

    with patch.object(bridge, "close") as mock_close:
        await bridge.receive_from_telephony()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_receive_from_telephony_exception(bridge):
    """Test handling of exceptions during telephony message processing."""
    bridge.telephony_websocket.iter_text.side_effect = Exception("Test error")

    with patch.object(bridge, "close") as mock_close:
        await bridge.receive_from_telephony()
        mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_close_method(bridge):
    # Test closing when connections are active
    await bridge.close()

    # Verify both connections were closed
    bridge.realtime_websocket.close.assert_called_once()
    bridge.telephony_websocket.close.assert_called_once()
    assert bridge._closed is True


@pytest.mark.asyncio
async def test_close_method_with_errors(mock_websocket, mock_openai_ws):
    # Setup the mocks to raise exceptions when close is called
    mock_openai_ws.close.side_effect = Exception("OpenAI close error")
    mock_websocket.close.side_effect = Exception("WebSocket close error")

    bridge = TelephonyRealtimeBridge(
        telephony_websocket=mock_websocket, realtime_websocket=mock_openai_ws
    )

    # Should handle exceptions gracefully
    await bridge.close()
    assert bridge._closed is True


@pytest.mark.asyncio
async def test_close_method_already_closed(bridge):
    # Mark as already closed
    bridge._closed = True

    # Call close
    await bridge.close()

    # Verify no additional close attempts
    bridge.realtime_websocket.close.assert_not_called()
    bridge.telephony_websocket.close.assert_not_called()


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.InputAudioBufferAppendEvent")
async def test_receive_from_telephony_user_stream_chunk(mock_audio_event, bridge):
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.session_initialized = True
    bridge.realtime_websocket.close_code = None
    bridge._closed = False

    # Setup mock data
    audio_chunk = {
        "type": "userStream.chunk",
        "audioChunk": "AAAA",
    }  # valid base64 for two null bytes

    # Set up the mock to return the message
    async def mock_iter():
        yield json.dumps(audio_chunk)
        return

    bridge.telephony_websocket.iter_text = mock_iter

    # Mock the audio event model
    mock_event = MagicMock()
    mock_event.model_dump_json.return_value = (
        '{"type": "input_audio_buffer.append", "audio": "AAAA"}'
    )
    mock_audio_event.return_value = mock_event

    # Run the method
    await bridge.receive_from_telephony()

    # Verify audio was sent to OpenAI
    bridge.realtime_websocket.send.assert_called_once_with(
        mock_event.model_dump_json.return_value
    )


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.InputAudioBufferCommitEvent")
async def test_receive_from_telephony_user_stream_stop(mock_commit_event, bridge):
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.session_initialized = True
    bridge.realtime_websocket.close_code = None
    bridge._closed = False

    # Prepare a base64-encoded audio chunk of at least 3200 bytes
    import base64
    audio_bytes = b"\x00" * 3200
    audio_chunk_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Setup mock data: first a chunk, then a stop
    stream_chunk = {"type": "userStream.chunk", "audioChunk": audio_chunk_b64}
    stream_stop = {"type": "userStream.stop"}

    # Set up the mock to return the messages in sequence
    async def mock_iter():
        yield json.dumps(stream_chunk)
        yield json.dumps(stream_stop)
        return

    bridge.telephony_websocket.iter_text = mock_iter

    # Mock the commit event model
    mock_event = MagicMock()
    mock_event.model_dump_json.return_value = '{"type": "input_audio_buffer.commit"}'
    mock_commit_event.return_value = mock_event

    # Run the method
    await bridge.receive_from_telephony()

    # Verify commit event was sent to OpenAI
    bridge.realtime_websocket.send.assert_any_call(
        mock_event.model_dump_json.return_value
    )


@pytest.mark.asyncio
async def test_receive_from_telephony_session_end(bridge):
    # Setup mock data
    session_end = {
        "type": "session.end",
        "conversationId": TEST_CONVERSATION_ID,
        "reasonCode": "client-disconnected",
        "reason": "Test end"
    }

    # Set up the mock to return the message
    async def mock_iter():
        yield json.dumps(session_end)

    bridge.telephony_websocket.iter_text = mock_iter

    # Run the method
    await bridge.receive_from_telephony()

    # Verify bridge was closed
    assert bridge._closed is True
    bridge.realtime_websocket.close.assert_called_once()
    bridge.telephony_websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_receive_from_telephony_disconnect(bridge):
    # Simulate WebSocket disconnect
    bridge.telephony_websocket.client_state.DISCONNECTED = (
        False  # Ensure initially connected
    )
    bridge.telephony_websocket.iter_text.side_effect = WebSocketDisconnect()

    # Run the method
    await bridge.receive_from_telephony()

    # Verify bridge was closed
    assert bridge._closed is True
    bridge.realtime_websocket.close.assert_called_once()
    bridge.telephony_websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_receive_from_telephony_exception(bridge):
    # Simulate a generic exception
    bridge.telephony_websocket.client_state.DISCONNECTED = (
        False  # Ensure initially connected
    )
    bridge.telephony_websocket.iter_text.side_effect = Exception("Test error")

    # Run the method
    await bridge.receive_from_telephony()

    # Verify bridge was closed
    assert bridge._closed is True
    bridge.realtime_websocket.close.assert_called_once()
    bridge.telephony_websocket.close.assert_called_once()


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.ResponseAudioDeltaEvent")
@patch("opusagent.telephony_realtime_bridge.PlayStreamStartMessage")
@patch("opusagent.telephony_realtime_bridge.PlayStreamChunkMessage")
async def test_receive_from_realtime_audio_delta(
    mock_chunk_msg, mock_start_msg, mock_audio_event, bridge
):
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.media_format = "raw/lpcm16"
    bridge._closed = False

    # Create mock response from OpenAI
    audio_delta = {"type": "response.audio.delta", "delta": "base64audio"}

    # Set up the mock to return the message
    mock_iter = MagicMock()
    mock_iter.__aiter__.return_value = [json.dumps(audio_delta)]
    bridge.realtime_websocket = mock_iter

    # Mock the audio event model
    mock_event = MagicMock()
    mock_event.delta = "base64audio"
    mock_audio_event.return_value = mock_event

    # Mock the stream messages
    mock_start = MagicMock()
    mock_start.model_dump.return_value = {
        "type": "playStream.start",
        "conversationId": "test-conversation-id",
        "streamId": "test-stream-id",
        "mediaFormat": "raw/lpcm16",
    }
    mock_start_msg.return_value = mock_start

    mock_chunk = MagicMock()
    mock_chunk.model_dump.return_value = {
        "type": "playStream.chunk",
        "conversationId": "test-conversation-id",
        "streamId": "test-stream-id",
        "audioChunk": "base64audio",
    }
    mock_chunk_msg.return_value = mock_chunk

    # Run the method
    await bridge.receive_from_realtime()

    # Verify audio was sent to telephony
    assert (
        bridge.telephony_websocket.send_json.call_count == 2
    )  # One for start, one for chunk


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.PlayStreamStopMessage")
async def test_receive_from_realtime_audio_done(mock_stop_msg, bridge):
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.active_stream_id = "test-stream-id"
    bridge._closed = False

    # Create mock response from OpenAI
    audio_done = {"type": "response.audio.done"}

    # Set up the mock to return the message
    mock_iter = MagicMock()
    mock_iter.__aiter__.return_value = [json.dumps(audio_done)]
    bridge.realtime_websocket = mock_iter

    # Mock the stop message
    mock_stop = MagicMock()
    mock_stop.model_dump.return_value = {
        "type": "playStream.stop",
        "conversationId": "test-conversation-id",
        "streamId": "test-stream-id",
    }
    mock_stop_msg.return_value = mock_stop

    # Run the method
    await bridge.receive_from_realtime()

    # Verify stream was stopped
    bridge.telephony_websocket.send_json.assert_called_once_with(mock_stop.model_dump())
    assert bridge.active_stream_id is None


@pytest.mark.asyncio
async def test_receive_from_realtime_log_events(bridge):
    # Create mock log event from OpenAI
    log_event = {"type": "session.updated", "data": {"some": "data"}}

    # Configure the mock
    bridge.realtime_websocket.__aiter__.return_value = [json.dumps(log_event)]

    # Run the method
    await bridge.receive_from_realtime()

    # Verify no audio was sent to telephony for log events
    bridge.telephony_websocket.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_receive_from_realtime_exception(bridge):
    # Configure the mock to raise an exception
    bridge.realtime_websocket.__aiter__.side_effect = Exception("Test error")

    # Run the method
    await bridge.receive_from_realtime()

    # Verify bridge was closed
    assert bridge._closed is True
    bridge.realtime_websocket.close.assert_called_once()
    bridge.telephony_websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_handle_log_event_error(bridge):
    # Setup
    error_event = {
        "type": "error",
        "code": "test_error",
        "message": "Test error message",
        "details": {"additional": "info"},
    }

    # Call the handler
    with patch("opusagent.base_realtime_bridge.logger") as mock_logger:
        await bridge.handle_log_event(error_event)

        # Verify error was logged properly
        mock_logger.error.assert_any_call(
            "OpenAI Error: test_error - Test error message"
        )


@pytest.mark.asyncio
async def test_handle_log_event_other(bridge):
    # Setup
    log_event = {"type": "other_log_type", "data": "some data"}

    # Call the handler - should not raise an exception
    await bridge.handle_log_event(log_event)


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.ResponseTextDeltaEvent")
async def test_handle_text_delta(mock_text_delta, bridge):
    # Setup
    text_event = {"type": "response.text.delta", "delta": "Hello, how can I help?"}

    # Mock the text delta event model
    mock_event = MagicMock()
    mock_event.delta = "Hello, how can I help?"
    mock_text_delta.return_value = mock_event

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_text_and_transcript(text_event)

        # Verify text delta was logged
        mock_logger.info.assert_called_with(f"Text delta received: {mock_event.delta}")


@pytest.mark.asyncio
async def test_handle_transcript_delta(bridge):
    # Setup
    transcript_event = {
        "type": ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
        "delta": "Hello, I heard you say",
    }

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_text_and_transcript(transcript_event)

        # Verify transcript delta was logged - use assert_called_with instead of assert_called_once_with
        # to check just the most recent call
        mock_logger.info.assert_called_with(
            f"Received audio transcript delta: {transcript_event.get('delta', '')}"
        )


@pytest.mark.asyncio
@patch("opusagent.telephony_realtime_bridge.ResponseDoneEvent")
@patch("opusagent.telephony_realtime_bridge.PlayStreamStopMessage")
async def test_handle_response_completion(mock_stop_msg, mock_response_done, bridge):
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.active_stream_id = "test-stream-id"

    response_done_event = {"type": ServerEventType.RESPONSE_DONE}

    # Mock the response done event model
    mock_event = MagicMock()
    mock_event.response = {"id": "test-response-123"}
    mock_response_done.return_value = mock_event

    # Mock the stop message
    mock_stop = MagicMock()
    mock_stop.model_dump.return_value = {
        "type": "playStream.stop",
        "conversationId": "test-conversation-id",
        "streamId": "test-stream-id",
    }
    mock_stop_msg.return_value = mock_stop

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_response_completion(response_done_event)

        # The method logs two messages, first about response completion and then about stopping the stream
        # Check that the first call was for response completion
        calls = mock_logger.info.call_args_list
        assert len(calls) >= 1  # Make sure there was at least one call
        assert calls[0] == call("Response generation completed: test-response-123")

        # Verify stream was stopped
        bridge.telephony_websocket.send_json.assert_called_once_with(
            mock_stop.model_dump()
        )
        assert bridge.active_stream_id is None


@pytest.mark.asyncio
async def test_handle_speech_detection_started(bridge):
    # Setup
    # Need to use the enum value from ServerEventType not the string literal
    speech_started_event = {"type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED}

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_speech_detection(speech_started_event)

        # Verify speech detection was logged
        mock_logger.info.assert_called_with("Speech started detected")
        assert bridge.speech_detected is True


@pytest.mark.asyncio
async def test_handle_speech_detection_stopped(bridge):
    # Setup
    # Need to use the enum value from ServerEventType not the string literal
    speech_stopped_event = {"type": ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED}

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_speech_detection(speech_stopped_event)

        # Verify speech detection was logged
        mock_logger.info.assert_called_with("Speech stopped detected")
        assert bridge.speech_detected is False


@pytest.mark.asyncio
async def test_handle_speech_detection_committed(bridge):
    # Setup
    buffer_committed_event = {"type": "input_audio_buffer.committed"}

    # Call the handler
    with patch("opusagent.telephony_realtime_bridge.logger") as mock_logger:
        await bridge.handle_speech_detection(buffer_committed_event)

        # Verify commitment was logged
        mock_logger.info.assert_called_with("Audio buffer committed")


@pytest.mark.asyncio
async def test_race_condition_prevention(bridge):
    """Test that race condition is prevented when user input arrives during active response."""
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.session_initialized = True
    bridge.realtime_websocket.close_code = None
    bridge._closed = False
    bridge.total_audio_bytes_sent = 3200  # Sufficient audio for commit
    
    # Simulate active response
    bridge.response_active = True
    bridge.response_id_tracker = "test-response-id"
    
    # Test data for user stream stop during active response
    stream_stop = {"type": "userStream.stop"}
    
    # Call handle_user_stream_stop during active response
    await bridge.handle_user_stream_stop(stream_stop)
    
    # Verify that no response.create was sent (since response is active)
    # The bridge should have queued the user input instead
    assert bridge.pending_user_input is not None
    assert bridge.pending_user_input["audio_committed"] is True
    assert "timestamp" in bridge.pending_user_input
    
    # Verify no response.create call was made
    sent_messages = [call[0][0] for call in bridge.realtime_websocket.send.call_args_list]
    response_create_messages = [msg for msg in sent_messages if '"type": "response.create"' in msg]
    assert len(response_create_messages) == 0, "No response.create should be sent during active response"


@pytest.mark.asyncio
async def test_response_state_tracking(bridge):
    """Test that response state is properly tracked through the lifecycle."""
    # Initial state
    assert bridge.response_active is False
    assert bridge.response_id_tracker is None
    assert bridge.pending_user_input is None
    
    # Simulate response created event
    response_created = {
        "type": "response.created",
        "response": {"id": "test-response-123"}
    }
    await bridge.handle_response_created(response_created)
    
    # Verify response is now active
    assert bridge.response_active is True
    assert bridge.response_id_tracker == "test-response-123"
    
    # Simulate response completion
    response_done = {
        "type": "response.done",
        "response": {"id": "test-response-123", "status": "completed"}
    }
    await bridge.handle_response_completion(response_done)
    
    # Verify response is no longer active
    assert bridge.response_active is False


@pytest.mark.asyncio
async def test_pending_input_processing(bridge):
    """Test that pending user input is processed after response completion."""
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.realtime_websocket.close_code = None
    bridge._closed = False
    
    # Set pending user input (as would happen during active response)
    bridge.pending_user_input = {
        "audio_committed": True,  
        "timestamp": 1234567890.0
    }
    
    # Simulate response completion
    response_done = {
        "type": "response.done", 
        "response": {"id": "test-response-123", "status": "completed"}
    }
    await bridge.handle_response_completion(response_done)
    
    # Verify pending input was processed (response.create should be called)
    sent_messages = [call[0][0] for call in bridge.realtime_websocket.send.call_args_list]
    response_create_messages = [msg for msg in sent_messages if '"type": "response.create"' in msg]
    assert len(response_create_messages) == 1, "response.create should be sent for queued input"
    
    # Verify pending input was cleared
    assert bridge.pending_user_input is None


@pytest.mark.asyncio
async def test_normal_flow_without_race_condition(bridge):
    """Test normal flow when no response is active."""
    # Setup
    bridge.conversation_id = "test-conversation-id"
    bridge.session_initialized = True
    bridge.realtime_websocket.close_code = None
    bridge._closed = False
    bridge.total_audio_bytes_sent = 3200  # Sufficient audio for commit
    bridge.response_active = False  # No active response
    
    # Test data
    stream_stop = {"type": "userStream.stop"}
    
    # Call handle_user_stream_stop when no response is active
    await bridge.handle_user_stream_stop(stream_stop)
    
    # Verify response.create was sent normally
    sent_messages = [call[0][0] for call in bridge.realtime_websocket.send.call_args_list]
    response_create_messages = [msg for msg in sent_messages if '"type": "response.create"' in msg]
    assert len(response_create_messages) == 1, "response.create should be sent when no active response"
    
    # Verify no pending input was created
    assert bridge.pending_user_input is None


@pytest.mark.asyncio
async def test_create_response_helper(bridge):
    """Test the _create_response helper method."""
    bridge.realtime_websocket.close_code = None
    bridge._closed = False
    
    # Call the helper method
    await bridge._create_response()
    
    # Verify response.create was sent with correct structure
    bridge.realtime_websocket.send.assert_called_once()
    sent_json = bridge.realtime_websocket.send.call_args[0][0]
    sent_data = json.loads(sent_json)
    
    assert sent_data["type"] == "response.create"
    assert sent_data["response"]["modalities"] == ["text", "audio"]
    assert sent_data["response"]["output_audio_format"] == "pcm16"
    assert sent_data["response"]["voice"] == "verse"
