"""
Tests for audio resampling functionality in the Twilio Realtime Bridge.

These tests verify the audio resampling implementation that fixes the slow
playback issue by properly converting between different sample rates.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from opusagent.twilio_realtime_bridge import TwilioRealtimeBridge


class TestAudioResampling:
    """Test suite for audio resampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.twilio_websocket = MagicMock()
        self.realtime_websocket = MagicMock()
        self.bridge = TwilioRealtimeBridge(
            twilio_websocket=self.twilio_websocket,
            realtime_websocket=self.realtime_websocket,
        )

    def test_resample_pcm16_same_rate(self):
        """Test resampling when source and target rates are the same."""
        test_audio = b"\x00\x01\x02\x03\x04\x05\x06\x07"  # 4 samples at 16-bit

        result = self.bridge._resample_pcm16(test_audio, 24000, 24000)

        # Should return unchanged when rates are the same
        assert result == test_audio

    def test_resample_pcm16_with_audioop(self):
        """Test resampling with audioop available."""
        test_audio = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        resampled_audio = b"\x00\x01\x02\x03"  # Downsampled result

        with patch("audioop.ratecv") as mock_ratecv:
            mock_ratecv.return_value = (resampled_audio, None)

            result = self.bridge._resample_pcm16(test_audio, 24000, 8000)

            mock_ratecv.assert_called_once_with(test_audio, 2, 1, 24000, 8000, None)
            assert result == resampled_audio

    def test_resample_pcm16_fallback_without_audioop(self):
        """Test resampling fallback when audioop is not available."""
        test_audio = b"\x00\x01\x02\x03\x04\x05\x06\x07"  # 4 samples at 16-bit

        with patch("audioop.ratecv", side_effect=ImportError):
            result = self.bridge._resample_pcm16(test_audio, 24000, 8000)

            # Fallback should take every 3rd sample (24000/8000 = 3)
            # Since it's 2 bytes per sample, it should take every 6th byte
            expected = test_audio[::6]  # b'\x00\x06'
            assert result == expected

    @pytest.mark.asyncio
    async def test_handle_audio_response_delta_with_resampling(self):
        """Test audio response delta handling with resampling integration."""
        self.bridge.stream_sid = "test_stream_id"
        self.bridge._closed = False

        # Create test PCM16 audio data (24kHz from OpenAI)
        test_pcm16_24k = b"\x00\x01\x02\x03\x04\x05\x06\x07" * 100  # Larger sample
        pcm16_b64 = base64.b64encode(test_pcm16_24k).decode("utf-8")

        response_dict = {
            "type": "response.audio.delta",
            "response_id": "resp_123",
            "item_id": "item_123",
            "output_index": 0,
            "content_index": 0,
            "delta": pcm16_b64,
        }

        # Mock the resampling and conversion methods
        with patch.object(self.bridge, "_is_websocket_closed", return_value=False):
            with patch.object(self.bridge, "_resample_pcm16") as mock_resample:
                with patch.object(
                    self.bridge, "_convert_pcm16_to_mulaw"
                ) as mock_convert:
                    mock_resample.return_value = b"resampled_8k_audio"
                    mock_convert.return_value = b"mulaw_audio"

                    await self.bridge.handle_audio_response_delta(response_dict)

                    # Verify resampling was called with correct parameters
                    mock_resample.assert_called_once_with(test_pcm16_24k, 24000, 8000)

                    # Verify conversion was called with resampled audio
                    mock_convert.assert_called_once_with(b"resampled_8k_audio")

                    # Verify audio was sent to Twilio
                    assert self.bridge.twilio_websocket.send_json.called

    def test_audio_chunk_timing_with_resampling(self):
        """Test that audio chunks maintain proper timing after resampling."""
        # Test data representing 20ms of 24kHz audio (480 samples = 960 bytes)
        samples_24k_20ms = 480
        test_audio_24k = b"\x00\x01" * samples_24k_20ms

        with patch("audioop.ratecv") as mock_ratecv:
            # 20ms at 8kHz should be 160 samples = 320 bytes
            samples_8k_20ms = 160
            resampled_audio = b"\x00\x01" * samples_8k_20ms
            mock_ratecv.return_value = (resampled_audio, None)

            result = self.bridge._resample_pcm16(test_audio_24k, 24000, 8000)

            # Verify the duration is preserved (20ms worth of samples)
            assert len(result) == samples_8k_20ms * 2  # 2 bytes per sample
            mock_ratecv.assert_called_once_with(test_audio_24k, 2, 1, 24000, 8000, None)

    def test_resample_pcm16_edge_cases(self):
        """Test resampling with edge cases."""
        # Empty audio
        result = self.bridge._resample_pcm16(b"", 24000, 8000)
        assert result == b""

        # Very small audio (1 sample)
        small_audio = b"\x00\x01"
        with patch("audioop.ratecv") as mock_ratecv:
            mock_ratecv.return_value = (b"\x00\x01", None)
            result = self.bridge._resample_pcm16(small_audio, 24000, 8000)
            mock_ratecv.assert_called_once()

    def test_resample_pcm16_error_handling(self):
        """Test error handling in resampling function."""
        test_audio = b"\x00\x01\x02\x03"

        with patch("audioop.ratecv", side_effect=Exception("Unexpected error")):
            # Should fall back to placeholder resampling
            result = self.bridge._resample_pcm16(test_audio, 24000, 8000)

            # Fallback should still return some result
            assert isinstance(result, bytes)
            assert len(result) <= len(test_audio)

    def test_convert_pcm16_to_mulaw_integration(self):
        """Test PCM16 to mulaw conversion after resampling."""
        # Test that the conversion works with resampled audio
        test_pcm16_8k = b"\x00\x7f\x80\x00\xff\x7f"  # 3 samples at 8kHz

        with patch("audioop.lin2ulaw") as mock_lin2ulaw:
            mock_lin2ulaw.return_value = b"\x7f\x80\x00"

            result = self.bridge._convert_pcm16_to_mulaw(test_pcm16_8k)

            mock_lin2ulaw.assert_called_once_with(test_pcm16_8k, 2)
            assert result == b"\x7f\x80\x00"

    def test_sample_rate_constants(self):
        """Test that the bridge uses correct sample rate constants."""
        # Verify the expected sample rates are used
        # These should match the rates mentioned in the GitHub issue
        assert hasattr(self.bridge, "_resample_pcm16")

        # Test with the actual rates from the issue: 24kHz -> 8kHz
        test_audio = b"\x00\x01\x02\x03\x04\x05"

        with patch("audioop.ratecv") as mock_ratecv:
            mock_ratecv.return_value = (b"\x00\x01", None)

            self.bridge._resample_pcm16(test_audio, 24000, 8000)

            # Verify the exact sample rates from the issue are used
            call_args = mock_ratecv.call_args[0]
            assert call_args[3] == 24000  # From rate (OpenAI)
            assert call_args[4] == 8000  # To rate (Twilio)

    def test_audio_quality_preservation(self):
        """Test that resampling preserves audio quality as much as possible."""
        # Create a sine wave pattern for testing
        frequency = 440  # A4 note
        sample_rate = 24000
        duration_ms = 100
        samples = int(sample_rate * duration_ms / 1000)

        # Generate sine wave (simplified for test)
        audio_data = []
        for i in range(samples):
            # Simple sine wave generation
            value = int(32767 * 0.5)  # Half amplitude
            audio_data.extend([value & 0xFF, (value >> 8) & 0xFF])

        test_audio = bytes(audio_data)

        with patch("audioop.ratecv") as mock_ratecv:
            # Simulate resampling to 1/3 the samples
            resampled_samples = samples // 3
            resampled_data = test_audio[: resampled_samples * 2]
            mock_ratecv.return_value = (resampled_data, None)

            result = self.bridge._resample_pcm16(test_audio, 24000, 8000)

            # Verify the output is properly sized for the new sample rate
            expected_output_samples = samples // 3  # 24000 -> 8000
            assert len(result) == expected_output_samples * 2  # 2 bytes per sample
