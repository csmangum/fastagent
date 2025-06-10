"""
FastAPI server for AudioCodes VoiceAI Connect real-time voice agent integration.

This module initializes and configures the FastAPI application that serves as
the webhook endpoint for AudioCodes VoiceAI Connect Enterprise platform. It
implements the WebSocket protocol defined by the AudioCodes Bot API to enable
real-time voice interactions between users and AI voice agents.

The server handles incoming WebSocket connections, routes messages to appropriate
handlers, and maintains conversation state throughout the call session.
"""

import asyncio
import os
import sys
from pathlib import Path
import wave
import base64

import dotenv
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from opusagent.config.logging_config import configure_logging
from opusagent.telephony_realtime_bridge import (
    TelephonyRealtimeBridge,
    initialize_session,
)
from opusagent.twilio_realtime_bridge import TwilioRealtimeBridge

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_URL = "wss://grand-collie-complete.ngrok-free.app/twilio-agent"
# Load environment variables from .env file if it exists
env_path = Path(".") / ".env"
if env_path.exists():
    dotenv.load_dotenv(env_path)

# Configure logging
logger = configure_logging("main")
# Get configuration from environment variables
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Create FastAPI application
app = FastAPI(
    title="Real-Time Voice Agent",
    description="Integration between AudioCodes VoiceAI Connect and OpenAI Realtime API",
    version="1.0.0",
)


@app.websocket("/voice-agent")
async def handle_call(telephony_websocket: WebSocket):
    """Handle WebSocket connections between telephony provider and OpenAI."""
    client_address = telephony_websocket.client
    logger.info(f"------ Incoming telephony connection from {client_address} ------")
    await telephony_websocket.accept()
    logger.info(f"------ Telephony connection accepted from {client_address} ------")

    try:
        logger.info("Attempting to connect to OpenAI Realtime API...")
        
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            subprotocols=["realtime"],
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=5,
            ping_timeout=20,
            close_timeout=10,
        ) as realtime_websocket:
            logger.info("OpenAI WebSocket connection established")
            bridge = TelephonyRealtimeBridge(telephony_websocket, realtime_websocket)
            logger.info("Telephony-Realtime bridge created")
            
            try:
                logger.info("Initializing realtime-websocket session...")
                await initialize_session(realtime_websocket)
                logger.info("realtime-websocket session initialized")
            except Exception as e:
                logger.error(f"Error initializing realtime-websocket session: {e}")
                raise

            # Run both tasks and handle cleanup
            try:
                logger.info("Starting bridge tasks...")
                await asyncio.gather(
                    bridge.receive_from_telephony(), bridge.receive_from_realtime()
                )
            except Exception as e:
                logger.error(f"Error in main connection loop: {e}")
                raise
            finally:
                logger.info("Closing bridge...")
                await bridge.close()
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Invalid status code from OpenAI: {e}")
        logger.error(f"Response headers: {e.response_headers}")
        await telephony_websocket.close()
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"OpenAI connection closed: {e}")
        logger.error(f"Close code: {e.code}")
        logger.error(f"Close reason: {e.reason}")
        await telephony_websocket.close()
    except Exception as e:
        logger.error(f"Error establishing OpenAI connection: {e}")
        await telephony_websocket.close()


@app.websocket("/twilio-agent")
async def handle_twilio_call(twilio_websocket: WebSocket):
    """Handle WebSocket connections between Twilio Media Streams and OpenAI."""
    client_address = twilio_websocket.client
    logger.info(f"------ Incoming Twilio connection from {client_address} ------")
    await twilio_websocket.accept()
    logger.info(f"------ Twilio connection accepted from {client_address} ------")

    try:
        logger.info("Attempting to connect to OpenAI Realtime API for Twilio...")
        
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            subprotocols=["realtime"],
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
            ping_interval=5,
            ping_timeout=20,
            close_timeout=10,
        ) as realtime_websocket:
            logger.info("OpenAI WebSocket connection established for Twilio")
            bridge = TwilioRealtimeBridge(twilio_websocket, realtime_websocket)
            logger.info("Twilio-Realtime bridge created")
            
            try:
                logger.info("Initializing Twilio realtime-websocket session...")
                await bridge.initialize_openai_session()
                logger.info("Twilio realtime-websocket session initialized")
            except Exception as e:
                logger.error(f"Error initializing Twilio realtime-websocket session: {e}")
                raise

            # Run both tasks and handle cleanup
            try:
                logger.info("Starting Twilio bridge tasks...")
                await asyncio.gather(
                    bridge.receive_from_twilio(), bridge.receive_from_realtime()
                )
            except Exception as e:
                logger.error(f"Error in Twilio main connection loop: {e}")
                raise
            finally:
                logger.info("Closing Twilio bridge...")
                await bridge.close()
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Invalid status code from OpenAI (Twilio): {e}")
        logger.error(f"Response headers: {e.response_headers}")
        await twilio_websocket.close()
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"OpenAI connection closed (Twilio): {e}")
        logger.error(f"Close code: {e.code}")
        logger.error(f"Close reason: {e.reason}")
        await twilio_websocket.close()
    except Exception as e:
        logger.error(f"Error establishing OpenAI connection (Twilio): {e}")
        await twilio_websocket.close()


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Handle incoming Twilio voice calls.
    
    This endpoint receives incoming voice calls from Twilio and responds with TwiML
    instructions to connect the call to our WebSocket endpoint for real-time AI interaction.
    """
    logger.info(f"------ Incoming Twilio voice call ------")    
    # Create a TwiML response
    response = VoiceResponse()
    logger.info(f" Response: {response}")
    # Connect the call to our WebSocket endpoint using the public URL
    connect = response.connect()
    logger.info(f"------ Connecting call to WebSocket endpoint ------")
    logger.info(f"------ {SERVER_URL} ------")
    connect.stream(url=SERVER_URL)
    logger.info(f"------ Connected call to WebSocket endpoint ------")
    
    # Return XML response with proper Content-Type header
    return Response(content=str(response), media_type="application/xml")


@app.get("/")
async def root():
    """Root endpoint to display basic information about the API.

    Returns:
        dict: Basic information about the API and its purpose.
    """
    return {
        "name": "Real-Time Voice Agent",
        "description": "Integration between AudioCodes VoiceAI Connect and OpenAI Realtime API",
        "version": "1.0.0",
        "endpoints": {
            "/voice-agent": "WebSocket endpoint for AudioCodes VoiceAI Connect",
            "/twilio-agent": "WebSocket endpoint for Twilio Media Streams",
            "/twilio/voice": "Webhook endpoint for incoming Twilio voice calls",
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on http://{HOST}:{PORT}")
    # Configure uvicorn with optimized settings for real-time audio
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        # Optimized WebSocket settings for low latency audio streaming
        websocket_ping_interval=10,  # Reduce ping frequency to avoid interference with audio
        websocket_max_size=4194304,  # 4MB - optimal for audio chunks without excessive buffering
        websocket_ping_timeout=30,  # Longer timeout for stability
        # Use HTTP/1.1 for lower overhead than HTTP/2
        http="h11",
        # Additional performance optimizations
        loop="asyncio",
        timeout_keep_alive=30,
        limit_concurrency=1000,
        limit_max_requests=10000,
    )
