import asyncio
import base64
import json
import logging
import threading
import tkinter as tk
from tkinter import ttk
import wave
import pyaudio
import websockets
import numpy as np
from scipy import signal
import queue
import uuid
from pathlib import Path
import sys

class AudioInterface:
    def __init__(self, root):
        print("[DEBUG] Initializing AudioInterface", file=sys.stderr)
        self.root = root
        self.root.title("Audio Chat Interface")
        
        # Audio settings
        self.CHUNK = 32000
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # WebSocket state
        self.ws = None
        self.connected = False
        self.conversation_id = None
        self.loop = None
        self.ws_thread = None
        
        # Audio state
        self.recording = False
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        
        # Create UI
        self.create_widgets()
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_thread, daemon=True)
        self.audio_thread.start()
        
        
        print("[DEBUG] AudioInterface initialized", file=sys.stderr)
        
    def create_widgets(self):
        print("[DEBUG] Creating widgets", file=sys.stderr)
        # Connection frame
        conn_frame = ttk.LabelFrame(self.root, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Hardcoded URL, no entry
        self.url_var = tk.StringVar(value="ws://localhost:8000/voice-agent")
        ttk.Label(conn_frame, text="ws://localhost:8000/voice-agent").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.RIGHT, padx=5)
        
        # Talk button
        self.talk_btn = ttk.Button(self.root, text="Hold to Talk", state=tk.DISABLED)
        self.talk_btn.pack(pady=20)
        
        # Status label
        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
        
        print("[DEBUG] Widgets created", file=sys.stderr)
        
    def toggle_connection(self):
        print(f"[DEBUG] toggle_connection: connected={self.connected}", file=sys.stderr)
        if not self.connected:
            self.connect()
        else:
            self.disconnect()
            
    def connect(self):
        print("[DEBUG] connect() called", file=sys.stderr)
        self.status_var.set("Connecting...")
        self.connect_btn.state(['disabled'])
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self._ws_thread, daemon=True)
        self.ws_thread.start()
        
    def _ws_thread(self):
        print("[DEBUG] _ws_thread() started", file=sys.stderr)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_connect())
            self.loop.run_until_complete(self._async_message_handler())
        except Exception as e:
            print(f"[DEBUG] WebSocket thread error: {e}", file=sys.stderr)
            self.root.after(0, self.disconnect)
        finally:
            try:
                self.loop.close()
            except Exception as e:
                print(f"[DEBUG] Error closing WebSocket loop: {e}", file=sys.stderr)
                
    def _connect_websocket(self):
        # This method is no longer needed as connection is handled in _ws_thread
        pass
        
    def disconnect(self):
        print("[DEBUG] disconnect() called", file=sys.stderr)
        if self.ws:
            try:
                if self.loop and self.loop.is_running():
                    self.loop.call_soon_threadsafe(self._async_disconnect)
            except Exception as e:
                print(f"[DEBUG] Error during disconnect: {e}", file=sys.stderr)
        self.connected = False
        self.status_var.set("Disconnected")
        self.connect_btn.configure(text="Connect")
        self.talk_btn.state(['disabled'])
        self.connect_btn.state(['!disabled'])
        
    async def _async_disconnect(self):
        print("[DEBUG] _async_disconnect() called", file=sys.stderr)
        if self.conversation_id and self.ws:
            try:
                session_end = {
                    "type": "session.end",
                    "conversationId": self.conversation_id,
                    "reasonCode": "normal",
                    "reason": "User disconnected"
                }
                await self.ws.send(json.dumps(session_end))
            except Exception as e:
                print(f"[DEBUG] Error sending session end: {e}", file=sys.stderr)
            try:
                await self.ws.close()
            except Exception as e:
                print(f"[DEBUG] Error closing websocket: {e}", file=sys.stderr)
            

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioInterface(root)
    root.mainloop()
