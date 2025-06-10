import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")

openai.api_key = OPENAI_API_KEY

# List of phrases to synthesize
phrases = [
    "Uh, sorry, can you hold on a sec? ... Okay, I lost my card.",
    "Wait, can you tell me when my last payment was?",
    "Actually, I moved. Can you send it to 456 Oak Ave?"
]

VOICE = "ash"
OUTPUT_DIR = "output_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, phrase in enumerate(phrases, 1):
    print(f"Synthesizing phrase {idx}: {phrase}")
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=VOICE,
        input=phrase,
        response_format="wav"
    )
    out_path = os.path.join(OUTPUT_DIR, f"replacement_card_{idx}.wav")
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"Saved: {out_path}") 