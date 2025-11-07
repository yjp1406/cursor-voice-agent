# Cursor Voice Agent Clone

Local voice-agent prototype using Python `venv`, SpeechRecognition, pyttsx3 and a placeholder for LLM/RAG.

## Quickstart
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. create `.env` with `OPENAI_API_KEY=...` if needed
5. python src/agent.py

## Notes
- Do not commit `venv/` unless you understand the consequences.
- On Linux, install system dependencies for audio (portaudio).
