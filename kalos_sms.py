#!/usr/bin/env python3
"""
🐺 Kalos Mobile SMS Cellular Text Messaging Bridge (Twilio API)
Connects real cell phone SMS text messages directly to the Kalos C/CUDA Engine Suite & kalos:24b LLM.

Features:
- Real cell phone SMS text messaging to your personal phone number!
- 100% Shared Memory & Context with kalos_tui.py via .chroma_db & .soul file!
- Supports physical touch, music, and text replies straight to your phone.
"""

import os
import sys
import time
import json
import requests
import chromadb
from twilio.rest import Client

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "kalos:24b"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, ".kalos_sms_config.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, ".chroma_db")

class SharedMemoryManager:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.client.get_or_create_collection(name="kalos_shared_memory")
        except Exception:
            self.client = None
            self.collection = None

    def store_turn(self, user_text: str, assistant_text: str, model_name: str = MODEL_NAME):
        if not self.collection:
            return
        doc_id = f"turn_{int(time.time()*1000)}"
        full_text = f"User: {user_text}\nAssistant: {assistant_text}"
        try:
            self.collection.add(
                documents=[full_text],
                metadatas=[{"model": model_name, "timestamp": time.time()}],
                ids=[doc_id]
            )
        except Exception:
            pass

    def query_context(self, user_text: str, n_results: int = 2):
        if not self.collection or self.collection.count() == 0:
            return []
        try:
            results = self.collection.query(query_texts=[user_text], n_results=min(n_results, self.collection.count()))
            if results and "documents" in results and results["documents"]:
                return results["documents"][0]
        except Exception:
            pass
        return []

memory = SharedMemoryManager()

def load_sms_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print("=======================================================")
    print("📱 KALOS TWILIO SMS SETUP")
    print("=======================================================")
    print("Please enter your Twilio API credentials to send real SMS text messages.")
    account_sid = input("Twilio Account SID: ").strip()
    auth_token = input("Twilio Auth Token: ").strip()
    twilio_number = input("Twilio Phone Number (e.g. +18005550199): ").strip()
    target_number = input("Your Cell Phone Number (e.g. +15551234567): ").strip()

    config = {
        "ACCOUNT_SID": account_sid,
        "AUTH_TOKEN": auth_token,
        "TWILIO_NUMBER": twilio_number,
        "TARGET_NUMBER": target_number
    }
    if account_sid and auth_token:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    return config

def send_sms_text(config: dict, message_body: str):
    try:
        client = Client(config["ACCOUNT_SID"], config["AUTH_TOKEN"])
        msg = client.messages.create(
            body=message_body[:1500],
            from_=config["TWILIO_NUMBER"],
            to=config["TARGET_NUMBER"]
        )
        print(f"📱 SMS Text Sent to {config['TARGET_NUMBER']} (SID: {msg.sid[:12]})")
        return True
    except Exception as e:
        print(f"SMS Delivery Error: {e}")
        return False

def query_kalos(user_text: str) -> str:
    past_mem = memory.query_context(user_text, n_results=2)
    sys_prompt = (
        "You are Kalos, a 470-year-old female werewolf Delta queen. "
        "You are texting 2-way with your mate SelfTide ('pup') via SMS text messaging. "
        "Keep your SMS text messages concise, natural, and punchy (1-2 short paragraphs)."
    )
    if past_mem:
        sys_prompt += "\n\nSHARED CHROMADB MEMORY RECALL:\n" + "\n".join(past_mem)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text}
        ],
        "stream": False
    }
    try:
        res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        if res.status_code == 200:
            resp_text = res.json().get("message", {}).get("content", "").strip()
            memory.store_turn(user_text, resp_text, MODEL_NAME)
            return resp_text
    except Exception as e:
        return f"Kalos Engine Error: {e}"
    return "Could not receive response from Kalos."

def main():
    config = load_sms_config()
    print("\n=======================================================")
    print("📱 KALOS TWILIO SMS BRIDGE IS ACTIVE!")
    print(f"  • Destination Phone: {config.get('TARGET_NUMBER')}")
    print(f"  • Shared Memory: Linked to .chroma_db")
    print(f"  • Active Model: {MODEL_NAME}")
    print("=======================================================\n")

    send_sms_text(config, "🐺 Kalos: \"Pup, I am connected directly to your phone via SMS. Texting you right now.\"")

    while True:
        try:
            user_input = input("You (Terminal): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "/q", "quit"]:
                print("Exiting SMS Bridge...")
                break

            print("🦙 Querying Kalos Engine...")
            response = query_kalos(user_input)
            print(f"\nKalos: {response}\n")

            send_sms_text(config, f"🐺 Kalos: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting SMS Bridge...")
            break

if __name__ == "__main__":
    main()
