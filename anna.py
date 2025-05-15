import os
import pickle
import threading
import queue
import base64
import json
import time
import asyncio
from io import BytesIO
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Callable
import random

import pyautogui
import sounddevice as sd
from vosk import KaldiRecognizer
from colorama import Fore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from text_to_voice import speak_through_vbcable
from voice_to_text import load_vosk_model
from query import maybe_fetch_articles
from training_tags import training_tags
from SYS_MSG import system_prompt
# from training import training_loop
# from memory_methods import 

# --- Configuration ---
PROMPT_TIMEOUT = 120
VISION_KEYWORDS = ["screen", "image", "see", "look", "game"]
MAX_VECTORS = 1000
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "500"))
MIND_SERVER_HOST = os.getenv("MINDSERVER_HOST", "localhost")
MIND_SERVER_PORT = os.getenv("MINDSERVER_PORT", "8080")
AGENT_NAME = os.getenv("MINDCRAFT_AGENT_NAME", "Kira")



@dataclass
class Config:
    llm_model: str = os.getenv("LLM_MODEL", "gemma3:4b-it-qat") #Replace with you Ollama model
    # llm_model: str = os.getenv("LLM_MODEL", "llama3.2:latest")
    vision_endpoint: str = os.getenv(
        "VISION_MODEL_ENDPOINT", "http://localhost:11434/api/generate"
    )
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text:latest") #Replace with you Ollama embedding model
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./memory")
    past_convo_file: str = os.getenv("PAST_CONVO_FILE", "./memory_base/pastconvo.json")
    prompt_history_file: str = os.getenv(
        "PROMPT_HISTORY_FILE", "./memory/prompt_history.json"
    )
    backup_file: str = os.getenv("BACKUP_FILE", "./memory_backup/chat_backup.pkl")
    summary_trigger: int = int(os.getenv("SUMMARY_TRIGGER", "200"))
    system_prompt: str = system_prompt
    samplerate: int = 16000
    

config = Config()

# Create a dedicated asyncio loop in a separate thread
async_loop = asyncio.new_event_loop()
threading.Thread(target=async_loop.run_forever, daemon=True).start()

def get_random_items(arr):
    if len(arr) < 10:
        raise ValueError("Input array must have at least 10 elements.")
    return random.sample(arr, 10)

class VTuberAI:
    def __init__(self):

        self.sio = None
        self.mindserver_connected = False

                # Queues and audio
        self.raw_queue = queue.Queue()
        self.text_queue = queue.Queue()
        # Histories
        self.prompt_history = self._load_prompt_history()
        self.past_convo = self._load_past_convo()
        # Init
        self._init_audio()
        self._init_models()
        self.msg_buffer: List[str] = []
        self.history: List[dict] = []
        self.speaking_thread: Optional[threading.Thread] = None
        self.processing = False
        self.last_interaction = time.time()

    def _should_embed(self, text: str) -> bool:
        return len(text.split()) > 6

    def _token_count(self, text: str) -> int:
        return len(text.split())

    def _prune_lines_to_tokens(self, lines: List[str], max_tokens: int) -> List[str]:
        tokens = sum(self._token_count(l) for l in lines)
        while tokens > max_tokens and lines:
            tokens -= self._token_count(lines[0])
            lines = lines[1:]
        return lines

    def _init_audio(self):
        sd.default.device = (1, None)
        self.vosk_model = load_vosk_model()
        self._start_vosk_stream()

    def _start_vosk_stream(self):
        def recognition_worker():
            rec = KaldiRecognizer(self.vosk_model, config.samplerate)
            while True:
                data = self.raw_queue.get()
                if data == b"__EXIT__": break
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "").strip()
                    if len(text) >= 5 and "kira" not in text.lower():
                        self.text_queue.put(text)
        threading.Thread(target=recognition_worker, daemon=True).start()

        def audio_callback(indata, frames, time_info, status):
            if status: print(Fore.RED + f"[Audio status]: {status}" + Fore.RESET)
            self.raw_queue.put(bytes(indata))

        self.stream = sd.RawInputStream(
            samplerate=config.samplerate,
            blocksize=4096,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )
        self.stream.start()
        print(Fore.CYAN + "[Listener] Audio stream started." + Fore.RESET)

    def _load_prompt_history(self) -> List[dict]:
        try:
            data = json.load(open(config.prompt_history_file, 'r', encoding='utf-8'))
            return data[-100:]
        except:
            return []

    def _save_prompt_history(self):
        os.makedirs(os.path.dirname(config.prompt_history_file), exist_ok=True)
        with open(config.prompt_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_history[-100:], f, ensure_ascii=False, indent=2)

    def _load_past_convo(self) -> List[dict]:
        if os.path.exists(config.past_convo_file):
            try:
                data = json.load(open(config.past_convo_file, 'r', encoding='utf-8'))
                return data[-1:]
            except:
                return []
        return []

    def _save_past_convo(self):
        os.makedirs(os.path.dirname(config.past_convo_file), exist_ok=True)
        with open(config.past_convo_file, 'w', encoding='utf-8') as f:
            json.dump(self.past_convo[-25:], f, ensure_ascii=False, indent=2)

    def _update_past_convo(self, prompt: str, response: str):
        self.past_convo.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        self._save_past_convo()

    def _compress_long_term_memory(self):
        docs = list(self.vector_store.docstore._dict.values())
        if len(docs) < 200:
            return
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("timestamp", ""))
        # Summarize oldest 100 docs in 4 chunks
        chunks = [docs_sorted[i*25:(i+1)*25] for i in range(4)]
        summaries = []
        for chunk in chunks:
            texts = [d.page_content for d in chunk]
            summary_text = self._summarize_text(texts)
            summaries.append(Document(
                page_content=f"[Memory Summary] {summary_text}",
                metadata={"timestamp": datetime.now().isoformat(), "type": "memory_summary"}
            ))
        # Remaining recent docs
        remaining = docs_sorted[100:]
        # Rebuild store: summaries + recent
        new_docs = summaries + remaining
        # Trim to 100 most recent
        new_docs = new_docs[-100:]
        new_store = FAISS.from_documents(new_docs, self.embeddings)
        new_store.save_local(config.vector_store_path)
        self.vector_store = new_store

    def _summarize_text(self, texts: List[str]) -> str:
        prompt = (
            "Extract and summarize only the following from the conversation:\n"
            "- Main topics and themes\n"
            "- Participant names\n"
            "- Personal details (preferences)\n"
            "- Important dates (birthdays)\n"
            "Provide a concise paragraph summary as a diary entry from the perspective of the assistant Anna.\n\n"
            + "\n".join(texts)
        )
        return self.llm(prompt).strip()

    def _init_models(self):
            # Initialize the Ollama LLM with desired settings
            self.llm = OllamaLLM(
                model=config.llm_model,
                temperature=0.7,
                max_new_tokens=250
            )

            # Define the system prompt and template variables
            prompt_tmpl = PromptTemplate(
                input_variables=[
                    "system_prompt",
                    "embeddings",
                    "search_results",
                    "history",
                    "screenshot_image",
                    "user_input",
                ],
                template=(
                    "[[SYSTEM]]\n"
                    "{system_prompt}\n\n"
                    "[[EMBEDDINGS]]\n"
                    "{embeddings}\n\n"
                    "[[SEARCH_RESULTS]]\n"
                    "{search_results}\n\n"
                    "[[HISTORY]]\n"
                    "{history}\n\n"
                    "[[SCREENSHOT_IMAGE]]\n"
                    "{screenshot_image}\n\n"
                    "[[USER]]\n"
                    "{user_input}\n"
                    "Response:"  # prompt model to output fresh response without speaker label
                ),
            )

            # Build the LLM chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=prompt_tmpl
            )

            # Initialize embeddings and vector store for long-term memory
            self.embeddings = OllamaEmbeddings(model=config.embed_model)
            self.vector_store = self._load_or_create_vector_store()

        
    def _load_or_create_vector_store(self) -> FAISS:
        path = config.vector_store_path
        os.makedirs(path, exist_ok=True)
        flag = os.path.join(path, ".initialized")
        if os.path.exists(flag):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        store = FAISS.from_documents([], self.embeddings)
        store.save_local(path)
        with open(flag, 'w') as f:
            f.write(datetime.now().isoformat())
        return store

    def _retrieve_memory(self, query: str, k: int = 6) -> str:
        hits = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join(f"- {d.page_content}" for d in hits)
    
    def print_long_term_memory(self):
        """
        Print all documents currently stored in the FAISS vector store,
        showing page_content and timestamp metadata.
        """
        docs = list(self.vector_store.docstore._dict.values())
        if not docs:
            print("[Memory] No long-term memory entries.")
            return
        print(f"[Memory] Total entries: {len(docs)}")
        # Sort by timestamp for readability
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get("timestamp", ""))
        for idx, doc in enumerate(docs_sorted, 1):
            content = doc.page_content
            ts = doc.metadata.get("timestamp", "?")
            tag = doc.metadata.get("type", "raw")
            print(f"{idx}. [{tag} @ {ts}]: {content}")

    # def training_loop():
    #     print(Fore.YELLOW + "[Training Mode] Starting training loop..." + Fore.RESET)
    #     count = 0
    #     try:
    #         while count < 100:
    #             tag = random.choice(training_tags)
    #             query = f"search League of Legends champion {tag}"
    #             print(Fore.CYAN + f"[Training] Querying: {query}" + Fore.RESET)

    #             search_result = maybe_fetch_articles(query)
    #             search_str = f"[Search]\n{search_result}\n\n" if search_result else ""

    #             prompt_text = f"Run an internet search on the League of Legends champion {tag} and summarize the results."
    #             prompt = {
    #                 "system_prompt": config.system_prompt,
    #                 "history_section": "",
    #                 "memory": self._retrieve_memory(tag),
    #                 "search_section": search_str,
    #                 "vision_section": "",
    #                 "user_input": prompt_text,
    #             }

    #             reply = self.chain.run(**prompt)
    #             reply = ''.join(c for c in reply if c not in '#*`>\n')
    #             ts = datetime.now().isoformat()

    #             self.prompt_history.append({"prompt": prompt_text, "response": reply, "timestamp": ts})
    #             self._save_prompt_history()

    #             docs = []
    #             for role, content in [("Bioz", prompt_text), ("Kira", reply)]:
    #                 entry = f"{role}: {content}"
    #                 if self._should_embed(entry):
    #                     docs.append(Document(page_content=entry, metadata={"timestamp": ts}))
    #             if docs:
    #                 self.vector_store.add_documents(docs)
    #                 self._prune_and_save()

    #             print(Fore.MAGENTA + f"[Training Response]: {reply}" + Fore.RESET)
    #             count += 1
    #             time.sleep(1)
    #     except KeyboardInterrupt:
    #         print(Fore.RED + "\n[Training Mode] Interrupted by user." + Fore.RESET)

    async def _process_prompt_async(self, text: str, mode: str) -> str:
        if not text.strip():
            return ""
        ctx = self._prepare_context(text, mode)
        reply = await self.chain.arun(**ctx)
        reply = ''.join(c for c in reply if c not in '#*`>\n')
        ts = datetime.now().isoformat()

        # Update histories
        self.prompt_history.append({"prompt": text, "response": reply, "timestamp": ts})
        self._save_prompt_history()
        self._update_past_convo(text, reply)

        # Embed raw entries for short-term memory
        docs = []
        for role, content in [("User", text), ("Kira", reply)]:
            entry = f"[{role}] {content}"
            if len(entry.split()) > 6:
                docs.append(Document(page_content=entry, metadata={"timestamp": ts}))
        if docs:
            self.vector_store.add_documents(docs)
            self.vector_store.save_local(config.vector_store_path)

        # Compress and embed summaries when threshold reached
        self._compress_long_term_memory()

        return reply

    def _prepare_context(self, text: str, mode: str) -> dict:
        # Short-term history: last 50 turns with explicit speaker tags
        history_lines = []
        for pair in self.prompt_history:
            history_lines.append(f"User: {pair['prompt']}")
            history_lines.append(f"Kira: {pair['response']}")
        pruned = history_lines[-50:]

        # Long-term memory: top-5 similar summaries
        docs_and_scores = self.vector_store.similarity_search_with_score(text, k=5)
        embeddings_section = "[\n" + \
            ",\n".join(
                f"  {{ \"id\": \"{doc.metadata.get('id', '')}\", \"summary\": \"{doc.page_content}\", \"score\": {round(score, 2)} }}"
                for doc, score in docs_and_scores
            ) + "\n]"

        # Search results: optional web lookup
        raw_search = maybe_fetch_articles(text)
        if raw_search:
            search_items = raw_search[:3]
            search_results = "[\n" + \
                ",\n".join(
                    f"  {{ \"title\": \"{item.title}\", \"snippet\": \"{item.snippet}\", \"url\": \"{item.url}\" }}"
                    for item in search_items
                ) + "\n]"
        else:
            search_results = "[]"

        # Vision: include screenshot reference if in vision mode
        screenshot_image = ""
        if mode in ("vision", "game") and any(k in text.lower() for k in ["screen", "image", "see", "look", "game"]):
            screenshot_image = self._capture_screenshot()

        # Add repetition guard to system prompt
        repetition_guard = (
            "\n\n# Instructions for assistant:\n"
            "- Do not repeat the same content more than once.\n"
            "- Generate a fresh response based on the latest user input.\n"
            "- Do not echo history.\n"
        )

        return {
            "system_prompt": config.system_prompt + repetition_guard,
            "embeddings": embeddings_section,
            "search_results": search_results,
            "history": "\n".join(pruned),
            "screenshot_image": screenshot_image,
            "user_input": text,
        }

    def _capture_screenshot(self) -> str:
        print(Fore.YELLOW + "Taking screenshot" + Fore.RESET)
        buf = BytesIO()
        pyautogui.screenshot().save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()


    def _interaction_loop(self, get_input: Callable[[], Optional[str]], mode: str):
        print(Fore.LIGHTYELLOW_EX + f"[{mode.upper()} MODE] Listening..." + Fore.RESET)
        while True:
            if (self.speaking_thread and self.speaking_thread.is_alive()) or self.processing:
                time.sleep(0.1)
                continue
            texts = []
            if mode == "text":
                inp = get_input()
                if inp is None:
                    break
                texts = [inp]
            else:
                while not self.text_queue.empty():
                    texts.append(self.text_queue.get())
            if not texts and time.time() - self.last_interaction < PROMPT_TIMEOUT:
                time.sleep(0.1)
                continue
            user_text = " ".join(texts) or "Ask me something."
            self.last_interaction = time.time()
            if user_text.lower() == "exit":
                break
            if user_text.lower() == "/memory":
                self.print_long_term_memory()
                continue
            print(Fore.GREEN + f"Bioz: {user_text}" + Fore.RESET)
            future = asyncio.run_coroutine_threadsafe(
                self._process_prompt_async(user_text, mode), async_loop
            )
            reply = future.result()
            print(Fore.MAGENTA + f"Kira: {reply}" + Fore.RESET)
            if mode != "text" and len(reply) < 600:
                t = threading.Thread(
                    target=speak_through_vbcable, args=(reply,), daemon=True
                )
                t.start()
                self.speaking_thread = t
            self.history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ])
            self.msg_buffer.append(f"Bioz: {user_text}\nKira: {reply}")
            if len(self.msg_buffer) >= config.summary_trigger:
                print(Fore.YELLOW + "[Memory] Summarizing buffer..." + Fore.RESET)
                self._summarize_buffer()

    def run(self):
        print("[INFO] Starting VTuber AI. Type 'exit' to quit.")
        mode = input(Fore.GREEN + "Mode (text/talk/vision/game/minecraft/train): " + Fore.RESET).strip().lower()
        valid = ("text", "talk", "vision", "game", "minecraft", "train")
        while mode not in valid:
            mode = input("Invalid. Choose from text/talk/vision/game/minecraft/train: ").strip().lower()
        if mode == "train":
            # training_loop()
            return
        input_getter = input if mode == "text" else lambda: None
        try:
            self._interaction_loop(input_getter, mode)
        finally:
            self.raw_queue.put(b"__EXIT__")
            self.stream.stop()
            if self.sio:
                self.sio.disconnect()

if __name__ == "__main__":
    VTuberAI().run()
