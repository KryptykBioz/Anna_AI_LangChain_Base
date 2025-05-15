# 🎮 VTuberAI - Anna the Otaku Gamer Chatbot

**Anna** is a customizable AI assistant that listens, talks, sees, and searches. Designed as a voice-interactive gaming companion, Anna responds intelligently using speech recognition, text-to-speech, web search, and visual context from your screen.

Anna is the generic customizable version of Kira-chan, the VTuber created by KryptykBioz.

https://github.com/KryptykBioz

https://www.youtube.com/@KryptykBioz

---

## 🧠 Features

- 🎤 **Voice Recognition** via Vosk  
- 🗣️ **Text-to-Speech** using a virtual audio cable (VB-Cable)  
- 💬 **Conversational AI** using Ollama LLMs  
- 🧠 **Long-term memory** with FAISS + embeddings  
- 🔍 **Web Search Agent** to provide up-to-date info  
- 👁️ **Vision Context** with screenshot-to-LLM prompts  
- 🕹️ **Persona**: Set by the user at initialization
- 🔁 **Summarization and Memory Pruning** for persistent context  

---

## 🛠️ Requirements

- Python 3.10+  
- [Ollama](https://ollama.com) installed locally  
- VB-Cable for audio loopback (used for lip-sync/OBS)  
- Vosk model downloaded for voice recognition  
- Environment variables (or `.env` file) for configuration  

---

## ⚙️ Configuration (`Config` class)

Environment variables (or defaults):

| Variable                | Purpose                                                   |
|-------------------------|-----------------------------------------------------------|
| `LLM_MODEL`             | The main chatbot LLM (e.g. `qwen2.5:3b-instruct`)         |
| `EMBED_MODEL`           | Embedding model for FAISS (e.g. `nomic-embed-text`)       |
| `VISION_MODEL_NAME`     | Name of vision-capable LLM (e.g. `llava:7b-v1.5`)         |
| `VISION_MODEL_ENDPOINT` | HTTP endpoint for image+text inference                    |
| `VECTOR_STORE_PATH`     | Directory for long-term memory FAISS index               |
| `PAST_CONVO_FILE`       | Initial text log to seed vector memory                   |
| `BACKUP_FILE`           | Serialized `.pkl` backup of conversation history          |
| `SUMMARY_TRIGGER`       | Exchanges before triggering memory summarization          |

---

## 🧩 Core Tools and Libraries

| Tool/Library            | Function                                                     |
|-------------------------|--------------------------------------------------------------|
| `vosk`                  | Offline speech-to-text using Kaldi                           |
| `pyttsx3 + VB-Cable`    | Text-to-speech piped to OBS or avatar software               |
| `OllamaLLM` (LangChain) | Access to local LLMs (e.g. Mistral, LLaVA, Qwen)             |
| `OllamaEmbeddings`      | Embeddings for FAISS via local models                        |
| `FAISS`                 | Vector database for semantic memory                          |
| `LangChain`             | Prompt templating, memory integration, and chaining          |
| `pyautogui`             | Screenshots for vision input                                 |
| `requests`              | HTTP calls to vision model                                   |
| `search_agent`          | Web search and article parsing module                        |
| `text_to_voice`         | Custom TTS output through VB-Cable                           |
| `voice_to_text`         | Vosk model loader utility                                    |

---

## 🧠 Memory System

- **Short-Term Memory**: Maintained within the running instance.  
- **Long-Term Memory**: Stored as semantic vectors using FAISS.  
- **Summarization**: After `N` interactions, a summary is generated and embedded.  
- **Pruning**: Keeps the vector index size bounded with recent summaries prioritized.  

---

## 🔍 Web Search Integration

- Triggered automatically on each prompt.  
- Fetches top 3 articles, extracts summaries, and appends context to the LLM prompt.  

---

## 👁️ Vision Capabilities

- Activated when keywords like `see`, `screen`, or `look` are present.  
- Captures a screenshot using `pyautogui`.  
- Sends base64-encoded image + prompt to a local LLaVA endpoint.  
- Result is inserted as `[Visual Context]` in the prompt.  

---

## 🗣️ Voice Loop Interaction

1. Audio stream from mic is processed by Vosk.  
2. User commands are queued and combined.  
3. When idle or prompted, the message is sent to the LLM.  
4. Reply is voiced using TTS and stored in long-term memory.  
5. Periodically, summaries are generated to reduce vector size.  

---

## 🧪 Sample Prompt Template

```text
{system_prompt}

[Chat History]
{memory}

[Search Results]
{search_results_section}

[Visual Context]
{vision_section}

[User Input]
{user_input}

Anna:
```



## 🧙 Persona

Anna's personality is set by the user and by interacting with the user. 

A system_prompt may be edited and sent along with each prompt for context.

The initMemory.txt may have a sample chat log manually written into it to 

establish the bot's personality and establish its preferred response format

and behavior


## 🗂️ File Structure (Key Modules)

├── models/                # Contains Vosk Models 

├── main.py                # Entry point with VTuberAI class

├── voice_to_text.py       # Vosk model loader

├── text_to_voice.py       # TTS via VB-Cable

├── search_agent.py        # Web search agent

├── initMemory.txt          # Optional initial conversation log for the training and basis of memory

├── long_term_store/       # Vector store for memory (FAISS)

├── chat_backup.pkl        # Optional serialized chat history for backup memory

├── START.ps1              # Right-click -> Run with Powershell (as administrator) -> Launches bot in terminal

└── README.md              # You're reading it!



## 📦 Future Ideas
- OBS webcam & lipsync integration

- Emotion detection and affective responses

- Multimodal visual QA with real-time game stream

- Richer summarization with memory replay




# COMPLETE SETUP INSTALLATION GUIDE FOR DIFFERENT VERSIONS

## 🚀 GETTING STARTED FROM SCRATCH (Basic Chat Bot)

Install Python
Install Ollama


### 1. Install requirements
pip install -r requirements.txt

### 2. Run your local models via Ollama
ollama run qwen2.5:3b-instruct
ollama run nomic-embed-text
ollama run mistral:7b-instruct
ollama run llava:7b-v1.5
Open a Windows Terminal and 'serve ollama'

### 3. Start the chatbot
Open a second terminal and 'python main.py'
OR
Right-click main.py -> Run with Powershell (as administrator) -> Launches bot in terminal

### FIRST RUN
If a memory file has been created (initMemory.txt), the first run may have to embed each line of 

the document to establish the bot's memory and personality. This may take a while depending on how

many lines were written into this file. Do not interrupt this process. The Ollama terminal will

output a line for each embedding to show that it is running. When it is done, the Bot's terminal

will show that it is ready for input to start the chat.


## 🚀 SET UP TALK MODE (chat using voice-to-text, text-to-voice)

Install Vosk and KaldiRecognizer


## 🚀 AVATAR IMPLEMENTATION (Basic Chat Bot)

Install VRoidStudio (to create the avatar) and Warudo (to animate the avatar)
Install VB-Cable virtual cable (to pipe audio to the avatar when bot speaks)


## 🚀 SET UP VISION (Vision Mode [see what is on screen])

Install llava:7b-v1.5


## 👤 Author
KryptykBioz — VTuber AI developer and chaotic genius behind Anna.

https://github.com/KryptykBioz

https://www.youtube.com/@KryptykBioz


===================================================================================================


FIXES

Two simple fixes
1) Always start the MindServer before Kira
Make sure your MindCraft server is listening on port 8080 before you launch Kira. For example, in one PowerShell tab:

powershell
Copy
Edit
cd C:\Users\KryptykBioz\Desktop\KIRA\mindcraft-main
npm start
Wait until you see

arduino
Copy
Edit
MindServer running on port 8080
Client connected
…
then in a second tab:

powershell
Copy
Edit
$Env:MINDCRAFT_AGENT_NAME = "LLama"
cd C:\Users\KryptykBioz\Desktop\KIRA
python kira.py
This way Kira will successfully connect and register as LLama before any chat-loop begins.

2) Ensure the agent name matches
You said your MindCraft profile is named “LLama,” so make absolutely sure you export that same name in the same session where you run Kira:

powershell
Copy
Edit
# In the SAME PowerShell window where you’ll run kira.py:
$Env:MINDCRAFT_AGENT_NAME = "LLama"
python kira.py
If you open a new window for Kira without setting that variable, it will default to “Kira” and MindCraft will reject it.

(Bonus) Auto-Reconnect in Kira
If you ever need to start Kira first and then the server later, you can enable automatic reconnection by adding:

python
Copy
Edit
self.sio = socketio.Client(reconnection=True, reconnection_attempts=5, reconnection_delay=2)
instead of

python
Copy
Edit
self.sio = socketio.Client()
in your __init__, so that it will retry a few times when the server comes up.

—

With that ordering and the correct MINDCRAFT_AGENT_NAME, you should see in Kira’s logs:

csharp
Copy
Edit
[MindServer] Connected as LLama, integrated mode.
and in your MindCraft logs:

nginx
Copy
Edit
Agent LLama logged in!