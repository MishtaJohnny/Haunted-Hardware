# 👻 Haunted Hardware Exorcist

An AI-powered, real-time system diagnostic app built with **FastAPI**. **TabPFN** classifies your machine's state, **Gemma 3** narrates the haunting and answers your questions, and a public demo is hosted live on **Render** (with Gemma served through **OpenRouter**).

Built for the [Hacktoberfest Open-Source AI Challenge, Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26).

---

## 🌟 Overview

Task Manager shows numbers. It doesn't tell you what they mean. **Haunted Hardware Exorcist** turns machine telemetry into a haunting you can actually read:

1. **TabPFN Classifier:** reads live system vitals (`CPU`, `RAM`, `Swap Pressure`, `RAM Growth Rate`, `Top Process CPU`) and classifies the machine into 4 entity states, with a **possession %** showing how far it is from calm:
   * 🕯️ **Sanctified Core:** the system is calm and healthy.
   * 🕳️ **Abyssal Heap:** a memory leak thrashing swap.
   * ⚙️ **Clockwork Daemon:** a rogue process pinning the CPU.
   * 🌋 **Infernal Crucible:** heavy overall stress and overheating.
2. **Culprit Detection:** the busiest processes show up as **Top Suspects** with CPU, RAM, and open network connections. The likely culprit is blamed on the matching demon.
3. **Gemma 3 Exorcist:** an LLM that turns raw metrics into a short, vivid exorcism log and a 3-step "ritual" with practical fixes.
4. **Ask the Ghost:** a streaming chat where you can question the ghost about your machine, grounded in your live readings.
5. **Cloud Demo (Render):** a public web demo with simulated telemetry, so anyone can try it without installing anything.

---

## 💀 Why I Built This

My cousin installed a game from a sketchy website. Something came along with it and kept his laptop's CPU pinned for about a week, and he ended up with a burned-out motherboard.

Sustained heat is hard on hardware. It speeds up charge leakage in flash storage, which can corrupt data and create bad sectors, and cheap or un-heatsinked SSD controllers can overheat badly enough to fail. For that whole week the machine was showing him something was wrong, but Task Manager only said "100%". Nothing told him *"this isn't normal for you, and this is the process doing it."* Haunted Hardware is my attempt at that translation layer.

> **What it is and isn't:** it spots *unusual resource behavior* and names the process responsible. It can't prove something is malware, and it only helps while it's running.

---

## 🛠️ Key Features

* **Real-time Telemetry:** live CPU, RAM, swap pressure, RAM growth rate, and process metrics via `psutil`.
* **Three Modes:** **Live** reads your real machine, **Simulate** triggers any demon in one click, and **Manual Séance** gives you vertical sliders (hatched bar = calm baseline, solid bar = your setting) to inject any stress and watch TabPFN classify it. **Perform Exorcism** resets everything to calm.
* **💬 Ask the Ghost:** questions like "Why is my PC slow?" or "Who is the culprit?" stream in word by word. The ghost is told to admit when the telemetry can't answer.
* **📖 Session Ghost Story:** Gemma turns your session history into a 5-sentence ghost story.
* **Model Switcher:** with two or more models in Ollama, swap between them live and compare speed. Each log shows how long it took.
* **🔔 Alerts:** a browser notification (and an optional Discord message) when possession stays above 80% for 30 seconds in Live mode.
* **📸 Exorcism Report Cards:** one-click PNG with your peak possession, haunted %, most common demon, and history chart.
* **Séance History and Session Stats:** a chart of the last 20 readings, plus peak possession and most common entity.

---

## 🧠 Why TabPFN?

[TabPFN](https://github.com/PriorLabs/TabPFN) is an open-weight tabular foundation model. Instead of training a classifier from scratch, it is pre-trained on a huge range of synthetic tables and learns from your example rows **in context**, in a single forward pass.

* **No training loop or tuning:** the app hands TabPFN a few hundred example readings at startup and it's ready to classify.
* **Small-data friendly:** it works well on small tables, which suits a hand-built set of "calm vs. haunted" examples.
* **Multi-variable states:** it classifies combinations of CPU, RAM, swap, and growth rate together, instead of one threshold per number.
* **Calibrated probabilities:** the possession % comes straight from its predicted class probabilities.

> The example rows are **synthetic**: patterns generated for each state, not data from real failing machines.

## 🦙 Why Gemma?

Gemma is an open-weight language model, so you choose where it runs:

* **Locally through [Ollama](https://ollama.com):** process names and system stats never leave your machine, it works offline, and there's no API bill. Swap models with `HAUNT_MODEL` or the dropdown.
* **On the Render demo through [OpenRouter](https://openrouter.ai):** a free small server can't host a language model, so the hosted demo calls **Gemma 3 4B** (`google/gemma-3-4b-it`) via OpenRouter. The demo only handles simulated readings and your chat questions.

---

## 🏗️ How It Works

```
Browser dashboard ──every 3s──▶ FastAPI server (app.py)
                                   │
                                   ├─ psutil   → CPU, RAM, swap, RAM growth, top processes
                                   ├─ TabPFN   → which demon + possession %
                                   ├─ rules    → which process to blame
                                   └─ Gemma    → exorcism log, ritual, chat, story
                                      (Ollama locally, OpenRouter on Render)
```

The whole dashboard (HTML, CSS, JavaScript) is served straight from `app.py`, so there's no front-end build step.

### 🔍 How to Verify What Actually Ran

The footer under the ghost log names the exact engine:

| Footer says | Meaning |
|---|---|
| `TabPFN (Local Engine)` | TabPFN running on your machine |
| `TabPFN (Cloud Engine)` | TabPFN through Prior Labs' hosted API |
| `RandomForest (Fallback)` | TabPFN wasn't available, so a scikit-learn model was used |
| `Gemma 3 4B (OpenRouter)` or your Ollama model name, plus seconds | the Gemma model that wrote the text |
| `pre-written` | Gemma wasn't reachable, so a built-in line was shown |

> **Privacy note:** with `TABPFN_TOKEN` set and `tabpfn-client` installed, TabPFN uses the **cloud** engine first, and with `OPENROUTER_API_KEY` set, Gemma requests go to OpenRouter, even on your own PC. For the fully private setup, install `requirements.txt` (local TabPFN) and set no cloud keys.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) with a Gemma model, for local AI narration and chat
* A free [TabPFN token](https://docs.priorlabs.ai/quickstart) (accept the non-commercial license at platform.priorlabs.ai)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MishtaJohnny/Haunted-Hardware.git
   cd Haunted-Hardware
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull a Gemma model:**
   ```bash
   ollama pull gemma3:4b
   ```
   Check [ollama.com/library](https://ollama.com/library) for the newest Gemma tag and set it with `HAUNT_MODEL`.

4. **Add your TabPFN token.** Copy `.env.example` to `.env` and fill it in:
   ```
   TABPFN_TOKEN=your_token_here
   ```
   `.env` is git-ignored. Never commit your keys.

5. **Run the app:**
   ```bash
   uvicorn app:app --port 8000
   ```
   Then open **http://localhost:8000** and click **Live (this PC)**.

### 🎮 Try It

1. Run a CPU burner in another terminal: `python -c "while True: pass"`
2. Watch a demon appear, then check **Top Suspects** for the culprit.
3. Open **Manual sliders**, push a bar up, then press **🕯️ Perform exorcism**.
4. Ask the ghost a question, or click **📖 Tell my session's ghost story**.
5. Download your **📸 Haunting report**.

### ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TABPFN_TOKEN` | none | Prior Labs token (also accepts `TABPFN_API_KEY`) |
| `HAUNT_MODEL` | `gemma3:4b` | Ollama model to use locally |
| `OLLAMA_URL` | `http://localhost:11434` | Point at Ollama running on another machine |
| `DEMO_MODE` | off | `1` = simulated data only (for hosting) |
| `OPENROUTER_API_KEY` | none | Serves Gemma 3 4B through OpenRouter (used by the Render demo) |
| `DISCORD_WEBHOOK_URL` | none | Optional: also post alerts to Discord |

---

## ☁️ Cloud Demo on Render

A website can't read your computer's processes, so the public demo runs with **`DEMO_MODE=1`**:

* **Simulated readings** and manual sliders instead of live telemetry.
* **TabPFN** through Prior Labs' hosted API (the footer shows `TabPFN (Cloud Engine)`), because TabPFN's PyTorch install is too heavy for a small free-tier server. If the token or API isn't available, it falls back to a RandomForest and says so.
* **Gemma 3 4B** through **OpenRouter**, for logs, rituals, chat, and stories. If OpenRouter is unreachable, it shows pre-written lines.
* The **model switcher** is local-only: with one cloud model there's nothing to switch.

**Render settings:**
* Build command: `pip install -r requirements-demo.txt`
* Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
* Environment variables: `DEMO_MODE=1`, `TABPFN_TOKEN`, `OPENROUTER_API_KEY`

---

## 📁 Project Structure

```
app.py                  FastAPI server and the whole dashboard (HTML/CSS/JS)
requirements.txt        Local install (includes TabPFN)
requirements-demo.txt   Lightweight install for hosting
.env.example            Template for secrets
```

---

## ⚠️ Limitations

* **Not antivirus.** It flags *unusual resource behavior*. Quiet malware won't trigger it, and a legitimate heavy app (a game, a build) will.
* **It only helps while it's running.** A week-long burn like my cousin's would need it running in the background. A background service is the next thing I'd build.
* **The example data is synthetic.** TabPFN learns from patterns generated for each state, not from real failing machines.
* **Culprit blame is rule-based.** TabPFN classifies the whole machine; simple thresholds (at least half the CPU, or 20% of RAM) pick the likely process.
* **Temperature** is shown where your OS exposes it (mostly Linux). It isn't used by the classifier.
* **Network connection counts** can be unavailable for some processes without admin rights.
* **Small models can be wrong.** The ghost only knows the telemetry it's given.
* **The hosted demo is simulated.** It uses made-up readings and cloud-served models. The local version is the private one.

---

## 🏆 Prize Categories

* **Best Use of TabPFN:** classifies machine state from tabular telemetry (`build_model()` in `app.py`).
* **Best Use of Gemma:** logs, rituals, chat, and stories, run locally through Ollama and on the demo through OpenRouter.
* **Best Use of Render:** hosts the public demo.

## 🗓️ Challenge Note

Built during the Hacktoberfest Open-Source AI Challenge Week 4 window (Oct 26 to 31, 2026).

## 🙏 Credits

Built with [TabPFN](https://github.com/PriorLabs/TabPFN), [Gemma](https://ai.google.dev/gemma), [Ollama](https://ollama.com), [OpenRouter](https://openrouter.ai), [FastAPI](https://fastapi.tiangolo.com), and [psutil](https://github.com/giampaolo/psutil). TabPFN and Gemma are used under their own license terms.
