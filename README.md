# Google Classroom Checker

A teacher tool that reads assignments straight from Google Classroom and checks student work
with a **local LLM** (Qwen on [Ollama](https://ollama.com)). Student work stays on your
computer: it goes from Google to your machine and is never sent to a cloud AI service.

## What it does

**Overview (no LLM needed)**
- A grid of students × assignments: turned in, late, missing, graded, returned
- Per-assignment list with turn-in time, grade and attachments
- Reads submitted Google Docs, Slides, Sheets, text files, `.py`/`.js` files and Colab/Jupyter notebooks
- CSV export of the whole course

**Checks (per assignment, run from the web page or the CLI)**

| Check | How it works |
|---|---|
| **Copying between students** | Exact wording (overlapping 5-word sequences), meaning (embeddings, so paraphrasing is caught), and code with renamed variables (identifier-normalised tokens). The most suspicious pairs are then reviewed by the LLM, which lists its evidence. |
| **Number answers** | You paste an answer key (`1: 42`, `2: 3.14 ± 0.01`, `3: 25%`, `4: 1/3 ± 1%`). Answers written like `Q1: …`, `1) …` or `Câu 1: …` are read with patterns; for anything else the LLM *extracts* the student's final answers. All comparisons are done in Python, never by the LLM, so the arithmetic is exact. |
| **AI-written text** | Style heuristics (stock AI phrases, very uniform sentence length, heavy markdown formatting, …) plus the LLM's opinion with reasons. These are signals to follow up on, **never proof**. |
| **Run code** | Runs `.py`, `.ipynb` (code cells) and `.js` submissions against your test cases (stdin → expected stdout, numbers compared with a tolerance) inside a locked-down Docker container. The LLM then reviews each program against the assignment and its test results. |

Every check run is saved as JSON in `reports/` so you can open it again later.

## 1. Google Cloud setup (one time)

1. Go to <https://console.cloud.google.com/>, create a project (any name).
2. **APIs & Services → Library**: enable **Google Classroom API** and **Google Drive API**.
3. **APIs & Services → OAuth consent screen**: choose *External* (or *Internal* on a
   Workspace for Education domain), fill in the app name and your email, and add yourself
   under **Test users**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - For the **web app**: type *Web application*, authorised redirect URI
     `http://localhost:8080/oauth2callback`.
   - For the **CLI**: type *Desktop app*.
   Download the JSON file(s).

Scopes used (all read-only): Classroom courses, rosters, student emails, coursework
submissions, and `drive.readonly` (to read attached files). Sign in with a **teacher**
account of the courses.

## 2. Local LLM setup

Install Ollama from <https://ollama.com/download>, then:

```bash
ollama pull qwen2.5:7b          # chat model (default)
ollama pull nomic-embed-text    # embeddings for paraphrase detection
```

Other good choices: `qwen2.5:14b` or `qwen3:8b` (better judgement, needs more memory),
`qwen2.5-coder:7b` (better code review), `qwen2.5:3b` (weak/older machines). Set the
model with `OLLAMA_MODEL`.

## 3. Run on your computer

Needs Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                   # edit FLASK_SECRET_KEY
cp ~/Downloads/client_secret_XXXX.json client_secret.json   # the *Web application* client
set -a; source .env; set +a            # Windows PowerShell: set the variables manually

python app.py
```

Open <http://localhost:8080> and sign in.

The code check needs Docker (Docker Desktop on Windows/macOS) and the sandbox images, pulled once:

```bash
docker pull python:3.12-slim
docker pull node:20-slim               # only for JavaScript submissions
```

### Command line (good for long runs on a whole class)

Uses the *Desktop app* client (`GOOGLE_DESKTOP_CLIENT_SECRETS=client_secret_desktop.json`);
the first run opens a browser to sign in and stores `token.json`.

```bash
python -m checker.cli status                     # is Ollama / Docker ready?
python -m checker.cli courses                    # course ids
python -m checker.cli works --course 123456789   # assignment ids
python -m checker.cli check --course 123456789 --work 987654321 \
    --checks duplicates,numbers,ai,code \
    --key examples/answer_key.txt --tests examples/tests.txt
```

`--no-llm` runs only the heuristics, pattern matching and tests (fast, no Ollama needed).
`--include-unsubmitted` also checks work that has not been turned in.

## 4. Run in Docker

`docker-compose.yml` starts the app and an Ollama container.

```bash
cp .env.example .env                                   # edit FLASK_SECRET_KEY
mkdir -p secrets
cp ~/Downloads/client_secret_XXXX.json secrets/client_secret.json   # Web application client

docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
docker pull python:3.12-slim                           # code sandbox image
```

Open <http://localhost:8080>. Reports and the CLI token are kept in `./data`.

- **Already have Ollama on your computer?** In `docker-compose.yml`, set
  `OLLAMA_URL: http://host.docker.internal:11434` and delete the `ollama` service.
- **NVIDIA GPU:** uncomment the `deploy:` block of the `ollama` service (needs
  nvidia-container-toolkit). On macOS, Docker cannot use the GPU; running Ollama natively
  is much faster there.
- **CLI inside Docker:** put the *Desktop app* client at `secrets/client_secret_desktop.json`, then
  ```bash
  docker compose run --rm --service-ports \
    -e GOOGLE_CLIENT_SECRETS=/secrets/client_secret_desktop.json app \
    python -m checker.cli courses
  ```
  and open the printed sign-in link in your browser.
- The app container mounts `/var/run/docker.sock` so it can start sandbox containers for
  student code. Anything with access to that socket has root-level control of the Docker
  host, so only run this on a machine you control. If you don't need the code check, remove
  that volume line.

## Input formats

**Answer key** (one per line, `#` for comments):

```
1: 42
2: 3.14 ± 0.01      # absolute tolerance
3: 25%              # same as 0.25
4: 1/3 ± 1%         # relative tolerance
```

Without a tolerance, answers within 0.1% count as correct.

**Code test cases**: blocks of stdin and expected stdout. Omit `--- input` for programs
that read nothing; omit `--- expected` to only check that the program runs without error.

```
--- input
3 4
--- expected
7
--- input
10 -2
--- expected
8
```

Examples are in [`examples/`](examples/).

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen2.5:7b` | chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `OLLAMA_TIMEOUT` | `300` | seconds per LLM call |
| `OLLAMA_NUM_CTX` | `8192` | context window; long submissions are truncated to fit |
| `CODE_RUNNER` | `docker` | `local` runs code directly on this machine with only CPU/memory limits. Use it only for trusted code or inside a throwaway VM. |
| `CODE_TIMEOUT` | `10` | seconds per test case |
| `CODE_MEMORY_MB` | `256` | memory per run |
| `PYTHON_IMAGE` / `NODE_IMAGE` | `python:3.12-slim` / `node:20-slim` | sandbox images |

## Code sandbox

Each test case runs in a fresh container with: no network, read-only filesystem (code is
mounted read-only), unprivileged user, all capabilities dropped, 256 MB memory, 1 CPU, 64
processes, and a hard time limit. The container is removed afterwards.

## Limitations

- `.docx` and `.pdf` attachments are not read yet. Ask students to submit Google Docs, or
  open those files in Google Drive and use *File → Save as Google Docs*.
- AI-text detection is unreliable by nature, especially on short texts and for students
  writing in a second language. Use it to decide what to look at, not to grade.
- A full check runs several LLM calls per student; with a 7B model on a laptop, expect about
  a minute per student. The CLI is better than the browser for big classes.

## Tests

```bash
pip install pytest
python -m pytest
```

## Project layout

```
app.py                 Flask web app (OAuth, pages, CSV export)
checker/classroom.py   Classroom + Drive API access
checker/status.py      turned in / late / missing / graded logic
checker/checks.py      runs selected checks for one assignment → report
checker/duplicates.py  copying detection
checker/numbers.py     answer-key checking
checker/aidetect.py    AI-writing indicators
checker/runner.py      sandboxed code execution
checker/llm.py         Ollama client
checker/cli.py         command line
templates/, static/    web UI
```
