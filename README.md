# Google Classroom Checker

A teacher tool that reads assignments straight from Google Classroom and checks student work
with a **local LLM** (Qwen on [Ollama](https://ollama.com)). Student work stays on your
computer: it goes from Google to your machine and is never sent to a cloud AI service.

## What it does

**Overview (no LLM needed)**
- A grid of students × assignments: turned in, late, missing, graded, returned
- Per-assignment list with turn-in time, grade and attachments
- Reads Google Docs/Slides/Sheets, Word/Excel/PowerPoint, text, `.py`/`.js`, Colab notebooks, and `.zip`/`.tar` bundles of those
- CSV export of the whole course

**Checks (per assignment, run from the web page or the CLI)**

| Check | How it works |
|---|---|
| **Copying between students** | Exact wording (overlapping 5-word sequences), meaning (embeddings, so paraphrasing is caught), and code with renamed variables (identifier-normalised tokens). The most suspicious pairs are then reviewed by the LLM, which lists its evidence. |
| **Model answer sheet** | Upload (or paste) your worked answer sheet once. It is condensed into a short answer key saved as markdown in `answer_keys/`, cached by content, and reused for every student and every re-run — the sheet itself is never sent to the model again. |
| **Number answers** | The condensed key fills this in, or you paste one yourself (`1: 42`, `2: 3.14 ± 0.01`, `3: 25%`, `4: 1/3 ± 1%`). Answers written like `Q1: …`, `1) …` or `Câu 1: …` are read with patterns; for anything else the LLM *extracts* the student's final answers. All comparisons are done in Python, never by the LLM, so the arithmetic is exact. |
| **Compare with the model answer** | For written answers, each student is checked against the few points a correct answer must contain. A point the student states almost word for word is matched in Python for free; only the rest is sent to the LLM, as a numbered list it answers with the ids it cannot find. |
| **AI-written text** | Style heuristics (stock AI phrases, very uniform sentence length, heavy markdown formatting, …) plus the LLM's opinion with reasons. These are signals to follow up on, **never proof**. |
| **Run code** | Runs `.py`, `.ipynb` (code cells) and `.js` submissions against your test cases (stdin → expected stdout, numbers compared with a tolerance) inside a locked-down Docker container. The LLM then reviews each program against the assignment and its test results. |

Every check run is saved as JSON in `reports/` so you can open it again later.

**After the checks (optional)**

| | How it works |
|---|---|
| **Follow-up list** | Every report opens with who to look at first: *Needs action* (likely copying, nothing readable), *Worth a look* (AI-writing indicators, failing code, low score, work cut short) and *No concerns*, each with the reason. Worked out in Python from the report, no LLM. |
| **Triage agent** | A [Dify](https://dify.ai) workflow on your local Qwen reads that list after every run and writes short follow-up notes: what to do first, what the class missed, and a neutral message draft. It talks to the checker through an MCP server and only ever sees students as codes (S01, S02 …), never names, emails or answers. See [section 6](#6-triage-agent-dify--mcp). |
| **Draft grades** | Score × the assignment's points, proposed for the class. The agent may suggest holding some (a likely copy); you tick which ones to save and press the button. Nothing is written to Classroom without that. |

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

Pull as many as you like and **choose per run** - the check page has a drop-down of the
models installed on the machine, and the CLI takes `--model`:

```bash
python -m checker.cli models      # what is installed, with size and context window
```

```
MODEL                          SIZE   PARAMS    CONTEXT  USE
nomic-embed-text:latest       0.3 GB     137M      2,048  embeddings
qwen3:14b                     9.3 GB    14.8B     40,960  chat
qwen2.5:7b                    4.7 GB     7.6B     32,768  chat
qwen2.5-coder:7b              4.7 GB     7.6B     32,768  chat
```

Ollama reports what each model is *for*, so chat models and embedding models are listed
separately rather than guessed from their names. Good choices: `qwen2.5:7b` as a default,
`qwen3:14b` for better judgement on Vietnamese written answers, `qwen2.5-coder:7b` for
code review, `qwen2.5:3b` on a weak machine. `OLLAMA_MODEL` sets the starting default.

Prompts are sized to the model you pick: the budget is capped to that model's own context
window, so choosing a small-window model shortens the prompts instead of overflowing them.
Switching model also rebuilds the condensed answer key, because a different model
summarises the sheet differently - the cache is keyed on the model too.

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
python -m checker.cli models                     # which LLMs are installed
python -m checker.cli courses                    # course ids
python -m checker.cli works --course 123456789   # assignment ids
python -m checker.cli answerkey --file examples/model_answer.md   # condense an answer sheet
python -m checker.cli check --course 123456789 --work 987654321 \
    --checks duplicates,numbers,rubric,ai,code \
    --model-answer examples/model_answer.md --tests examples/tests.txt
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

## 5. Package the image

Build a single image and keep it as a file, so the same build can be moved to another
machine (or restored later) without rebuilding:

```bash
docker build -t ggc-checker:1.0 -t ggc-checker:latest .
mkdir -p dist
docker save ggc-checker:1.0 ggc-checker:latest | gzip > dist/ggc-checker-1.0.tar.gz
```

On the other machine:

```bash
docker load -i ggc-checker-1.0.tar.gz
mkdir -p secrets data
cp client_secret_XXXX.json secrets/client_secret.json      # Web application client
docker run -d --name ggc -p 8080:8080     -e FLASK_SECRET_KEY="$(openssl rand -hex 32)"     -e OAUTHLIB_INSECURE_TRANSPORT=1     -e OLLAMA_URL=http://host.docker.internal:11434     --add-host host.docker.internal:host-gateway     -v "$PWD/secrets:/secrets:ro" -v "$PWD/data:/data"     ggc-checker:1.0
```

The image carries only the app. Ollama, its models and the code-sandbox image are not
inside it — point `OLLAMA_URL` at an Ollama you already run (as above), or bring up the
`ollama` service from `docker-compose.yml`. Reports, cached answer keys and the CLI token
live in the mounted `/data`, so they survive `docker rm`.

`requirements.txt` is pinned: the build downloads exact versions, so the image rebuilds the
same next month. (On a slow link the `pip install` layer can still time out — rerun the
build, the earlier layers are cached.)

Run the CLI inside the container:

```bash
docker exec ggc python -m checker.cli answerkey --file examples/model_answer.md
docker exec ggc python -m checker.cli status
```

## 6. Triage agent (Dify + MCP)

After every check run the checker starts a Dify workflow. The workflow reads the
run's follow-up list through the checker's MCP server, has your local model write the
notes, and saves them back with a draft-grade proposal. The report page shows them as
soon as they land; the page refreshes by itself while the agent is working.

```
check run ──► report ──► Dify workflow ──MCP──► triage_brief   (codes only)
                              │ local Qwen writes the notes
                              └──────MCP──► save_triage ──► report page: notes + draft grades
                                                                  │ you tick and approve
                                                                  ▼
                                                       Google Classroom (draft grades)
```

**What the agent can and cannot do.** Its tools are `list_reports`, `triage_brief`,
`student_detail`, `grade_proposal` (read only) and `save_triage` (stores its notes and
proposes grades). None of them reaches Google: saving grades happens only when you press
the button on the report page, in your own session. The brief contains codes, scores and
signals, never names, emails or student answers, and names in quoted evidence are
replaced with codes too. Keep the model local anyway: the workflow ships set to
`qwen3:30B` on Ollama.

**Set it up (once)**

1. In the checker's `.env`, set a long random `AGENT_TOKEN` (e.g. `openssl rand -hex 32`).
   Running the checker with `python app.py`? Also set `HOST=0.0.0.0`, so Dify's containers
   can reach it.
2. Dify blocks requests to your own computer by default. In **Dify's** `docker/.env` add
   `SSRF_PROXY_ALLOW_PRIVATE_DOMAINS=host.docker.internal`, then `docker compose up -d`
   in Dify's `docker/` folder.
3. In Dify: **Tools → MCP → Add MCP server (HTTP)**
   - Name: `GGClassroom Checker` · Server identifier: **`ggclassroom`** (the workflow finds the tools by this)
   - Server URL: `http://host.docker.internal:8080/agent/mcp`
   - Header: `Authorization` = `Bearer <your AGENT_TOKEN>`

   Dify should list the five tools.
4. **Studio → Import DSL file** → `examples/dify/ggclassroom-triage.yml`. Check that the
   *Write follow-up notes* node uses your local model, then **Publish**.
5. In the workflow: **API Access → API Key → Create**. Put the key in the checker's `.env`
   as `DIFY_TRIAGE_KEY` (and `DIFY_API_URL=http://localhost/v1` when the checker runs
   outside Docker). Restart the checker.

From now on every check is triaged. The report page also has **Triage again**, and the
command line has

```bash
python -m checker.cli triage                     # the follow-up list for the newest report
python -m checker.cli triage --report <file> --agent    # and the agent's notes
```

`check` triages automatically when `DIFY_TRIAGE_KEY` is set (`--no-triage` to skip).

The same tools work from any MCP client, for example a Dify chat agent where you ask
"which students in Lab 3 should I talk to?", or Claude Code
(`claude mcp add --transport http ggclassroom http://localhost:8080/agent/mcp --header "Authorization: Bearer <token>"`).
Without MCP, `POST /agent/tools/<name>` with a JSON body and the same header does the same.

### Draft grades

The **Draft grades** section of a report proposes score × points for every marked
student (unreadable work gets no proposal). Untick anyone you are not sure about, press
**Save ticked draft grades to Classroom**, and each student's result is shown next to
their name. Draft grades are visible only to you until you return the work in Classroom.

Saving needs two things:

- **Permission.** Start the checker with `GRADE_WRITE=1` and sign in again. Google then asks
  for "view and manage coursework and grades" instead of view-only. Add the scope
  `.../auth/classroom.coursework.students` to your OAuth consent screen first.
- **An assignment this app created.** Google lets an app write grades only on coursework
  made by the same Google Cloud project. Assignments created on the Classroom website
  cannot be written by any outside app. The page says so and offers
  **Download CSV** with the proposed grades to type in instead.


The four checks always run in this order, and each one is a separate set of small LLM
calls - nothing accumulates into one growing conversation:

| | Check | LLM calls |
|---|---|---|
| 1 | Copying between students | one embedding per student, then one comparison per suspicious pair (5 by default) |
| 2 | Answers against the key | one extraction per student only for answers the patterns could not read; written points are compared one student at a time |
| 3 | AI-written text | one per student with written text |
| 4 | Running code | one review per student with code |

The web page shows a progress bar while this happens: the run goes into a background
thread, the browser follows `/job/<id>` and the bar fills as each student is finished.
Closing the tab does not stop the run - the report lands in `reports/` either way. The CLI
draws the same bar on stderr, so `python -m checker.cli check ... > out.txt` still works.

**Staying inside the context window.** Ollama does not reject an over-long prompt: it drops
tokens from the *start*, which silently eats the system prompt and the "reply with JSON"
instruction, and the model then answers with prose. So every prompt is built to fit:

* the budget comes from `OLLAMA_NUM_CTX` minus the system prompt and `OLLAMA_REPLY_TOKENS`,
  converted with `OLLAMA_CHARS_PER_TOKEN` (3.0 - measured as 5.2 chars/token for English,
  3.9 for Vietnamese and 3.3 for code on `qwen2.5:7b`, so 3.0 is the safe floor);
* `num_predict` holds the reply to the reserve, so prompt + answer cannot overflow;
* a prompt with several parts shares the budget out: short parts (the assignment text, the
  list of required points) are kept whole, and only the long ones are trimmed - so the
  instructions can never be crowded out by one enormous submission;
* when something *is* trimmed, the report and the CSV say so in a `Truncated` column
  instead of quietly marking a student on half their work.

Lowering `OLLAMA_NUM_CTX` to 4096 to save memory therefore shortens the prompts to match,
rather than breaking the checks.

## Scores and the CSV

A student whose attachments could not be read at all is left **unmarked** rather than
scored zero, and the CSV says `NOTHING READABLE` in its *Read* column - a blank score and
a zero must never look the same in a column a teacher copies marks from.

The **final score comes only from the answer sheet**: a numeric question is right or wrong,
a written question earns the fraction of its required points the student covered, and each
question counts once. Copying, AI indicators and code test results sit next to the score in
the report but are never folded into it - they are signals for you to act on, not marks.

Every run writes `reports/<timestamp>_<assignment>.json` and a matching `.csv`. The CSV has
one row per student: score, per-question marks, answers correct, written points, AI
indicators, code tests passed, who they look most similar to, and anything that was
truncated. **Download CSV** on the report page regenerates it from the JSON, and the file is
written with a BOM so Excel on Windows shows Vietnamese names correctly.

## Model answer sheet

Give the checker your answer sheet once and it does the typing for you:

```bash
python -m checker.cli answerkey --file dapan.md      # see what it extracted
```

```
# Answer key - dapan.md

## Numeric answers

1: 42
2: 3.14 ± 0.01

## Required points for written answers

- 3: năng lượng không tự sinh ra; chuyển hoá từ dạng này sang dạng khác; đơn vị Joule
```

The result is written to `answer_keys/<hash>.md` (and `.json`). Because it is keyed on the
sheet's content and the model name, re-uploading the same sheet costs nothing; changing one
line in the sheet builds a new key. Read the generated file before you grade a class with it
— a 7B model paraphrases, and you are the one signing the marks. Edit it and pass it with
`--key` if you want to correct something.

Then use it from the command line:

```bash
python -m checker.cli check --course 123 --work 456 \
    --checks numbers,rubric --model-answer dapan.md
```

or in the browser, under **Model answer sheet** on the check page: upload a `.md`/`.txt`
file or a `.docx`, or select-all-copy-paste from a Google Doc into the box next to it
(`.pdf` and the old `.doc` are not read).

Why it is cheap: the sheet goes to the model **once**, not once per student. Numeric answers
are then compared in Python with no LLM call at all, and written answers send only the point
list — a few words each — instead of the whole document.

`--checks rubric` needs the LLM. For Vietnamese answer sheets `qwen2.5:7b` is weak at judging
negated paraphrases; `OLLAMA_MODEL=qwen3:14b` is noticeably more accurate here if you have
the memory for it.

## Input formats

**Answer key** (one per line, `#` for comments):

```
1: 42
2: 3.14 ± 0.01      # absolute tolerance
3: 25%              # same as 0.25
4: 1/3 ± 1%         # relative tolerance
```

A tolerance may also be written out in words or brackets, which is how answer sheets
usually put it: `2: 3,14 (sai số 0,01)`, `3: 2.5 (tol 0.1)`. Decimal commas work in both
the value and the tolerance. Without a tolerance, answers within 0.1% count as correct.

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
| `OLLAMA_MODEL` | `qwen2.5:7b` | the chat model a run starts with; pick another per run |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | the embedding model a run starts with |
| `OLLAMA_TIMEOUT` | `300` | seconds per LLM call |
| `OLLAMA_NUM_CTX` | `8192` | context window; prompts are packed to fit it |
| `OLLAMA_EMBED_NUM_CTX` | `2048` | the embedding model's window, which is much smaller |
| `OLLAMA_CHARS_PER_TOKEN` | `3.0` | estimate used to pack prompts (see below) |
| `OLLAMA_REPLY_TOKENS` | `700` | reserved for the reply, and the cap on it |
| `ANSWER_KEYS_DIR` | `answer_keys` | where condensed answer keys are cached |
| `CODE_RUNNER` | `docker` | `local` runs code directly on this machine with only CPU/memory limits. Use it only for trusted code or inside a throwaway VM. |
| `CODE_TIMEOUT` | `10` | seconds per test case |
| `CODE_MEMORY_MB` | `256` | memory per run |
| `PYTHON_IMAGE` / `NODE_IMAGE` | `python:3.12-slim` / `node:20-slim` | sandbox images |
| `AGENT_TOKEN` | *(off)* | turns on the MCP server at `/agent/mcp`; clients send it as a Bearer token |
| `APP_BASE_URL` | `http://localhost:8080` | used in the review links the agent hands back |
| `DIFY_API_URL` / `DIFY_TRIAGE_KEY` | `http://localhost/v1` / *(off)* | the triage workflow; with a key, every run is triaged |
| `DIFY_TIMEOUT` / `TRIAGE_LANGUAGE` | `900` / `Vietnamese` | seconds to wait for the workflow; language of its notes |
| `GRADE_WRITE` | `0` | `1` asks Google for permission to save draft grades |
| `GRADE_STEP` | `0.1` | draft grades are rounded to this |

## Code sandbox

Each test case runs in a fresh container with: no network, read-only filesystem (code is
mounted read-only), unprivileged user, all capabilities dropped, 256 MB memory, 1 CPU, 64
processes, and a hard time limit. The container is removed afterwards.

## Limitations

- **Read:** Google Docs/Slides/Sheets, `.docx`, `.xlsx`, `.pptx`, plain text, `.py`,
  `.js`, `.ipynb`, and `.zip`/`.tar` holding any of those. Office files give up their
  text only: tables come out as lines, and images, equations and charts are skipped.
  Files are identified by their contents, so an attachment with no extension still works.
- **Not read:** `.pdf` (needs a PDF parser), `.rar` and `.7z` (proprietary), the old
  `.doc`/`.xls`, and photos of handwritten work (needs text recognition). These are
  listed under "Not checked" with what to ask for instead, and the student is left
  **unmarked** - not scored zero, because nothing was seen.
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
checker/answerkey.py   condenses the model answer sheet, compares written answers
checker/documents.py   reads .docx/.xlsx/.pptx/.zip/.tar attachments (stdlib only)
checker/results.py     final score from the answer sheet, CSV export
checker/numbers.py     answer-key checking
checker/aidetect.py    AI-writing indicators
checker/runner.py      sandboxed code execution
checker/llm.py         Ollama client, context-window packing
checker/triage.py      follow-up list from a report, student codes, stored agent notes
checker/grades.py      draft-grade proposals, approval, saving to Classroom
checker/agent_tools.py the agent's tools (shared by MCP and the JSON API)
checker/mcp.py         MCP server (Streamable HTTP, JSON replies, stdlib only)
checker/dify.py        starts the Dify triage workflow after a run
checker/cli.py         command line
examples/dify/         the Dify triage workflow to import
templates/, static/    web UI (incl. the progress page)
```
