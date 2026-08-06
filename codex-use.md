
## Delegating to Codex

Codex CLI is installed natively on Windows and can be driven from Claude Code as a
sub-agent. No terminal multiplexer is needed — do **not** install itmux / Cygwin tmux
for this. Native Windows TUIs inside a Cygwin pty have raw-mode and key-handling
problems, and `codex exec` is headless anyway.

- Binary: `C:\Users\sahas\AppData\Local\Programs\OpenAI\Codex\bin\codex.exe`
- Version verified: `codex-cli 0.145.0`
- User config: `~/.codex/config.toml`
- Repo config: `.codex/config.toml` (currently only the Atlassian MCP server)

### Basic loop

```powershell
# turn 1 — final message written to a file for easy reading back
codex exec -o .\tmp\codex-last.md "Task here"

# turn 2 — same session, its context intact
codex exec resume --last "Now also handle the partial-fill case"

# pin an explicit session when several codex threads run in parallel
codex exec resume <session-uuid> "follow-up"
```

Sessions persist under `CODEX_HOME/sessions`, so each resume is a fresh process on the
same thread. Working shape: Claude writes the task, fires `codex exec`, reads the `-o`
file, critiques, then `codex exec resume --last "<critique>"`.

### Model and reasoning effort

Model is a flag; effort is a config override. Both work on `exec` and `exec resume`, so
effort can be escalated mid-thread.

```powershell
codex exec -m gpt-5.6-sol -c model_reasoning_effort="high" -o .\tmp\codex-last.md "Task"
codex exec resume --last -m gpt-5.6-sol -c model_reasoning_effort="high" "Follow-up"
```

- Effort values enabled in this setup: `low`, `medium`, `high`, `xhigh`, `ultra`, `max`
- Config defaults: `model = "gpt-5.6-luna"`, `model_reasoning_effort = "medium"`
- Reusable presets beat repeated flags: drop `~/.codex/<name>.config.toml`, then
  `codex exec -p <name> "..."`

### Other useful `exec` flags

- `-o <file>` — write final message to a file (cleaner than scraping stdout)
- `--json` — JSONL event stream, including tool calls, not just prose
- `--output-schema <file>` — force a structured final answer against a JSON Schema
- `-C <dir>` / `--add-dir <dir>` — set or widen the writable working root
- `-s <mode>` — `read-only` | `workspace-write` | `danger-full-access`

Sandbox caveat: the user config sets `sandbox_mode = "danger-full-access"` globally with
`[windows] sandbox = "elevated"`, so passing `-s workspace-write` *tightens* codex rather
than enabling it. Omit `-s` to keep the default; pass it deliberately when a narrower box
is wanted for an agent-driven run.

### Long unattended runs

Fire `codex exec` with the Bash/PowerShell tool's `run_in_background` and get notified on
exit. Real tmux is only worth it if codex must run long while Claude Code stays
interactive — and then WSL + `apt install tmux` beats itmux.

## Architecture
