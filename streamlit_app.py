"""FLUX AI PLAYGROUND — public web demo (bring your own API key).

A safe, hosted slice of pi: chat with the agent and watch it plan, then use file
tools in an isolated per-session workspace.

Safety model (important — this is a public app):
  * **No shell.** ``run_bash`` is disabled, so visitors can never execute
    commands on the host.
  * **Sandboxed.** File tools (and uploads) are confined to a fresh temp
    directory per session; the Sandbox blocks ``../`` escapes.
  * **Your key, your session.** The API key you paste is used only for this
    session to talk to the model — never stored, logged, or committed.

Providers include free tiers (Groq, OpenRouter) so anyone can try it with a
free key — no credit card.
"""

from __future__ import annotations

import sys
import tempfile
import base64
import hashlib
import json
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pi_agent.agent import Agent  # noqa: E402
from pi_agent.config import SYSTEM_PROMPT, AgentConfig  # noqa: E402
from pi_agent.llm import (  # noqa: E402
    PROVIDERS,
    Usage,
    build_provider,
    estimate_cost,
    list_models,
    model_supports_vision,
)
from pi_agent.sandbox import Sandbox  # noqa: E402
from pi_agent.skills import build_system_prompt, load_skills  # noqa: E402
from pi_agent.tools.registry import build_default_tools  # noqa: E402
from pi_agent.upload import extract_zip_into_sandbox  # noqa: E402

SKILLS_DIR = Path(__file__).parent / "skills"
ASSETS_DIR = Path(__file__).parent / "TexturesAssets"
STATUS_ICON = {"done": "✅", "in_progress": "⏳", "pending": "⬜"}
UPLOAD_TYPES = [
    "zip",
    "csv",
    "tsv",
    "xlsx",
    "py",
    "js",
    "ts",
    "java",
    "go",
    "rs",
    "c",
    "cpp",
    "sh",
    "txt",
    "md",
    "json",
    "yaml",
    "yml",
    "html",
    "css",
]
DATA_EXTS = {"csv", "tsv", "xlsx", "json"}

st.set_page_config(
    page_title="FLUX AI PLAYGROUND",
    page_icon=str(ASSETS_DIR / "DarkModeLogo.png"),
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
            @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&display=swap');
            :root { --flux-purple: #a855f7; --flux-magenta: #e879f9; --flux-ink: #050308;
                --flux-panel: #0d0914; --flux-panel-raised: #140d1d; --flux-text: #f6f3ff;
                --flux-muted: #aaa3b8; }
      #MainMenu, footer { visibility: hidden; }
      /* never hide the sidebar open/close control */
      [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
      [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {
        visibility: visible !important; }
            .stApp { background: radial-gradient(circle at 50% -12%, #241035 0, var(--flux-ink) 42rem); color: var(--flux-text); }
            .block-container { padding: 2.8rem clamp(.75rem, 4vw, 3rem) 7rem; max-width: 1180px; }
            [data-testid="stSidebar"] { background: #08050c; border-right: 1px solid rgba(168,85,247,.24); }
            [data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.1rem; }
            /* purple light line under the brand lockup */
      .hero-rule { height: 3px; border: 0; border-radius: 3px; margin: .4rem 0 1.4rem;
                background: linear-gradient(90deg, var(--flux-magenta), var(--flux-purple), transparent); }
      /* buttons */
      .stButton button, .stDownloadButton button {
                border-radius: 10px; border: 1px solid rgba(139,92,246,.45); font-weight: 600;
        transition: transform .12s ease, border-color .12s ease; }
      .stButton button:hover, .stDownloadButton button:hover {
                border-color: var(--flux-magenta); transform: translateY(-1px); }
      /* chat bubbles + inputs */
            [data-testid="stChatMessage"] { border: 1px solid rgba(168,85,247,.12); border-radius: 16px;
                background: rgba(13,9,20,.68); padding: 1rem 1.1rem; margin: .8rem 0; }
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background: rgba(20,13,29,.84); }
      [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] { border-radius: 10px; }
      /* sidebar */
      /* provider pills */
      .pill { display:inline-block; padding:.2rem .65rem; margin:.15rem .25rem; border-radius:999px;
                font-size:.78rem; background:rgba(139,92,246,.14); border:1px solid rgba(208,0,255,.3);
                color:#d9c8ff; white-space:nowrap; }
            .flux-brand { display:flex; flex-direction:column; align-items:center; gap:.75rem; padding: 1rem 0 .5rem; }
            .flux-brand img { width:min(132px, 32vw); height:auto; border-radius:24px; }
                        .flux-wordmark { font-family:"Fredoka One", "Trebuchet MS", sans-serif; font-size:2.7rem;
                                line-height:1.1; font-weight:400; letter-spacing:.08em; color:#fff; text-align:center; }
            .flux-wordmark span { color:var(--flux-magenta); }
            .flux-kicker { color:var(--flux-magenta); font-size:.72rem; font-weight:700; letter-spacing:.16em;
                text-transform:uppercase; }
            .edit-hint { color:var(--flux-muted); font-size:.78rem; margin:.2rem 0 .7rem; }
            [data-testid="stExpander"] { border-color: rgba(168,85,247,.18); background: rgba(13,9,20,.45); }
            [data-testid="stFileUploader"] section { border-color: rgba(168,85,247,.34); background: rgba(20,13,29,.7); }
            [data-testid="stChatInput"] { border-color: rgba(232,121,249,.42); background: var(--flux-panel-raised); }
                        @media (max-width: 600px) {
                            .block-container { padding: 1rem .65rem 6rem; }
                            .flux-wordmark { font-size:1.5rem; letter-spacing:.05em; }
                            .flux-brand img { width:96px; border-radius:18px; }
                            .pill { font-size:.7rem; padding:.18rem .45rem; }
                            [data-testid="stChatMessage"] { padding:.75rem; margin:.55rem 0; }
                        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_args(args: dict | None) -> str:
    parts = []
    for key, value in (args or {}).items():
        text = str(value).replace("\n", " ")
        parts.append(f"{key}={text[:50] + '…' if len(text) > 50 else text}")
    return ", ".join(parts)


def _sandbox_dir() -> str:
    if "sandbox_dir" not in st.session_state:
        st.session_state.sandbox_dir = tempfile.mkdtemp(prefix="pi_demo_")
    return st.session_state.sandbox_dir


@st.cache_data(show_spinner="Fetching available models…", ttl=3600)
def _available_models(provider: str, key: str) -> list[str]:
    """Live model ids for this key (cached per provider+key). [] on failure."""
    try:
        return list_models(provider, api_key=key or None)
    except Exception:  # noqa: BLE001 - any failure -> fall back to presets
        return []


_LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
_IMAGE_UPLOAD_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]
_ARTIFACT_EXTS = {".pptx", ".pdf", ".docx", ".xlsx"}
_MAX_VIEW_BYTES = 200_000


def _message_content(message: dict) -> str | list[dict]:
    """Build provider-neutral content from one visible chat message."""
    images = message.get("images", [])
    if not images:
        return message["content"]
    return [{"type": "text", "text": message["content"]}, *images]


def _sync_agent_history() -> None:
    """Keep the model transcript aligned with the editable visible transcript."""
    agent = st.session_state.get("agent")
    if agent is not None:
        agent.messages = [
            {"role": message["role"], "content": _message_content(message)}
            for message in st.session_state.get("messages", [])
        ]


def _chat_snapshot() -> str:
    return json.dumps(st.session_state.get("messages", []), indent=2)


def _render_message(message: dict, index: int) -> None:
    """Render a message plus an edit control that never calls the model."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for image in message.get("images", []):
            st.image(base64.b64decode(image["data"]), caption=image.get("name", "Attached image"))
        st.markdown('<div class="edit-hint">Edit this entry to refine future context.</div>', unsafe_allow_html=True)
        with st.expander("Edit message" if message["role"] == "user" else "Edit response"):
            with st.form(f"edit_message_{index}"):
                edited = st.text_area("Text", value=message["content"], key=f"edit_text_{index}")
                if st.form_submit_button("Save edit"):
                    st.session_state.messages[index]["content"] = edited
                    st.session_state.edit_notice = (
                        "Saved. FLUX will use this edited message as context on the next turn."
                    )
                    st.rerun()


def _render_workspace_browser() -> None:
    """Show every workspace file and allow safe text edits in place."""
    root = Path(_sandbox_dir())
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return
    rels = [str(p.relative_to(root)) for p in files]
    has_artifact = any(p.suffix.lower() in _ARTIFACT_EXTS for p in files)
    with st.expander(f"🗂️ Workspace files ({len(rels)}) — view, edit & download", expanded=has_artifact):
        sel = st.selectbox("File", rels, key="wb_file")
        path = root / sel
        st.download_button(
            f"⬇ Download {Path(sel).name}",
            data=path.read_bytes(),
            file_name=Path(sel).name,
            key=f"wb_dl_{sel}",
        )
        suffix = path.suffix.lower()
        if suffix in _IMG_EXTS:
            st.image(str(path))
        elif path.stat().st_size > _MAX_VIEW_BYTES:
            st.caption(f"{path.stat().st_size:,} bytes — too large to preview, download instead.")
        else:
            try:
                text = path.read_text(encoding="utf-8")
                with st.form(f"wb_edit_{sel}"):
                    edited = st.text_area("Contents", value=text, height=320, key=f"wb_text_{sel}")
                    if st.form_submit_button("Save file"):
                        path.write_text(edited, encoding="utf-8")
                        st.success(f"Saved {sel}")
                        st.rerun()
            except UnicodeDecodeError:
                st.caption("Binary file — download instead.")


def _render_plan(box, steps) -> None:
    rows = [
        f"{STATUS_ICON.get(s.get('status'), '⬜')} {s.get('step', '')}"
        for s in steps
        if isinstance(s, dict)
    ]
    if rows:
        box.markdown("**📋 Plan**\n\n" + "\n\n".join(rows))


st.session_state.setdefault("messages", [])

# ── Sidebar: provider, model, key, upload, controls ──────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # Default to Groq: free key (no card), fast, and tool-capable — a first-time
    # visitor should land on a provider they can actually use in one minute.
    provider = st.selectbox(
        "Provider",
        list(PROVIDERS),
        index=list(PROVIDERS).index("groq"),
        format_func=lambda p: f"{p.title()}  🆓" if PROVIDERS[p].free else p.title(),
    )
    spec = PROVIDERS[provider]
    api_key = st.text_input(
        f"{provider.title()} API key",
        type="password",
        help="Used only for this session. Never stored, logged, or sent anywhere "
        "except the model provider.",
    )
    if spec.requires_key:
        free_note = " — 🆓 free, no credit card" if spec.free else ""
        st.caption(f"[Get a {provider.title()} key →]({spec.key_url}){free_note}")
    else:
        st.caption(
            "🖥️ Local & free — runs against Ollama at localhost:11434 "
            "(works when you run this app locally, not on the hosted demo)."
        )

    # Model picker — populated live from the provider's /models endpoint when a
    # key is present (always the real, current ids), else falls back to presets.
    _options = list(spec.models)
    if api_key or not spec.requires_key:
        _fetched = _available_models(provider, api_key or "")
        if _fetched:
            _options = _fetched
    _CUSTOM = "✏️ custom…"
    _picked = st.selectbox(f"Model ({len(_options)} available)", [*_options, _CUSTOM], index=0)
    model = (
        st.text_input("Custom model id", value=spec.default_model)
        if _picked == _CUSTOM
        else _picked
    )
    vision_enabled = model_supports_vision(provider, model)

    use_skills = st.toggle("Use skills (plan, tests, review, debug, …)", value=True)

    uploaded = st.file_uploader(
        "📎 Upload a file or project .zip",
        type=UPLOAD_TYPES,
        help="A file (or a zipped project) lands in the sandbox; then ask "
        "“review <file>” or “explain this project”.",
    )
    if uploaded is not None:
        ext = uploaded.name.lower().rsplit(".", 1)[-1]
        upload_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get("last_upload_hash") != upload_hash:
            st.session_state.last_upload_hash = upload_hash
            if ext == "zip":
                res = extract_zip_into_sandbox(uploaded.getvalue(), Sandbox(_sandbox_dir()))
                if res.error:
                    st.warning(res.error)
                else:
                    st.success(
                        f"Extracted **{len(res.extracted)}** files — contents are available below."
                    )
                    if res.skipped:
                        st.caption(f"Skipped {len(res.skipped)} (limits / unsafe paths).")
            elif ext in DATA_EXTS:
                if uploaded.size > 10_000_000:
                    st.warning("Data file too large (>10 MB).")
                else:
                    dest = Path(_sandbox_dir()) / Path(uploaded.name).name
                    try:
                        dest.write_bytes(uploaded.getvalue())
                        st.success(f"Uploaded **{dest.name}** — ask FLUX to analyze it.")
                    except OSError:
                        st.warning("Could not save that file.")
            elif uploaded.size > 200_000:
                st.warning("File too large (>200 KB). Zip it and upload as a project instead.")
            else:
                dest = Path(_sandbox_dir()) / Path(uploaded.name).name
                try:
                    dest.write_bytes(uploaded.getvalue())
                    st.success(f"Uploaded **{dest.name}** — ask FLUX to review it.")
                except OSError:
                    st.warning("Could not save that file.")

    if vision_enabled:
        image_upload = st.file_uploader(
            "🖼️ Attach image to next message",
            type=_IMAGE_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="Enabled because the selected model supports image input.",
            key=f"image_upload_{st.session_state.get('image_upload_version', 0)}",
        )
        st.caption("Images are sent only with your next FLUX message.")
    else:
        image_upload = []
        st.caption("Image input is unavailable for this model.")

    st.markdown("---")
    chat_name = st.text_input("Chat name", value="My FLUX chat")
    if st.session_state.messages and st.button("💾 Save chat", use_container_width=True):
        safe_name = "".join(c for c in chat_name.strip() if c.isalnum() or c in "-_ ").strip()
        if safe_name:
            chat_dir = Path(_sandbox_dir()) / ".flux_chats"
            chat_dir.mkdir(exist_ok=True)
            (chat_dir / f"{safe_name}.json").write_text(_chat_snapshot(), encoding="utf-8")
            st.success(f"Saved **{safe_name}**")
    saved_chats = sorted((Path(_sandbox_dir()) / ".flux_chats").glob("*.json")) if (Path(_sandbox_dir()) / ".flux_chats").exists() else []
    if saved_chats:
        selected_chat = st.selectbox("Saved chats", [p.stem for p in saved_chats])
        if st.button("↩ Load chat", use_container_width=True):
            loaded_path = next(p for p in saved_chats if p.stem == selected_chat)
            st.session_state.messages = json.loads(loaded_path.read_text(encoding="utf-8"))
            st.session_state.preserve_messages = True
            st.rerun()

    if st.session_state.get("edit_notice"):
        st.caption(st.session_state.pop("edit_notice"))

    if st.button("🧹 Clear conversation", use_container_width=True):
        for k in ("messages", "agent", "agent_key", "sess_in", "sess_out"):
            st.session_state.pop(k, None)
        st.session_state.messages = []
        st.rerun()

    if st.session_state.get("messages"):
        _transcript = "\n\n".join(
            f"**{m['role']}**:\n\n{m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "💬 Export chat (.md)",
            _transcript,
            file_name="flux-ai-playground-chat.md",
            use_container_width=True,
        )
        st.download_button(
            "🗃️ Export chat JSON",
            _chat_snapshot(),
            file_name="flux-ai-playground-chat.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    _cost_box = st.empty()

    def _render_session_meter() -> None:
        tin = st.session_state.get("sess_in", 0)
        tout = st.session_state.get("sess_out", 0)
        if not (tin or tout):
            return
        if spec.free:
            tail = "🆓 free tier"
        else:
            est = estimate_cost(model, Usage(tin, tout))
            tail = f"~${est:.4f}" if est is not None else "cost n/a"
        _cost_box.caption(f"📊 Session: {tin + tout:,} tokens · {tail}")

    _render_session_meter()
    st.caption(
        "🔒 **Safe demo:** shell disabled, file tools sandboxed to a temporary "
        "per-session folder. Your key stays in your session."
    )
    st.caption("FLUX AI PLAYGROUND · [Source on GitHub](https://github.com/Ashutosh0428/pi-agent)")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
        f"""
        <div class="flux-brand">
            <picture>
                <source media="(prefers-color-scheme: light)" srcset="data:image/png;base64,{base64.b64encode((ASSETS_DIR / 'LightModeLogo.png').read_bytes()).decode()}">
                <img src="data:image/png;base64,{base64.b64encode((ASSETS_DIR / 'DarkModeLogo.png').read_bytes()).decode()}" alt="FLUX AI PLAYGROUND logo">
            </picture>
            <div class="flux-wordmark"><span>FLUX</span> AI PLAYGROUND</div>
      <div style="color:#9aa0aa; font-size:.97rem; margin-top:.3rem;">
                A transparent AI playground — it <b>plans</b>, runs tools, and explains
        code &amp; data in a sandboxed workspace. Bring your own key (free options too).
      </div>
      <div style="margin-top:.7rem;">
        <span class="pill">🧠 8 providers</span>
        <span class="pill">📋 planner</span>
        <span class="pill">🤝 sub-agents</span>
        <span class="pill">📊 data → slides</span>
        <span class="pill">🔒 sandboxed</span>
      </div>
    </div>
    <hr class="hero-rule"/>
    """,
    unsafe_allow_html=True,
)

if spec.requires_key and not api_key:
    st.info(
        "👈 Pick a provider and paste an API key to start. "
        "**No paid key?** Choose **Groq** or **OpenRouter** (🆓) — free key, no card. "
        "Then ask, e.g., *“write a Python function that reverses a string, save it to "
        "utils.py, then add a test.”* Or upload a project **.zip** and ask me to "
        "*explain this project*."
    )
    _render_workspace_browser()
    st.stop()


def _get_agent() -> Agent:
    """Build (or reuse) an agent for the current provider/model/key/skills."""
    fingerprint = f"{provider}|{model}|{api_key[-6:]}|{use_skills}"
    if st.session_state.get("agent_key") != fingerprint:
        system_prompt = SYSTEM_PROMPT
        if use_skills:
            system_prompt = build_system_prompt(SYSTEM_PROMPT, load_skills(SKILLS_DIR))
        agent = Agent(
            provider=build_provider(model, provider, api_key=api_key),
            registry=build_default_tools(
                enable_shell=False,  # no raw shell on a public app
                enable_safe_command=True,  # restricted, read-only run_command is safe
                enable_subagents=True,  # sequential delegate (no recursion)
                enable_data=True,  # analyze_data + make_slides (fixed/safe)
            ),
            sandbox=Sandbox(_sandbox_dir()),
            config=AgentConfig(
                model=model,
                provider=provider,
                system_prompt=system_prompt,
                enable_shell=False,
                auto_approve=True,  # mutations are confined to the temp sandbox
                stream=True,  # live token streaming via assistant_delta events
            ),
        )
        st.session_state.agent = agent
        st.session_state.agent_key = fingerprint
        if not st.session_state.pop("preserve_messages", False):
            st.session_state.messages = []
    return st.session_state.agent


agent = _get_agent()
st.session_state.setdefault("messages", [])

for _index, msg in enumerate(st.session_state.messages):
    _render_message(msg, _index)

# Starter prompts — shown only on an empty conversation, so a first-time
# visitor can click once instead of inventing a prompt.
STARTERS = (
    (
        "✍️ Write a function + test",
        "Write a Python function that reverses a string, save it to utils.py, then write a pytest test for it and show me both files.",
    ),
    (
        "📦 Explain my uploaded project",
        "Explain this project: its purpose, how the pieces flow together, and the main components. If the workspace is empty, tell me to upload a project .zip in the sidebar first.",
    ),
    (
        "📊 Analyze data → slide deck",
        "Analyze the uploaded data file like a data scientist (stats, missing values, correlations), then make a short .pptx slide deck of the findings. If the workspace is empty, tell me to upload a CSV in the sidebar first.",
    ),
)
if not st.session_state.messages:
    st.caption("✨ Try one:")
    _chip_cols = st.columns(len(STARTERS))
    for _col, (_label, _text) in zip(_chip_cols, STARTERS):
        if _col.button(_label, use_container_width=True):
            st.session_state.queued_prompt = _text

# ── Chat turn ────────────────────────────────────────────────────────────────
prompt = st.session_state.pop("queued_prompt", None) or st.chat_input(
    "Ask FLUX to plan, write, review, or edit code…"
)
if prompt:
    image_blocks = [
        {
            "type": "image",
            "data": base64.b64encode(image.getvalue()).decode(),
            "media_type": image.type,
            "name": image.name,
        }
        for image in image_upload
    ]
    st.session_state.image_upload_version = st.session_state.get("image_upload_version", 0) + 1
    st.session_state.messages.append({"role": "user", "content": prompt, "images": image_blocks})
    _sync_agent_history()
    with st.chat_message("user"):
        st.markdown(prompt)
        for image in image_blocks:
            st.image(base64.b64decode(image["data"]), caption=image["name"])

    # Tell the model what's already in its workspace (uploaded files / extracted
    # projects), so weaker tool-users don't have to discover files. rglob so files
    # inside an extracted project's subfolders are listed too; cap to keep it lean.
    root = Path(_sandbox_dir())
    all_files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    workspace = all_files[:60]
    effective_prompt = prompt
    if workspace:
        more = (
            f" (+{len(all_files) - len(workspace)} more)" if len(all_files) > len(workspace) else ""
        )
        workspace_note = (
            f"(Files in your working directory: {', '.join(workspace)}{more}. "
            "Use read_file / list_dir to open them before reviewing or editing.)\n\n"
        )
        effective_prompt = workspace_note + prompt
    if image_blocks:
        effective_prompt = [
            {"type": "text", "text": effective_prompt},
            *image_blocks,
        ]

    # Route the most relevant skills for THIS prompt (token saver on free tiers);
    # the index of all skills stays in the prompt so the model knows the rest.
    if use_skills:
        agent.config.system_prompt = build_system_prompt(
            SYSTEM_PROMPT, load_skills(SKILLS_DIR), prompt=prompt, top_k=3
        )

    with st.chat_message("assistant"):
        plan_box = st.empty()
        status = st.status("FLUX is working…", expanded=True)
        answer_box = st.empty()
        usage_box = {"in": 0, "out": 0}
        stream_buf = {"text": ""}

        def on_event(kind, payload):
            if kind == "plan":
                _render_plan(plan_box, payload)
            elif kind == "assistant_delta":
                # Live token stream. Text emitted before a tool call is interim
                # narration — the tool_call branch clears it so only the final
                # answer remains in the box.
                stream_buf["text"] += payload
                answer_box.markdown(stream_buf["text"] + "▌")
            elif kind == "tool_call":
                stream_buf["text"] = ""
                answer_box.empty()
                status.write(f"🔧 `{payload.name}({_fmt_args(payload.args)})`")
            elif kind == "tool_result":
                out = payload["output"]
                status.write(f"```\n{out[:500] + ' …' if len(out) > 500 else out}\n```")
            elif kind == "usage":
                usage_box["in"] += payload["turn"].input_tokens
                usage_box["out"] += payload["turn"].output_tokens

        agent.on_event = on_event
        try:
            answer = agent.run(effective_prompt)
            status.update(label="done", state="complete", expanded=False)
            answer_box.markdown(answer or "_(no text response)_")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer or "_(no text response)_", "images": []}
            )
        except Exception as exc:  # provider error body has no key; scrub anyway, to be safe
            status.update(label="error", state="error")
            detail = str(exc)
            if api_key:
                detail = detail.replace(api_key, "***")
            st.error(f"Request failed ({type(exc).__name__}): {detail[:500]}")

        tok = usage_box["in"] + usage_box["out"]
        if spec.free:
            st.caption(f"🆓 free tier · {tok} tokens" if tok else "🆓 free tier")
        elif tok:
            est = estimate_cost(model, Usage(usage_box["in"], usage_box["out"]))
            st.caption(f"📊 {tok} tokens" + (f" · ~${est:.4f}" if est is not None else ""))

        # Session totals (sidebar meter) — accumulated across turns.
        st.session_state.sess_in = st.session_state.get("sess_in", 0) + usage_box["in"]
        st.session_state.sess_out = st.session_state.get("sess_out", 0) + usage_box["out"]
        _render_session_meter()


# Workspace browser — rendered LAST, so files the agent created during this
# very turn (code, decks, reports) are already listed, viewable, downloadable.
_render_workspace_browser()
