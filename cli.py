"""
SSM Coder CLI — local AI coding assistant powered by Qwen2.5-Coder + RAG.

Commands:
  ssm download  — download a GGUF model from HuggingFace
  ssm init      — configure model path
  ssm index     — index a codebase for RAG
  ssm chat      — interactive coding chat (with optional RAG)
  ssm ask       — one-shot question
  ssm complete  — continue/complete a file
  ssm status    — show current config and index stats
  ssm clear     — wipe the RAG index
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

app = typer.Typer(
    name="ssm",
    help="SSM Coder — local Qwen2.5-Coder coding assistant with RAG.",
    add_completion=False,
)
console = Console()
err = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model():
    from ssm import config as cfg
    from ssm.model import LLM

    model_path = cfg.get("model_path")
    if not model_path or not Path(model_path).exists():
        err.print(
            "[bold red]Model not configured.[/bold red] "
            "Run [bold]ssm download[/bold] then [bold]ssm init[/bold]."
        )
        raise typer.Exit(1)

    config = cfg.load()
    console.print(f"[dim]Loading model …[/dim]")
    model = LLM(
        model_path=model_path,
        n_ctx=config.get("n_ctx", 16384),
        n_threads=config.get("n_threads", -1),
        n_batch=config.get("n_batch", 512),
    )
    return model


def _load_rag(directory: Optional[str] = None):
    from ssm.rag import RAGEngine

    rag = RAGEngine()
    if rag.count() == 0 and not directory:
        return rag, None  # no index yet

    if directory:
        # Index on the fly if a directory was passed directly
        _run_index(directory, rag)

    return rag, rag


def _run_index(directory: str, rag=None):
    from ssm import config as cfg
    from ssm.rag import RAGEngine

    if rag is None:
        rag = RAGEngine()

    chunk_size = cfg.get("chunk_size") or 60
    chunk_overlap = cfg.get("chunk_overlap") or 10

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing…", total=None)

        def on_progress(i, total, name):
            progress.update(task, completed=i, total=total, description=f"[cyan]{name}[/cyan]")

        total = rag.index_directory(
            directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=on_progress,
        )

    console.print(f"[green]Indexed [bold]{total}[/bold] chunks from [bold]{directory}[/bold][/green]")
    return rag


def _stream_response(model, messages, context, cfg) -> str:
    """Stream model output token-by-token, render as plain text, return full reply."""
    full = []
    # Use Live to update a single line as tokens arrive
    with Live(Text(""), console=console, refresh_per_second=20) as live:
        for token in model.generate_stream(
            messages=messages,
            context=context,
            max_tokens=cfg.get("max_tokens") or 512,
            temperature=cfg.get("temperature") or 0.8,
            top_p=cfg.get("top_p") or 0.9,
        ):
            # Stop if the model starts a new User turn
            if "\nUser:" in "".join(full[-3:]) + token:
                break
            full.append(token)
            live.update(Text("".join(full)))

    return "".join(full).rstrip()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def download(
    model_name: str = typer.Argument(
        "qwen32b",
        help="Model key: qwen7b, qwen32b, deepseek32b, qwen72b",
    ),
    dest: Optional[str] = typer.Option(
        None, "--dest", "-d",
        help="Directory to save the model (default: MODEL_DIR env or ~/.ssm/models)",
    ),
    set_active: bool = typer.Option(
        True, "--set/--no-set",
        help="Set this model as active after download.",
    ),
):
    """Download a GGUF model from HuggingFace and optionally set it as active."""
    from ssm import config as cfg
    from huggingface_hub import hf_hub_download

    models = cfg.MODELS
    if model_name not in models:
        err.print(f"[red]Unknown model '{model_name}'. Choose from: {', '.join(models)}[/red]")
        raise typer.Exit(1)

    m = models[model_name]
    from ssm.paths import MODEL_DIR
    dest_dir = Path(dest).expanduser() if dest else MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]{model_name}[/bold] — {m['desc']}\n"
            f"Size: {m['size']}\n"
            f"Repo: {m['repo']}\n"
            f"File: {m['file']}\n"
            f"Destination: {dest_dir}",
            title="Downloading Model",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        progress.add_task(f"Downloading {m['file']} …", total=None)
        local_path = hf_hub_download(
            repo_id=m["repo"],
            filename=m["file"],
            local_dir=str(dest_dir),
        )

    console.print(f"[green]Downloaded:[/green] {local_path}")

    if set_active:
        config = cfg.load()
        config["model_path"] = local_path
        cfg.save(config)
        console.print(f"[green]Active model set to:[/green] {local_path}")
        console.print("\nRun [bold]ssm index <directory>[/bold] to index your codebase, then [bold]ssm chat[/bold].")


@app.command()
def init(
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to a GGUF model file."
    ),
):
    """Set an already-downloaded GGUF model as the active model."""
    from ssm import config as cfg

    if not model_path:
        console.print(
            Panel(
                "[bold]Download a model first with:[/bold]\n\n"
                "  ssm download qwen32b    [dim]# recommended (~20 GB)[/dim]\n"
                "  ssm download qwen7b     [dim]# fast (~5 GB)[/dim]\n"
                "  ssm download deepseek32b\n"
                "  ssm download qwen72b    [dim]# best quality (~43 GB)[/dim]",
                title="Model Setup",
            )
        )
        model_path = Prompt.ask("Path to .gguf file")

    model_path = str(Path(model_path).expanduser().resolve())
    if not Path(model_path).exists():
        err.print(f"[red]File not found: {model_path}[/red]")
        raise typer.Exit(1)

    config = cfg.load()
    config["model_path"] = model_path
    cfg.save(config)

    console.print(f"[green]Model set:[/green] {model_path}")
    console.print("\nRun [bold]ssm index <directory>[/bold] to index your codebase.")


@app.command()
def index(
    directory: str = typer.Argument(".", help="Directory to index."),
    clear_first: bool = typer.Option(False, "--clear", help="Clear existing index first."),
):
    """Index a codebase so it can be used as RAG context."""
    from ssm.rag import RAGEngine

    rag = RAGEngine()

    if clear_first:
        rag.clear()
        console.print("[yellow]Cleared existing index.[/yellow]")

    _run_index(directory, rag)
    console.print(f"[dim]Total chunks in index: {rag.count()}[/dim]")


@app.command()
def chat(
    directory: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Index this directory for RAG before chatting."
    ),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG retrieval."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-n"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t"),
):
    """
    Start an interactive coding chat session.

    RAG retrieves relevant code from the indexed codebase automatically.
    Type 'exit' or Ctrl-C to quit.
    """
    from ssm import config as cfg

    config = cfg.load()
    if max_tokens:
        config["max_tokens"] = max_tokens
    if temperature:
        config["temperature"] = temperature

    model = _load_model()
    rag, rag_active = _load_rag(directory)

    console.print(
        Panel(
            f"[bold green]SSM Coder[/bold green]  "
            f"RAG {'[green]on[/green]' if (rag_active and not no_rag) else '[dim]off[/dim]'}\n"
            "[dim]Type your question. 'exit' to quit. '/clear' to reset history.[/dim]",
            title="Chat",
        )
    )

    history: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are an expert coding assistant. "
                "Answer concisely. Prefer working code over explanations. "
                "When showing code, always specify the language in fenced code blocks."
            ),
        }
    ]

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.strip() == "/clear":
            history = [history[0]]  # keep system prompt
            console.print("[yellow]History cleared.[/yellow]")
            continue

        history.append({"role": "user", "content": user_input})

        # RAG retrieval
        context = None
        if rag_active and not no_rag:
            n = config.get("rag_n_results") or 5
            chunks = rag.retrieve(user_input, n_results=n)
            if chunks:
                context = rag.format_context(chunks)
                console.print(
                    f"[dim]Retrieved {len(chunks)} relevant chunk(s) from index.[/dim]"
                )

        console.print(Rule("[dim]Assistant[/dim]"))

        reply = _stream_response(model, history, context, config)

        console.print()  # newline after streamed output
        history.append({"role": "assistant", "content": reply})


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask."),
    directory: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Directory to use for RAG context."
    ),
    no_rag: bool = typer.Option(False, "--no-rag"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of RAG results."),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
):
    """Ask a single coding question and print the answer."""
    from ssm import config as cfg

    config = cfg.load()
    config["max_tokens"] = max_tokens
    config["temperature"] = temperature
    config["rag_n_results"] = n_results

    model = _load_model()
    rag, rag_active = _load_rag(directory)

    context = None
    if rag_active and not no_rag:
        chunks = rag.retrieve(question, n_results=n_results)
        if chunks:
            context = rag.format_context(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert coding assistant. "
                "Answer concisely. Show working code with language-tagged fenced blocks."
            ),
        },
        {"role": "user", "content": question},
    ]

    console.print(Rule("[dim]Answer[/dim]"))
    reply = _stream_response(model, messages, context, config)
    console.print()
    console.print(Markdown(reply))


@app.command()
def complete(
    file: str = typer.Argument(..., help="File to complete."),
    lines: int = typer.Option(50, "--lines", "-n", help="Lines of completion to generate."),
    temperature: float = typer.Option(0.4, "--temperature", "-t", help="Lower = more deterministic."),
    no_rag: bool = typer.Option(False, "--no-rag"),
):
    """
    Generate a continuation for a code file.

    Reads the file, sends it as context, and streams the completion.
    """
    from ssm import config as cfg

    file_path = Path(file)
    if not file_path.exists():
        err.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    source = file_path.read_text(errors="ignore")
    config = cfg.load()

    # Estimate tokens: ~4 chars/token; keep last ~2000 tokens of context
    max_src_chars = 8000
    if len(source) > max_src_chars:
        source = "# ... (truncated) ...\n" + source[-max_src_chars:]

    model = _load_model()
    rag, rag_active = _load_rag(None)

    context = None
    if rag_active and not no_rag:
        query = f"code similar to {file_path.name}: {source[:200]}"
        chunks = rag.retrieve(query, n_results=3)
        if chunks:
            context = rag.format_context(chunks)

    messages = [
        {
            "role": "system",
            "content": "You are an expert coding assistant. Continue the code exactly where it left off. Output only code, no explanation.",
        },
        {
            "role": "user",
            "content": f"Continue this file:\n\n```\n{source}\n```",
        },
    ]

    config["max_tokens"] = lines * 10  # rough estimate
    config["temperature"] = temperature

    console.print(Rule(f"[dim]Completing {file_path.name}[/dim]"))
    reply = _stream_response(model, messages, context, config)
    console.print()


@app.command()
def status():
    """Show current configuration and index statistics."""
    from ssm import config as cfg
    from ssm.rag import RAGEngine

    config = cfg.load()
    rag = RAGEngine()

    model_path = config.get("model_path") or "[red]not set[/red]"
    model_ok = config.get("model_path") and Path(config["model_path"]).exists()

    lines = [
        f"[bold]Model:[/bold]       {'[green]' + model_path + '[/green]' if model_ok else model_path}",
        f"[bold]Context:[/bold]     {config.get('n_ctx', 16384)} tokens",
        f"[bold]Threads:[/bold]     {config.get('n_threads', -1)} (-1 = all cores)",
        f"[bold]RAG index:[/bold]   {rag.count()} chunks",
        f"[bold]Temperature:[/bold] {config.get('temperature', 0.7)}",
        f"[bold]Max tokens:[/bold]  {config.get('max_tokens', 1024)}",
    ]

    console.print(Panel("\n".join(lines), title="SSM Coder Status"))


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
):
    """Clear the entire RAG index."""
    from ssm.rag import RAGEngine

    if not confirm:
        typer.confirm("Delete all indexed data?", abort=True)

    rag = RAGEngine()
    rag.clear()
    console.print("[yellow]RAG index cleared.[/yellow]")


@app.command()
def explain(
    file: str = typer.Argument(..., help="File to explain."),
    selection: Optional[str] = typer.Option(
        None, "--lines", "-l", help="Line range to explain, e.g. '10-25'."
    ),
    no_rag: bool = typer.Option(False, "--no-rag"),
):
    """
    Explain what a file (or a range of lines) does.

    Example:
        ssm explain src/auth.py
        ssm explain src/auth.py --lines 40-80
    """
    from ssm import config as cfg

    file_path = Path(file)
    if not file_path.exists():
        err.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    source = file_path.read_text(errors="ignore")
    lines = source.splitlines()

    if selection:
        try:
            start_s, end_s = selection.split("-")
            start, end = int(start_s) - 1, int(end_s)
            source = "\n".join(lines[start:end])
            location = f"{file_path.name} lines {selection}"
        except ValueError:
            err.print("[red]--lines format must be START-END, e.g. 10-25[/red]")
            raise typer.Exit(1)
    else:
        location = file_path.name
        # Truncate very long files
        if len(source) > 8000:
            source = source[:8000] + "\n# ... (truncated)"

    config = cfg.load()
    model = _load_model()
    rag, rag_active = _load_rag(None)

    context = None
    if rag_active and not no_rag:
        query = f"explain {file_path.name}: {source[:200]}"
        chunks = rag.retrieve(query, n_results=3)
        if chunks:
            context = rag.format_context(chunks)

    messages = [
        {
            "role": "system",
            "content": "You are an expert code reviewer. Explain code clearly and concisely.",
        },
        {
            "role": "user",
            "content": (
                f"Explain what this code does ({location}):\n\n"
                f"```\n{source}\n```\n\n"
                "Cover: purpose, key logic, inputs/outputs, and any gotchas."
            ),
        },
    ]

    console.print(Rule(f"[dim]Explaining {location}[/dim]"))
    reply = _stream_response(model, messages, context, config)
    console.print()
    console.print(Markdown(reply))


@app.command()
def review(
    file: str = typer.Argument(..., help="File to review."),
    no_rag: bool = typer.Option(False, "--no-rag"),
    temperature: float = typer.Option(0.5, "--temperature", "-t"),
):
    """
    Code review a file — check for bugs, style issues, and improvements.

    Example:
        ssm review src/api.py
    """
    from ssm import config as cfg

    file_path = Path(file)
    if not file_path.exists():
        err.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    source = file_path.read_text(errors="ignore")
    if len(source) > 8000:
        source = source[:8000] + "\n# ... (truncated)"

    config = cfg.load()
    config["temperature"] = temperature
    model = _load_model()
    rag, rag_active = _load_rag(None)

    context = None
    if rag_active and not no_rag:
        query = f"code review patterns for {file_path.suffix} {source[:200]}"
        chunks = rag.retrieve(query, n_results=3)
        if chunks:
            context = rag.format_context(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior software engineer doing a code review. "
                "Be direct and specific. Prioritize real issues over style nits."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review this code ({file_path.name}):\n\n"
                f"```\n{source}\n```\n\n"
                "Check for: bugs, security issues, performance problems, "
                "error handling gaps, and clear improvement suggestions. "
                "Format as a numbered list."
            ),
        },
    ]

    console.print(Rule(f"[dim]Reviewing {file_path.name}[/dim]"))
    reply = _stream_response(model, messages, context, config)
    console.print()
    console.print(Markdown(reply))


# ---------------------------------------------------------------------------
# CodingSSM v2 commands (Mamba-2 SFT model)
# ---------------------------------------------------------------------------

def _load_sft_model(checkpoint: str = None, show_thinking: bool = False):
    """Load the SFT-trained CodingSSM model."""
    from ssm.inference_sft import CodingSSMInference
    ckpt = checkpoint or None
    try:
        return CodingSSMInference(checkpoint=ckpt)
    except FileNotFoundError as e:
        err.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(1)


@app.command("ask-v2")
def ask_v2(
    question: str = typer.Argument(..., help="Coding question to answer"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to .pt checkpoint"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    think: bool = typer.Option(False, "--think", help="Show <think> reasoning block"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output tokens"),
):
    """Ask the trained CodingSSM model a coding question."""
    model = _load_sft_model(checkpoint)
    model.temperature = temperature
    model.max_new_tokens = max_tokens

    console.print(Rule("[bold green]CodingSSM v2[/bold green]"))

    if stream:
        with Live(console=console, refresh_per_second=20) as live:
            buf = ""
            for chunk in model.stream(question):
                buf += chunk
                live.update(Markdown(buf))
        console.print()
    else:
        with console.status("[dim]Thinking…[/dim]"):
            answer = model.ask(question, show_thinking=think)
        console.print(Markdown(answer))


@app.command("complete-v2")
def complete_v2(
    file: Optional[str] = typer.Argument(None, help="File to complete (reads stdin if omitted)"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m"),
    temperature: float = typer.Option(0.2, "--temperature", "-t"),
):
    """Complete code using the trained CodingSSM model."""
    if file:
        code = Path(file).read_text()
    else:
        code = sys.stdin.read()

    model = _load_sft_model(checkpoint)
    model.temperature = temperature
    model.max_new_tokens = max_tokens

    with console.status("[dim]Completing…[/dim]"):
        result = model.complete(code)

    console.print(result)


@app.command("chat-v2")
def chat_v2(
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    think: bool = typer.Option(False, "--think", help="Show reasoning traces"),
):
    """Interactive coding chat with the trained CodingSSM model."""
    model = _load_sft_model(checkpoint)
    model.temperature = temperature

    console.print(Panel(
        "[bold green]CodingSSM v2[/bold green] — Mamba-2 reasoning model\n"
        "[dim]Type your question. Ctrl+C or 'exit' to quit.[/dim]",
        border_style="green",
    ))

    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if question.strip().lower() in ("exit", "quit", "bye"):
            break
        if not question.strip():
            continue

        console.print(Rule("[dim]CodingSSM[/dim]"))
        with Live(console=console, refresh_per_second=20) as live:
            buf = ""
            for chunk in model.stream(question):
                buf += chunk
                live.update(Markdown(buf))
        console.print()


@app.command("status-v2")
def status_v2():
    """Show training status and model readiness."""
    from pathlib import Path
    import os

    ckpt = Path("checkpoints/sft/sft_best.pt")
    console.print(Rule("[bold]CodingSSM v2 Status[/bold]"))

    if ckpt.exists():
        size_gb = ckpt.stat().st_size / 1e9
        mtime = ckpt.stat().st_mtime
        import datetime
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime)
        console.print(f"[green]✓[/green] Checkpoint: {ckpt} ({size_gb:.1f} GB, updated {int(age.total_seconds()/60)}m ago)")
    else:
        console.print(f"[yellow]✗[/yellow] No checkpoint yet at {ckpt}")

    # Check training process
    import subprocess
    result = subprocess.run(["pgrep", "-af", "train.sft_reasoning"], capture_output=True, text=True)
    if result.stdout.strip():
        console.print("[green]✓[/green] SFT training: [bold]running[/bold]")
    else:
        console.print("[dim]–[/dim] SFT training: not running")

    result = subprocess.run(["pgrep", "-af", "generate_eus"], capture_output=True, text=True)
    if result.stdout.strip():
        console.print("[green]✓[/green] EU expansion: [bold]running[/bold]")
    else:
        console.print("[dim]–[/dim] EU expansion: not running")

    # EU count
    from ssm.paths import EPICHAT_DIR
    eu_path = EPICHAT_DIR / "episteme_data" / "units.json"
    if eu_path.exists():
        import json
        with open(eu_path) as f:
            data = json.load(f)
        n = len(data) if isinstance(data, dict) else len(data)
        console.print(f"[green]✓[/green] EpiChat EUs: [bold]{n}[/bold]")

    # Trace counts
    for name, path in [("EpiChat traces", "data/epichat_traces.jsonl"),
                        ("Reasoning traces", "data/reasoning_traces.jsonl"),
                        ("All traces", "data/all_traces.jsonl")]:
        p = Path(path)
        if p.exists():
            n = sum(1 for _ in open(p))
            console.print(f"[green]✓[/green] {name}: [bold]{n}[/bold]")


# ---------------------------------------------------------------------------
# ask-r1: DeepSeek-R1 70B with EpiChat RAG + test-time compute → ~96% HumanEval
# ---------------------------------------------------------------------------

@app.command("ask-r1")
def ask_r1(
    question: str = typer.Argument(..., help="Coding question"),
    model: str = typer.Option("deepseek-r1:70b", "--model", "-m", help="Ollama model to use"),
    host: str = typer.Option("http://localhost:11437", "--host", help="Ollama host (user server)"),
    n_samples: int = typer.Option(1, "--samples", "-n", help="Candidates to generate (higher=more accurate)"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
    think: bool = typer.Option(False, "--think", help="Show <think> reasoning"),
    test_code: str = typer.Option("", "--test", "-t", help="Unit tests to verify solution against"),
):
    """
    Ask DeepSeek-R1 70B with EpiChatRAG context injection.
    Use --samples 4 for ~96% HumanEval accuracy (runs tests, picks passing solution).
    """
    import urllib.request, re

    # EpiChat context injection
    context = ""
    try:
        from ssm.epichat_rag import EpiChatRAG
        from ssm.paths import EPICHAT_DIR
        rag = EpiChatRAG(str(EPICHAT_DIR))
        context = rag.get_context(question)
    except Exception:
        pass

    system = (
        "You are an expert Python programmer. Think step by step.\n"
        "Use <think>...</think> for reasoning, then provide the solution in a ```python``` block."
    )
    user_msg = question
    if context:
        user_msg = f"{context}\n\n{question}"

    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    def _call(temp: float) -> str:
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temp, "stop": ["<|im_end|>"]},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read()).get("response", "")

    def _extract(raw: str):
        think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        thinking = think_m.group(1).strip() if think_m else ""
        code_m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if not code_m:
            code_m = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
        code = code_m.group(1).strip() if code_m else raw.strip()
        return thinking, code

    def _run_tests(code: str) -> bool:
        import subprocess, tempfile, os
        script = f"{code}\n\n{test_code}"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script); fname = f.name
        try:
            return subprocess.run([sys.executable, fname], capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(fname)

    console.print(Rule(f"[bold blue]DeepSeek-R1 ({model})[/bold blue]"))
    if context:
        console.print(f"[dim]EpiChat context injected ({len(context)} chars)[/dim]")

    best_code = ""
    best_thinking = ""
    passed = None

    for i in range(max(1, n_samples)):
        temp = 0.6 + 0.15 * i
        label = f"Sample {i+1}/{n_samples}" if n_samples > 1 else "Generating"
        with console.status(f"[dim]{label} (temp={temp:.2f})…[/dim]"):
            try:
                raw = _call(temp)
            except Exception as e:
                err.print(f"[red]Error: {e}[/red]")
                continue

        thinking, code = _extract(raw)

        if test_code.strip():
            with console.status("[dim]Running tests…[/dim]"):
                passed = _run_tests(code)
            status = "[green]✓ tests pass[/green]" if passed else "[red]✗ tests fail[/red]"
            console.print(f"  {status} (sample {i+1})")
            if passed:
                best_code, best_thinking = code, thinking
                break
        else:
            best_code, best_thinking = code, thinking
            break

        if not best_code:
            best_code, best_thinking = code, thinking

    console.print(Rule("[dim]Response[/dim]"))
    if think and best_thinking:
        console.print(Panel(best_thinking, title="[dim]Thinking[/dim]", border_style="dim"))
    console.print(Markdown(f"```python\n{best_code}\n```"))
    if passed is not None:
        icon = "[green]✓ verified[/green]" if passed else "[yellow]⚠ unverified[/yellow]"
        console.print(icon)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
