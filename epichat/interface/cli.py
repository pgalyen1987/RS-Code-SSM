from __future__ import annotations
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box
from rich.text import Text
from rich.prompt import Prompt

from core.knowledge_graph import KnowledgeGraph
from language.query_engine import QueryEngine, QueryResult


console = Console()


class EpistemeShell:
    """
    EPISTEME Code Intelligence Shell — Commands
    -------------------------------------------
    <question>             — query the knowledge graph
    why <question>         — trace causal/justification chain
    gaps <topic>           — list epistemic gaps for a topic
    explain <concept>      — deep explanation with code examples
    pattern <problem>      — find relevant design patterns / algorithms
    complexity <algo>      — show time/space complexity
    tradeoffs <topic>      — show known tradeoffs
    examples <topic>       — show code examples (optional: examples <topic> kotlin)
    security <topic>       — show cybersecurity-relevant knowledge
    assert <statement>     — add a new belief to the graph
    revise <statement>     — revise beliefs related to a statement
    stats                  — show knowledge graph statistics
    save                   — persist the knowledge graph
    help                   — show this help message
    exit / quit            — exit EPISTEME
    """

    BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  EPISTEME  ·  Code Intelligence & Justified Belief System        ║
║  Algorithms · OOP · Design Patterns · Kotlin · Security          ║
╚══════════════════════════════════════════════════════════════════╝
Type 'help' for commands, 'exit' to quit.
"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg     = knowledge_graph
        self.engine = QueryEngine(knowledge_graph)

    # ------------------------------------------------------------------
    # Shell loop
    # ------------------------------------------------------------------

    def run(self):
        console.print(self.BANNER, style="bold cyan")
        while True:
            try:
                raw = Prompt.ask("\n[bold green]EPISTEME[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                self._cmd_exit()
                break

            if not raw:
                continue

            cmd, _, rest = raw.partition(" ")
            cmd = cmd.lower()

            dispatch = {
                "exit":       self._cmd_exit,
                "quit":       self._cmd_exit,
                "help":       self._cmd_help,
                "stats":      self._cmd_stats,
                "save":       self._cmd_save,
                "why":        lambda: self._cmd_why(rest),
                "gaps":       lambda: self._cmd_gaps(rest),
                "explain":    lambda: self._cmd_explain(rest),
                "pattern":    lambda: self._cmd_pattern(rest),
                "complexity": lambda: self._cmd_complexity(rest),
                "tradeoffs":  lambda: self._cmd_tradeoffs(rest),
                "examples":   lambda: self._cmd_examples(rest),
                "security":   lambda: self._cmd_security(rest),
                "assert":     lambda: self._cmd_assert(rest),
                "revise":     lambda: self._cmd_revise(rest),
            }

            if cmd in dispatch:
                result = dispatch[cmd]()
                if result is False:
                    break
            else:
                self._cmd_query(raw)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_query(self, question: str):
        with console.status("[bold yellow]Querying…[/bold yellow]"):
            result = self.engine.query(question)
        self._render_result(result)

    def _cmd_why(self, question: str):
        if not question:
            console.print("[red]Usage: why <question>[/red]"); return
        with console.status("[bold yellow]Tracing justification chain…[/bold yellow]"):
            result = self.engine.ask_why(question)
        self._render_result(result, show_full_chain=True)

    def _cmd_explain(self, concept: str):
        if not concept:
            console.print("[red]Usage: explain <concept>[/red]"); return
        with console.status("[bold yellow]Explaining…[/bold yellow]"):
            result = self.engine.explain_concept(concept)
        self._render_result(result)

        # Also show code examples
        examples = self.engine.find_code_examples(concept)
        for eu in examples[:2]:
            self._render_code_eu(eu)

    def _cmd_pattern(self, problem: str):
        if not problem:
            console.print("[red]Usage: pattern <problem description>[/red]"); return
        with console.status("[bold yellow]Finding patterns…[/bold yellow]"):
            patterns = self.engine.find_pattern(problem)
        if not patterns:
            console.print("[yellow]No matching patterns found.[/yellow]"); return

        table = Table(title=f"Patterns for: {problem[:60]}", box=box.ROUNDED)
        table.add_column("Type",        style="bold", width=14)
        table.add_column("Proposition", width=60)
        table.add_column("Conf",        justify="right", width=6)
        table.add_column("When to use", width=30)

        for eu in patterns[:6]:
            table.add_row(
                eu.knowledge_type.value,
                eu.proposition[:60],
                f"{eu.confidence:.0%}",
                (eu.when_to_use or "")[:30],
            )
        console.print(table)

        # Show code for first pattern with snippet
        for eu in patterns:
            if eu.code_snippet:
                self._render_code_eu(eu)
                break

    def _cmd_complexity(self, algorithm: str):
        if not algorithm:
            console.print("[red]Usage: complexity <algorithm>[/red]"); return
        with console.status("[bold yellow]Looking up complexity…[/bold yellow]"):
            eu = self.engine.get_complexity(algorithm)
        if not eu:
            console.print(f"[yellow]No complexity data found for '{algorithm}'.[/yellow]"); return

        table = Table(title=f"Complexity: {eu.proposition[:50]}", box=box.SIMPLE)
        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_row("Proposition",    eu.proposition[:80])
        table.add_row("Time",           eu.time_complexity  or "N/A")
        table.add_row("Space",          eu.space_complexity or "N/A")
        table.add_row("When to use",    eu.when_to_use      or "N/A")
        table.add_row("Tradeoffs",      eu.tradeoffs        or "N/A")
        table.add_row("Confidence",     f"{eu.confidence:.1%}")
        console.print(table)

        if eu.code_snippet:
            self._render_code_eu(eu)

    def _cmd_tradeoffs(self, topic: str):
        if not topic:
            console.print("[red]Usage: tradeoffs <topic>[/red]"); return
        with console.status("[bold yellow]Finding tradeoffs…[/bold yellow]"):
            eus = self.engine.find_tradeoffs(topic)
        if not eus:
            console.print(f"[yellow]No tradeoff knowledge found for '{topic}'.[/yellow]"); return

        lines = []
        for eu in eus[:5]:
            if eu.tradeoffs:
                lines.append(f"[bold]{eu.proposition[:60]}[/bold]")
                lines.append(f"  → {eu.tradeoffs}")
                lines.append("")
        console.print(Panel(
            "\n".join(lines).strip(),
            title=f"[bold yellow]Tradeoffs: {topic}[/bold yellow]",
            border_style="yellow",
        ))

    def _cmd_examples(self, args: str):
        parts  = args.split()
        if not parts:
            console.print("[red]Usage: examples <topic> [language][/red]"); return
        language = parts[-1] if len(parts) > 1 else None
        topic    = " ".join(parts[:-1]) if language else args

        with console.status("[bold yellow]Finding code examples…[/bold yellow]"):
            eus = self.engine.find_code_examples(topic, language=language)
        if not eus:
            console.print(f"[yellow]No code examples found for '{topic}'{' in ' + language if language else ''}.[/yellow]")
            return

        console.print(f"[bold]Code examples for: {topic}[/bold]  ({len(eus)} found)")
        for eu in eus[:3]:
            self._render_code_eu(eu)

    def _cmd_security(self, topic: str):
        if not topic:
            console.print("[red]Usage: security <topic>[/red]"); return
        with console.status("[bold yellow]Checking security knowledge…[/bold yellow]"):
            eus = self.engine.security_check(topic)
        if not eus:
            # Fall back to general query filtered by cybersecurity keywords
            eus = self.kg.find_similar(topic, top_k=8, threshold=0.35,
                                       domain="cybersecurity")
        if not eus:
            console.print(f"[yellow]No security knowledge found for '{topic}'.[/yellow]"); return

        table = Table(title=f"Security: {topic[:50]}", box=box.ROUNDED)
        table.add_column("Principle", width=70)
        table.add_column("Conf", justify="right", width=6)

        for eu in eus[:6]:
            table.add_row(eu.proposition[:70], f"{eu.confidence:.0%}")
        console.print(table)

    def _cmd_gaps(self, topic: str):
        if not topic:
            console.print("[red]Usage: gaps <topic>[/red]"); return
        gaps = self.engine.what_dont_i_know(topic)
        if not gaps:
            console.print(f"[green]No known gaps for '{topic}'.[/green]"); return
        console.print(Panel(
            "\n".join(f"• {g}" for g in gaps),
            title=f"[bold red]Epistemic Gaps: {topic}[/bold red]",
            border_style="red",
        ))

    def _cmd_assert(self, statement: str):
        if not statement:
            console.print("[red]Usage: assert <statement>[/red]"); return
        with console.status("[bold yellow]Evaluating belief…[/bold yellow]"):
            judgment = self.engine.assert_belief(statement)
        if judgment.accepted:
            console.print(f"[green]✓ Accepted[/green] (conf {judgment.confidence:.1%})")
        else:
            console.print(f"[red]✗ Rejected[/red] (conf {judgment.confidence:.1%})")
            if judgment.gaps:
                console.print(f"  Gap: {judgment.gaps[0]}")

    def _cmd_revise(self, statement: str):
        if not statement:
            console.print("[red]Usage: revise <statement>[/red]"); return
        self.engine.revise_belief(
            statement, new_evidence="User-provided revision", evidence_strength=-0.3
        )
        console.print(f"[yellow]Beliefs related to '{statement[:50]}' revised downward.[/yellow]")

    def _cmd_stats(self):
        stats = self.kg.stats()
        table = Table(title="Knowledge Graph Statistics", box=box.ROUNDED)
        table.add_column("Metric",  style="bold")
        table.add_column("Value",   justify="right")
        table.add_row("Total EUs",             str(stats["total_units"]))
        table.add_row("Total edges",           str(stats["total_edges"]))
        table.add_row("Total stored (all-time)", str(stats["total_stored"]))
        table.add_row("Total pruned",          str(stats["total_pruned"]))
        table.add_row("Avg confidence",        f"{stats['avg_confidence']:.2%}")
        console.print(table)

        if stats["domains"]:
            dom = Table(title="Domains", box=box.SIMPLE)
            dom.add_column("Domain")
            dom.add_column("EUs", justify="right")
            for d, cnt in sorted(stats["domains"].items(), key=lambda x: -x[1]):
                dom.add_row(d, str(cnt))
            console.print(dom)

    def _cmd_save(self):
        self.kg.save()
        console.print("[green]Graph saved.[/green]")

    def _cmd_help(self):
        console.print(Panel(
            self.__doc__ or "",
            title="[bold]EPISTEME Help[/bold]",
            border_style="blue",
        ))

    def _cmd_exit(self):
        console.print("[bold cyan]Goodbye.[/bold cyan]")
        return False

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_result(self, result: QueryResult, show_full_chain: bool = False):
        if result.confidence >= 0.80:
            conf_style = "bold green"
        elif result.confidence >= 0.50:
            conf_style = "bold yellow"
        else:
            conf_style = "bold red"

        conf_bar = self._bar(result.confidence)

        console.print(Panel(
            Text(result.answer),
            title=(f"[bold]Answer[/bold] — "
                   f"[{conf_style}]{result.confidence:.1%}[/{conf_style}] {conf_bar}"),
            subtitle=f"Type: {result.knowledge_type}",
            border_style="cyan",
            padding=(0, 1),
        ))

        if result.justification_chain:
            chain = result.justification_chain if show_full_chain else result.justification_chain[:4]
            lines = []
            for i, eu in enumerate(chain):
                conn = "└─" if i == len(chain) - 1 else "├─"
                ktype = eu.knowledge_type.value[:4]
                lines.append(f"{conn} [{ktype}|{eu.confidence:.2f}] {eu.proposition[:80]}")
            console.print(Panel(
                "\n".join(lines),
                title=f"[bold]Justification Chain[/bold] ({len(result.justification_chain)})",
                border_style="blue",
                padding=(0, 1),
            ))

        if result.gaps:
            console.print(Panel(
                "\n".join(f"• {g}" for g in result.gaps[:5]),
                title="[bold red]Epistemic Gaps[/bold red]",
                border_style="red",
                padding=(0, 1),
            ))

        if result.sources:
            console.print(f"[dim]Sources: {', '.join(result.sources[:4])}[/dim]")

        if result.competing_views:
            console.print(
                f"[dim yellow]{len(result.competing_views)} competing view(s) "
                f"— use 'explain' for deeper analysis[/dim yellow]"
            )

    def _render_code_eu(self, eu):
        """Render an EU that has a code snippet."""
        lang = (eu.language or "text").lower()
        lang_map = {"kotlin": "kotlin", "python": "python", "java": "java",
                    "javascript": "javascript", "typescript": "typescript"}
        lexer = lang_map.get(lang, "text")

        meta = []
        if eu.time_complexity:
            meta.append(f"Time: {eu.time_complexity}")
        if eu.space_complexity:
            meta.append(f"Space: {eu.space_complexity}")
        if eu.when_to_use:
            meta.append(f"Use when: {eu.when_to_use[:60]}")
        if eu.tradeoffs:
            meta.append(f"Tradeoffs: {eu.tradeoffs[:60]}")

        subtitle = "  |  ".join(meta) if meta else None

        console.print(Panel(
            Syntax(eu.code_snippet, lexer, theme="monokai", line_numbers=True),
            title=f"[bold magenta]{eu.language or 'Code'} Example[/bold magenta]  "
                  f"[dim]{eu.proposition[:60]}[/dim]",
            subtitle=subtitle,
            border_style="magenta",
        ))

    @staticmethod
    def _bar(value: float, width: int = 10) -> str:
        filled = round(value * width)
        return "█" * filled + "░" * (width - filled)
