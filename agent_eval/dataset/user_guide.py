"""Path C — Manual wizard. Path B — Semi-auto curation TUI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.table import Table

from agent_eval.dataset.schema import EvalDataset, EvalExample
from agent_eval.discovery.registry import ComponentRegistry


class UserWizard:
    """Path C — manual wizard. Walks user through 5 mandatory categories
    (typical / edge_case / adversarial / failing / multi_turn) with
    optional LLM few-shot expansion."""

    def __init__(self, registry: ComponentRegistry, console: Console | None = None, llm: Any | None = None):
        self.registry = registry
        self.console = console or Console()
        self.llm = llm

    def run(self, expand_with_llm: bool = True, expand_multiplier: int = 3) -> EvalDataset:
        import click

        c = self.console
        c.print("[bold]agent-eval — manual dataset wizard[/bold]")
        self._render_registry_table()
        self._show_example_queries()
        c.print("\nFor each example, provide a query and (optionally) keywords/reference.")
        examples: list[EvalExample] = []
        for category, qtype in (
            ("typical", "general"),
            ("edge_case", "edge_case"),
            ("adversarial", "adversarial"),
            ("failing", "general"),
            ("multi_turn", "general"),
        ):
            c.print(f"\n[bold cyan]Category: {category}[/bold cyan]")
            n = click.prompt(f"How many '{category}' examples?", type=int, default=2)
            for i in range(n):
                query = click.prompt(f"  [{i+1}/{n}] Query")
                kw = click.prompt("       Expected keywords (comma-separated, blank=none)", default="", show_default=False)
                ref = click.prompt("       Reference answer (optional, blank=skip)", default="", show_default=False)
                cx = click.prompt("       Complexity [simple/medium/complex]", type=click.Choice(["simple", "medium", "complex"]), default="medium")
                examples.append(EvalExample(
                    input={"query": query},
                    reference_answer=ref or None,
                    expected_answer_keywords=[k.strip() for k in kw.split(",") if k.strip()],
                    query_type=qtype,
                    complexity=cx,
                    created_by="manual",
                    tags=[category],
                ))

        if expand_with_llm and self.llm is not None and click.confirm(
            f"\nExpand with {expand_multiplier}× LLM variants? (~{len(examples) * expand_multiplier} new examples)",
            default=True,
        ):
            try:
                expanded = asyncio.run(self.expand_with_llm(examples, self.llm, expand_multiplier))
                examples.extend(expanded)
                c.print(f"[green]Generated {len(expanded)} variants.[/green]")
            except Exception as e:
                c.print(f"[red]LLM expansion failed: {e}. Keeping originals only.[/red]")

        # Show cost estimate.
        try:
            from agent_eval.dataset.auto_generator import AutoDatasetGenerator

            ds = EvalDataset(
                name="manual_wizard",
                description="Hand-curated via UserWizard",
                examples=examples,
                component_registry_hash=self.registry.hash(),
            )
            est = AutoDatasetGenerator(self.registry).estimate_evaluation_cost(ds)
            c.print(f"\n[yellow]Estimated evaluation cost: ~${est.estimated_usd} ({est.estimated_judge_calls} judge calls).[/yellow]")
            if not click.confirm("Proceed?", default=True):
                c.print("[red]Aborted.[/red]")
                raise click.Abort()
            return ds
        except Exception:
            return EvalDataset(name="manual_wizard", description="Hand-curated", examples=examples,
                                component_registry_hash=self.registry.hash())

    def _render_registry_table(self) -> None:
        c = self.console
        t = Table(title="Discovered components")
        for col in ("Type", "Name"):
            t.add_column(col)
        for a in self.registry.agents:
            t.add_row("agent", a.name)
        for tool in self.registry.tools:
            t.add_row("tool", tool.name)
        for srv in self.registry.mcp_servers:
            t.add_row("mcp_server", srv.name)
        for mb in self.registry.memory_backends:
            t.add_row("memory_backend", mb.type)
        c.print(t)

    def _show_example_queries(self) -> None:
        c = self.console
        c.print("\n[bold]Examples of good test cases per agent type:[/bold]")
        if not self.registry.agents:
            return
        for agent in self.registry.agents[:3]:
            t = (agent.agent_type or "").lower()
            if "search" in t or "search" in agent.name.lower():
                c.print(f"  • {agent.name} (search): 'What is the capital of France?', 'Latest iPhone?'")
            elif "research" in t or "research" in agent.name.lower():
                c.print(f"  • {agent.name} (research): 'Compare A vs B on metric C in year D.'")
            elif "orchestrator" in t or "router" in t:
                c.print(f"  • {agent.name} (orchestrator): mix simple + complex queries to test routing.")
            else:
                c.print(f"  • {agent.name}: typical and edge-case queries appropriate for its purpose.")

    @staticmethod
    async def expand_with_llm(seeds: list[EvalExample], llm: Any, multiplier: int = 3) -> list[EvalExample]:
        """Few-shot LLM expansion: for each seed, generate `multiplier` paraphrased variants."""
        if not seeds or multiplier < 1:
            return []
        prompt = (
            "Given the following test examples, generate paraphrased variants that test the same intent "
            "but differ in wording, length, or framing. Return a JSON list of objects, each with keys: "
            "`query` (string), `complexity` (simple|medium|complex), `query_type` (general|search|research|adversarial|edge_case).\n\n"
            "Original examples:\n"
            + "\n".join(f"- {json.dumps({'query': s.input.get('query', ''), 'complexity': s.complexity, 'query_type': s.query_type})}" for s in seeds[:20])
            + f"\n\nFor each example above, generate exactly {multiplier} variants. Return ONLY the JSON list, no commentary."
        )
        try:
            if hasattr(llm, "ainvoke"):
                msg = await llm.ainvoke(prompt)
            else:
                msg = llm.invoke(prompt)
            text = getattr(msg, "content", str(msg))
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            if not isinstance(data, list):
                return []
            out = []
            for item in data:
                if not isinstance(item, dict) or "query" not in item:
                    continue
                out.append(EvalExample(
                    input={"query": item["query"]},
                    complexity=item.get("complexity", "medium"),
                    query_type=item.get("query_type", "general"),
                    created_by="manual",
                    tags=["llm_expanded"],
                ))
            return out
        except Exception:
            return []


class SemiAutoWizard:
    """Path B — semi-auto curation. Generates candidates from auto-gen + harvester, then
    walks the user through accept/reject/edit decisions interactively.
    """

    def __init__(
        self,
        registry: ComponentRegistry,
        langsmith_client: Any | None = None,
        llm: Any | None = None,
        console: Console | None = None,
        project_name: str | None = None,
    ):
        self.registry = registry
        self.client = langsmith_client
        self.llm = llm
        self.console = console or Console()
        self.project_name = project_name

    def run(self, n_candidates: int = 30) -> EvalDataset:
        import click

        from agent_eval.dataset.auto_generator import AutoDatasetGenerator

        c = self.console
        c.print("[bold]agent-eval — semi-auto curation wizard (PATH B)[/bold]")
        c.print(f"Generating ~{n_candidates} candidate examples...")
        gen = AutoDatasetGenerator(self.registry, langsmith_client=self.client, llm=self.llm)
        candidate_ds = asyncio.run(gen.generate(n_examples=n_candidates, project_name=self.project_name))
        c.print(f"[green]Got {len(candidate_ds.examples)} candidates. Reviewing one by one.[/green]")
        c.print("Actions: [a]ccept, [r]eject, [e]dit query, [k]eywords, [s]kip remaining (accept all rest)")

        kept: list[EvalExample] = []
        skip_remaining = False
        for i, ex in enumerate(candidate_ds.examples):
            if skip_remaining:
                kept.append(ex)
                continue
            c.print(f"\n[bold cyan]Candidate {i+1}/{len(candidate_ds.examples)}[/bold cyan]")
            c.print(f"  query: {ex.input.get('query', ex.input)}")
            c.print(f"  type: {ex.query_type}  complexity: {ex.complexity}  source: {ex.created_by}")
            if ex.expected_answer_keywords:
                c.print(f"  keywords: {ex.expected_answer_keywords}")
            action = click.prompt("  Action [a/r/e/k/s]", default="a", show_default=False).lower().strip()
            if action == "r":
                continue
            if action == "s":
                kept.append(ex)
                skip_remaining = True
                continue
            if action == "e":
                new_q = click.prompt("  New query", default=ex.input.get("query", ""))
                ex.input["query"] = new_q
            if action == "k":
                kw = click.prompt("  Keywords (comma-separated)", default=",".join(ex.expected_answer_keywords))
                ex.expected_answer_keywords = [k.strip() for k in kw.split(",") if k.strip()]
            kept.append(ex)

        validation = gen.validate_dataset(candidate_ds.model_copy(update={"examples": kept}))
        if not validation.passed:
            c.print(f"[yellow]Validation issues:[/yellow] {validation.issues}")

        return EvalDataset(
            name="semi_auto_curated",
            description=f"Semi-auto curated from {len(candidate_ds.examples)} candidates ({len(kept)} kept)",
            examples=kept,
            component_registry_hash=self.registry.hash(),
        )
