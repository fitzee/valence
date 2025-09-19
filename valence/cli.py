"""Typer-based CLI for Valence."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

from valence.evalrun import run_evaluation
from valence.report import generate_report
from valence.ci import ci_main

app = typer.Typer(
    name="valence",
    help="Adaptive, immune-inspired evaluation for LLMs/agents",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    model: str = typer.Option(
        "stub",
        "--model",
        help="Model to evaluate (e.g., stub, openai:gpt-4o, anthropic:claude-3-haiku, azure-openai:gpt-4)"
    ),
    seeds: Path = typer.Option(..., "--seeds", help="Path to seeds JSON file"),
    packs: Path = typer.Option(..., "--packs", help="Path to packs directory or file"),
    out: Path = typer.Option(..., "--out", help="Output directory for results"),
    max_gens: int = typer.Option(1, "--max-gens", help="Maximum mutation generations"),
    mutations_per_failure: int = typer.Option(
        4, "--mutations-per-failure", help="Number of mutations per failing prompt"
    ),
    memory: Optional[Path] = typer.Option(None, "--memory", help="Path to memory JSONL file"),
    llm_mutations: bool = typer.Option(
        False, "--llm-mutations", help="Enable LLM-based semantic mutations (requires API keys)"
    ),
    mutation_model: str = typer.Option(
        "openai:gpt-4o-mini", "--mutation-model", help="Model to use for LLM mutations"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run an evaluation suite."""
    setup_logging(verbose)
    
    try:
        if not seeds.exists():
            console.print(f"[red]Error: Seeds file not found: {seeds}[/red]")
            sys.exit(1)
        
        if not packs.exists():
            console.print(f"[red]Error: Packs path not found: {packs}[/red]")
            sys.exit(1)
        
        console.print(f"[blue]Starting evaluation run...[/blue]")
        console.print(f"  Model: {model}")
        console.print(f"  Seeds: {seeds}")
        console.print(f"  Packs: {packs}")
        console.print(f"  Output: {out}")
        
        if llm_mutations:
            console.print(f"  [yellow]LLM mutations enabled with: {mutation_model}[/yellow]")
        else:
            console.print(f"  Mutations: Basic only (use --llm-mutations for semantic)")
        
        # Show available API keys if using real providers
        if ":" in model:
            provider = model.split(":")[0]
            if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
                console.print(f"  [green]✓ OpenAI API key found[/green]")
            elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                console.print(f"  [green]✓ Anthropic API key found[/green]")
            elif provider == "azure-openai":
                has_all = all([
                    os.environ.get("AZURE_OPENAI_KEY"),
                    os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                ])
                if has_all:
                    console.print(f"  [green]✓ Azure OpenAI credentials found[/green]")
                else:
                    console.print(f"  [yellow]! Missing Azure OpenAI credentials[/yellow]")
        
        metadata = run_evaluation(
            model_name=model,
            seeds_path=seeds,
            packs_path=packs,
            output_dir=out,
            memory_path=memory,
            max_generations=max_gens,
            mutations_per_failure=mutations_per_failure,
            use_llm_mutations=llm_mutations,
            mutation_model=mutation_model,
        )
        
        console.print(f"\n[green]✓ Evaluation complete![/green]")
        console.print(f"  Run ID: {metadata.run_id}")
        console.print(f"  Total prompts: {metadata.total_prompts}")
        console.print(f"  Failures: {metadata.total_failures}")
        console.print(f"  Passes: {metadata.total_passes}")
        console.print(f"  Errors: {metadata.total_errors}")
        
        if metadata.total_prompts > 0:
            pass_rate = metadata.total_passes / metadata.total_prompts * 100
            console.print(f"  Pass rate: {pass_rate:.1f}%")
        
        console.print(f"\n[blue]Results saved to: {out}[/blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(2)


@app.command()
def report(
    input: Path = typer.Option(..., "--in", help="Input directory with evaluation results"),
    output: Path = typer.Option(..., "--out", help="Output path for HTML report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Build a static HTML report."""
    setup_logging(verbose)
    
    try:
        if not input.exists():
            console.print(f"[red]Error: Input directory not found: {input}[/red]")
            sys.exit(1)
        
        console.print(f"[blue]Generating report...[/blue]")
        console.print(f"  Input: {input}")
        console.print(f"  Output: {output}")
        
        generate_report(run_dir=input, output_path=output)
        
        console.print(f"\n[green]✓ Report generated![/green]")
        console.print(f"  View report: {output}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Report generation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(3)


@app.command()
def ci(
    run_dir: Path = typer.Argument(..., help="Evaluation run directory"),
    baseline_dir: Optional[Path] = typer.Argument(None, help="Baseline run directory for comparison"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="CI configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run CI gate checks on evaluation results."""
    setup_logging(verbose)
    
    try:
        from valence.ci import check_ci_gates, generate_ci_report
        
        console.print(f"[blue]Running CI gate checks...[/blue]")
        console.print(f"  Run: {run_dir}")
        if baseline_dir:
            console.print(f"  Baseline: {baseline_dir}")
        
        results = check_ci_gates(run_dir, baseline_dir, config_path)
        
        # Generate report
        report_path = run_dir / "ci_report.txt"
        generate_ci_report(results, report_path)
        
        # Print summary
        if results["summary"]["passed"]:
            console.print("[green]✅ All CI gates passed[/green]")
        else:
            console.print("[red]❌ CI gate violations found:[/red]")
            for violation in results["summary"]["violations"]:
                console.print(f"   {violation}")
        
        console.print(f"\n[blue]CI report saved to: {report_path}[/blue]")
        
        # Exit with appropriate code
        if not results["summary"]["passed"]:
            raise typer.Exit(code=1)
            
    except Exception as e:
        console.print(f"[red]CI check failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Show version information."""
    from valence import __version__
    
    console.print(f"Valence v{__version__}")


if __name__ == "__main__":
    app()