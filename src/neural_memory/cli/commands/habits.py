"""Learned workflow habit commands."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

import typer

from neural_memory.cli._helpers import get_config, get_storage, output_result, run_async

habits_app = typer.Typer(help="Learned workflow habit commands")


@habits_app.command("list")
def habits_list(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List learned workflow habits.

    Shows all habits discovered through action sequence mining,
    including step sequences, frequencies, and confidence scores.

    Examples:
        nmem habits list
        nmem habits list --json
    """

    async def _list() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]

            if json_output:
                output_result(
                    {
                        "habits": [
                            {
                                "name": h.summary or "unnamed",
                                "steps": h.metadata.get("_workflow_actions", []),
                                "frequency": h.metadata.get("_habit_frequency", 0),
                                "confidence": h.metadata.get("_habit_confidence", 0.0),
                                "fiber_id": h.id,
                            }
                            for h in habits
                        ],
                        "count": len(habits),
                    },
                    True,
                )
            else:
                if not habits:
                    typer.echo(
                        "No learned habits yet. Use NeuralMemory tools to build action history.\n"
                        "Run `nmem habits status` to see progress toward pattern detection."
                    )
                    return

                typer.echo(f"Learned habits ({len(habits)}):")
                for h in habits:
                    steps = h.metadata.get("_workflow_actions", [])
                    freq = h.metadata.get("_habit_frequency", 0)
                    conf = h.metadata.get("_habit_confidence", 0.0)
                    typer.echo(f"  {h.summary or 'unnamed'}")
                    typer.echo(f"    Steps: {' → '.join(steps)}")
                    typer.echo(f"    Frequency: {freq}, Confidence: {conf:.2f}")
        finally:
            await storage.close()

    run_async(_list())


@habits_app.command("show")
def habits_show(
    name: Annotated[str, typer.Argument(help="Habit name to show details for")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show details of a specific learned habit.

    Examples:
        nmem habits show recall-edit-test
        nmem habits show recall-edit-test --json
    """

    async def _show() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern") and f.summary == name]

            if not habits:
                typer.echo(f"No habit found with name: {name}")
                raise typer.Exit(code=1)

            habit = habits[0]
            steps = habit.metadata.get("_workflow_actions", [])
            freq = habit.metadata.get("_habit_frequency", 0)
            conf = habit.metadata.get("_habit_confidence", 0.0)

            if json_output:
                output_result(
                    {
                        "name": habit.summary or "unnamed",
                        "steps": steps,
                        "frequency": freq,
                        "confidence": conf,
                        "fiber_id": habit.id,
                        "neuron_count": len(habit.neuron_ids),
                        "synapse_count": len(habit.synapse_ids),
                        "created_at": habit.created_at.isoformat(),
                    },
                    True,
                )
            else:
                typer.echo(f"Habit: {habit.summary or 'unnamed'}")
                typer.echo(f"  Steps: {' → '.join(steps)}")
                typer.echo(f"  Frequency: {freq}")
                typer.echo(f"  Confidence: {conf:.2f}")
                typer.echo(f"  Neurons: {len(habit.neuron_ids)}")
                typer.echo(f"  Synapses: {len(habit.synapse_ids)}")
                typer.echo(f"  Created: {habit.created_at.isoformat()}")
        finally:
            await storage.close()

    run_async(_show())


@habits_app.command("clear")
def habits_clear(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Clear all learned habits.

    Removes all WORKFLOW fibers with _habit_pattern metadata.
    This does not affect action event history.

    Examples:
        nmem habits clear
        nmem habits clear --force
    """

    async def _clear() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            fibers = await storage.get_fibers(limit=1000)
            habits = [f for f in fibers if f.metadata.get("_habit_pattern")]

            if not habits:
                typer.echo("No habits to clear.")
                return

            if not force:
                confirm = typer.confirm(f"Clear {len(habits)} learned habits?")
                if not confirm:
                    typer.echo("Cancelled.")
                    return

            cleared = 0
            for h in habits:
                await storage.delete_fiber(h.id)
                cleared += 1

            typer.echo(f"Cleared {cleared} learned habits.")
        finally:
            await storage.close()

    run_async(_clear())


@habits_app.command("status")
def habits_status(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show progress toward habit detection.

    Displays emerging action patterns that haven't yet reached the
    frequency threshold to become learned habits. Helps you understand
    how close your workflows are to being recognized.

    Examples:
        nmem habits status
        nmem habits status --json
    """

    async def _status() -> None:
        from neural_memory.engine.sequence_mining import mine_sequential_pairs
        from neural_memory.utils.timeutils import utcnow

        config = get_config()
        storage = await get_storage(config)
        try:
            brain = await storage.get_brain(storage._current_brain_id or "")
            if not brain:
                typer.secho("No brain configured.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            brain_config = brain.config
            min_freq = brain_config.habit_min_frequency
            window = brain_config.sequential_window_seconds
            now = utcnow()

            # Get recent action events (same window as learn_habits)
            since = now - timedelta(days=30)
            events = await storage.get_action_sequences(since=since)

            if len(events) < 2:
                if json_output:
                    output_result(
                        {
                            "action_events": len(events),
                            "threshold": min_freq,
                            "emerging_patterns": [],
                            "message": "Not enough action events yet. Keep using NeuralMemory tools.",
                        },
                        True,
                    )
                else:
                    typer.echo(f"Action events (last 30 days): {len(events)}")
                    typer.echo(f"Minimum for habit detection: {min_freq} occurrences")
                    typer.echo()
                    typer.echo("Not enough action events yet. Keep using NeuralMemory tools.")
                return

            # Mine pairs (same logic as learn_habits)
            pairs = mine_sequential_pairs(events, window)

            # Count sessions
            session_ids = {e.session_id for e in events if e.session_id}
            total_sessions = max(len(session_ids), 1)

            # Get existing habits to exclude
            fibers = await storage.get_fibers(limit=1000)
            existing_habits = {
                tuple(f.metadata.get("_workflow_actions", []))
                for f in fibers
                if f.metadata.get("_habit_pattern")
            }

            # Separate into learned vs emerging
            emerging: list[dict[str, str | int | float]] = []
            for pair in pairs:
                steps = (pair.action_a, pair.action_b)
                if steps in existing_habits:
                    continue
                progress = min(pair.count / min_freq, 1.0)
                emerging.append(
                    {
                        "pattern": f"{pair.action_a} -> {pair.action_b}",
                        "count": pair.count,
                        "threshold": min_freq,
                        "progress": round(progress, 2),
                        "remaining": max(0, min_freq - pair.count),
                    }
                )

            # Sort by progress descending
            emerging.sort(key=lambda e: float(e["progress"]), reverse=True)
            # Cap at 10 for readability
            emerging = emerging[:10]

            if json_output:
                output_result(
                    {
                        "action_events": len(events),
                        "sessions": total_sessions,
                        "threshold": min_freq,
                        "existing_habits": len(existing_habits),
                        "emerging_patterns": emerging,
                    },
                    True,
                )
            else:
                typer.echo(f"Action events (last 30 days): {len(events)}")
                typer.echo(f"Sessions: {total_sessions}")
                typer.echo(f"Learned habits: {len(existing_habits)}")
                typer.echo(f"Threshold for habit detection: {min_freq} occurrences")
                typer.echo()

                if not emerging:
                    typer.echo("No emerging patterns detected yet.")
                    typer.echo("Keep using recall, remember, and other tools to build patterns.")
                    return

                typer.echo("Emerging patterns:")
                for e in emerging:
                    bar_filled = round(float(e["progress"]) * 10)
                    bar = "#" * bar_filled + "-" * (10 - bar_filled)
                    status = "READY" if e["remaining"] == 0 else f"{e['remaining']} more needed"
                    typer.echo(
                        f"  {e['pattern']!s:<30} [{bar}] {e['count']}/{e['threshold']} ({status})"
                    )

                ready_count = sum(1 for e in emerging if e["remaining"] == 0)
                if ready_count > 0:
                    typer.echo(
                        f"\n{ready_count} pattern(s) ready — run `nmem consolidate --strategy learn_habits` to materialize."
                    )
        finally:
            await storage.close()

    run_async(_status())
