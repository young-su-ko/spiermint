from __future__ import annotations

import csv
from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="Score target-conditioned binder PLLs with MINT.",
    no_args_is_help=True,
)


@app.command()
def main(
    target: str | None = typer.Option(None, help="Target protein sequence."),
    binder: str | None = typer.Option(None, help="Binder protein sequence."),
    input_csv: Path | None = typer.Option(None, help="CSV with target and binder columns."),
    output_csv: Path | None = typer.Option(None, help="Output CSV path."),
    target_col: str = typer.Option("target_sequence", help="Target column for CSV scoring."),
    binder_col: str = typer.Option("binder_sequence", help="Binder column for CSV scoring."),
    score_col: str = typer.Option("spiermint_score", help="Output score column for CSV scoring."),
    approx: bool = typer.Option(False, help="Use Gordon-style approximate binder PLL."),
    device: str | None = typer.Option(None, help="Device override, e.g. cpu or cuda:0."),
    mask_batch_size: int = typer.Option(64, help="Exact PLL mask batch size."),
) -> None:
    if input_csv is None and (target is None or binder is None):
        raise typer.BadParameter("Pass --target and --binder, or pass --input-csv.")

    try:
        from .scoring import MINTScorer

        scorer = MINTScorer(
            device=device,
            mask_batch_size=mask_batch_size,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if input_csv is not None:
        output = output_csv or input_csv.with_name(f"{input_csv.stem}_spiermint.csv")
        try:
            score_csv(
                scorer,
                input_csv=input_csv,
                output_csv=output,
                target_col=target_col,
                binder_col=binder_col,
                score_col=score_col,
                approx=approx,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f"output_csv={output}")
        return

    assert target is not None and binder is not None
    result = scorer.score(target, binder, approx=approx, return_details=True)
    typer.echo(f"score={result.score}")
    typer.echo(f"pll={result.pll}")
    typer.echo(f"length={result.length}")
    typer.echo(f"mode={result.mode}")


def score_csv(
    scorer,
    input_csv: Path,
    output_csv: Path,
    target_col: str,
    binder_col: str,
    score_col: str,
    approx: bool,
) -> None:
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} is missing a header row.")
        missing = [column for column in (target_col, binder_col) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
        rows = list(reader)

    fieldnames = list(reader.fieldnames)
    if score_col not in fieldnames:
        fieldnames.append(score_col)

    for row in rows:
        row[score_col] = scorer.score(row[target_col], row[binder_col], approx=approx)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
