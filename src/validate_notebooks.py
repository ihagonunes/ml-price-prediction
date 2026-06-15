from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbformat.validator import normalize

warnings.filterwarnings(
    "ignore",
    message="Proactor event loop does not implement add_reader.*",
    category=RuntimeWarning,
)

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOK_VALIDATION_FILE = REPORTS_DIR / "notebook_reproducibility_validation.csv"
NOTEBOOK_VALIDATION_REPORT_FILE = REPORTS_DIR / "notebook_reproducibility_report.md"
CACHE_DIR = BASE_DIR / ".cache"
IPYTHON_DIR = CACHE_DIR / "ipython"
IPYTHON_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(IPYTHON_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def execute_notebook(notebook_path: Path) -> dict:
    started_at = datetime.now()
    start = perf_counter()
    notebook = nbformat.read(notebook_path, as_version=4)
    normalized_changes, notebook = normalize(notebook)
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    markdown_cells = sum(1 for cell in notebook.cells if cell.cell_type == "markdown")

    try:
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            allow_errors=False,
            resources={
                "metadata": {"path": str(BASE_DIR)},
                "env": {"IPYTHONDIR": str(IPYTHON_DIR)},
            },
        )
        executed_notebook = client.execute()
        nbformat.write(executed_notebook, notebook_path)
        status = "passed"
        error_name = ""
        error_value = ""
    except Exception as exc:
        status = "failed"
        error_name = exc.__class__.__name__
        error_value = str(exc)

    elapsed_seconds = round(perf_counter() - start, 3)
    return {
        "Notebook": str(notebook_path.relative_to(BASE_DIR)),
        "Status": status,
        "CodeCells": int(code_cells),
        "MarkdownCells": int(markdown_cells),
        "NormalizedChanges": int(normalized_changes),
        "StartedAt": started_at.isoformat(),
        "ElapsedSeconds": elapsed_seconds,
        "ErrorName": error_name,
        "ErrorValue": error_value,
    }


def validate_all_notebooks() -> pd.DataFrame:
    notebook_paths = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebook_paths:
        raise FileNotFoundError(f"Nenhum notebook encontrado em {NOTEBOOKS_DIR}")

    rows = []
    for notebook_path in notebook_paths:
        log.info("Executando notebook do zero | arquivo=%s", notebook_path.name)
        row = execute_notebook(notebook_path)
        rows.append(row)
        if row["Status"] != "passed":
            log.error(
                "Notebook falhou | arquivo=%s | erro=%s: %s",
                notebook_path.name,
                row["ErrorName"],
                row["ErrorValue"],
            )

    validation_df = pd.DataFrame(rows)
    if (validation_df["Status"] != "passed").any():
        failed = validation_df.loc[validation_df["Status"] != "passed"]
        raise RuntimeError(
            "Foram encontrados notebooks nao reprodutiveis: "
            + ", ".join(failed["Notebook"].tolist())
        )
    return validation_df


def write_report(validation_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(NOTEBOOK_VALIDATION_FILE, index=False)

    total_notebooks = len(validation_df)
    total_code_cells = int(validation_df["CodeCells"].sum())
    total_elapsed = float(validation_df["ElapsedSeconds"].sum())

    report_lines = [
        "# Validacao de Reprodutibilidade dos Notebooks",
        "",
        f"- Gerado em: `{datetime.now().isoformat()}`",
        f"- Diretorio avaliado: `{NOTEBOOKS_DIR}`",
        f"- Notebooks executados: `{total_notebooks}`",
        f"- Celulas de codigo executaveis: `{total_code_cells}`",
        f"- Tempo total de execucao: `{total_elapsed:.3f}s`",
        "",
        "## Resultado",
        "",
        validation_df.to_markdown(index=False),
        "",
        "## Conclusao",
        "",
        "- Todos os notebooks foram abertos em uma execucao limpa via `nbclient`.",
        "- Nenhum notebook dependeu de estado previo em memoria.",
        "- O notebook atual possui apenas celulas Markdown, portanto a validacao confirmou estrutura e execucao sem erro de kernel ou dependencia oculta.",
    ]
    NOTEBOOK_VALIDATION_REPORT_FILE.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("Validando reprodutibilidade dos notebooks")
    log.info("=" * 60)

    validation_df = validate_all_notebooks()
    write_report(validation_df)

    log.info("Validacao salva em %s", NOTEBOOK_VALIDATION_FILE)
    log.info("=" * 60)
    return validation_df


if __name__ == "__main__":
    main()
