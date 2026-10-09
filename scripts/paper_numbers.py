"""Write the paper's numbers as LaTeX macros from the final-experiment outputs (#130).

    python -m scripts.paper_numbers [--root DIR] [--mapping YAML] [--out numbers.tex]
                                    [--list] [--strict]

The mapping (``scripts/data/paper_numbers.yaml``) gives every macro a source file
(relative to --root, the ``final-experiments/`` folder), a way to extract one value
and a format. Extraction is one of:

- ``json: a.b.0.c``: a key path (integer parts index lists);
- ``csv: {column, where, agg}``: rows filtered by ``where`` (a value, a list of
  values, ``"*"`` for non-empty, or ``{not: ...}``), then ``agg``: ``value`` (exactly
  one row), ``sum``, ``mean``, ``sd``, ``median``, ``q1``, ``q3``, ``p95``, ``min``,
  ``max``, ``count`` or ``share`` (matching rows over the rows matching ``of``);
- ``regex: PATTERN``: searched in the text file; ``group`` is a group number, or
  ``[numerator, denominator]`` for a ratio;
- ``expr: A - B``: arithmetic over other macros' raw values (``emit: false`` keeps a
  helper value out of the output).

An entry with ``each`` (a list of substitution dicts, or a list of such lists for a
product) expands into one macro per substitution; ``$name`` / ``${name}`` in its
strings is replaced. A missing file, key, column or match renders ``\\textbf{??}``
and is listed on stderr; the script never fails on a missing input (``--strict``
makes the exit code 1 when anything is missing).
"""

from __future__ import annotations

import argparse
import ast
import csv
import decimal
import itertools
import json
import math
import operator
import re
import statistics
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

DEFAULT_ROOT = Path.home() / "marc-2026-work/experiments/ablations/final-system/final-experiments"
DEFAULT_MAPPING = Path(__file__).resolve().parent / "data" / "paper_numbers.yaml"
# The paper itself is written elsewhere; the macros live with the paper material in the repo.
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "paper" / "final-submission" / "numbers.tex"
MISSING = r"\textbf{??}"
NAME_RE = re.compile(r"^[A-Za-z]+$")
KINDS = ("json", "csv", "regex", "expr")
LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}

Value = Union[int, float, str]


class Missing(Exception):
    """The value cannot be produced (yet); the message says why."""


@dataclass
class Macro:
    name: str
    population: str
    kind: str
    spec: Union[str, dict]
    source: Optional[str] = None
    fmt: dict = field(default_factory=dict)
    emit: bool = True
    group: Union[int, list, None] = None


@dataclass
class Result:
    macro: Macro
    text: Optional[str] = None
    value: Optional[Value] = None
    reason: Optional[str] = None


# Mapping


def _substitute(item, subs: dict):
    if isinstance(item, str):
        return string.Template(item).safe_substitute(subs)
    if isinstance(item, list):
        return [_substitute(part, subs) for part in item]
    if isinstance(item, dict):
        return {_substitute(key, subs): _substitute(value, subs) for key, value in item.items()}
    return item


def expand(entry: dict) -> list[dict]:
    """One entry per substitution in ``each`` (a product over nested lists)."""
    entry = dict(entry)
    each = entry.pop("each", None)
    if not each:
        return [entry]
    groups = each if isinstance(each[0], list) else [each]
    out = []
    for combo in itertools.product(*groups):
        subs = {}
        for part in combo:
            subs.update({key: str(value) for key, value in part.items()})
        out.append(_substitute(entry, subs))
    return out


def _group(group):
    """A regex group number or ``[numerator, denominator]``; ``each`` gives strings."""
    if isinstance(group, list):
        return [int(part) for part in group]
    return None if group is None else int(group)


def parse_macros(entries: list[dict]) -> list[Macro]:
    macros, seen = [], set()
    for raw in entries:
        for entry in expand(raw):
            name = entry.get("name", "")
            if not NAME_RE.match(name):
                raise ValueError(f"macro name {name!r} must be letters only (LaTeX)")
            if name in seen:
                raise ValueError(f"duplicate macro {name}")
            seen.add(name)
            kinds = [kind for kind in KINDS if kind in entry]
            if len(kinds) != 1:
                raise ValueError(f"{name}: needs exactly one of {', '.join(KINDS)}")
            if kinds[0] != "expr" and not entry.get("source"):
                raise ValueError(f"{name}: needs a source file")
            macros.append(Macro(
                name=name,
                population=entry.get("population", ""),
                kind=kinds[0],
                spec=entry[kinds[0]],
                source=entry.get("source"),
                fmt=entry.get("format") or {},
                emit=entry.get("emit", True),
                group=_group(entry.get("group")),
            ))
    return macros


def load_mapping(path: Path) -> list[Macro]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return parse_macros(data.get("macros") or [])


# Extraction


class Sources:
    """Parsed input files, read once each."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, str], object] = {}

    def _path(self, rel: str) -> Path:
        path = self.root / rel
        if not path.is_file():
            raise Missing(f"no file {rel}")
        return path

    def _get(self, rel: str, kind: str, read):
        key = (rel, kind)
        if key not in self._cache:
            try:
                self._cache[key] = read(self._path(rel))
            except Missing as error:
                self._cache[key] = error
            except (OSError, ValueError, csv.Error) as error:
                self._cache[key] = Missing(f"unreadable {rel}: {error}")
        cached = self._cache[key]
        if isinstance(cached, Missing):
            raise cached
        return cached

    def json(self, rel: str):
        return self._get(rel, "json", lambda p: json.loads(p.read_text(encoding="utf-8")))

    def csv(self, rel: str) -> list[dict]:
        def read(path: Path) -> list[dict]:
            with path.open(newline="", encoding="utf-8") as stream:
                return list(csv.DictReader(stream))
        return self._get(rel, "csv", read)

    def text(self, rel: str) -> str:
        return self._get(rel, "text", lambda p: p.read_text(encoding="utf-8"))


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _quantile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics, as ``scripts.stage1_scores``."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _match(cell: Optional[str], want) -> bool:
    cell = "" if cell is None else str(cell).strip()
    if isinstance(want, dict):
        excluded = want.get("not")
        excluded = excluded if isinstance(excluded, list) else [excluded]
        return cell != "" and not any(_match(cell, item) for item in excluded)
    if isinstance(want, list):
        return any(_match(cell, item) for item in want)
    if want == "*":
        return cell != ""
    left, right = _number(cell), _number(want)
    if left is not None and right is not None:
        return left == right
    return cell == str(want)


def _filter(rows: list[dict], where: Optional[dict]) -> list[dict]:
    if not where:
        return rows
    for column in where:
        if rows and column not in rows[0]:
            raise Missing(f"no column {column}")
    return [row for row in rows if all(_match(row.get(c), w) for c, w in where.items())]


AGGREGATES = {
    "sum": sum,
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
    "q1": lambda values: _quantile(values, 0.25),
    "q3": lambda values: _quantile(values, 0.75),
    "p95": lambda values: _quantile(values, 0.95),
}


def extract_csv(rows: list[dict], spec: dict) -> Value:
    agg = spec.get("agg", "value")
    selected = _filter(rows, spec.get("where"))
    if agg == "count":
        return len(selected)
    if agg == "share":
        base = _filter(rows, spec.get("of"))
        if not base:
            raise Missing("no rows to share over")
        return len(selected) / len(base)
    column = spec.get("column")
    if rows and column not in rows[0]:
        raise Missing(f"no column {column}")
    if agg == "value":
        if len(selected) != 1:
            raise Missing(f"{len(selected)} rows match, expected 1")
        cell = selected[0].get(column)
        if cell is None or str(cell).strip() == "":
            raise Missing(f"empty {column}")
        return _number(cell) if _number(cell) is not None else str(cell)
    cells = [row.get(column) for row in selected]
    values = [_number(cell) for cell in cells if cell is not None and str(cell).strip() != ""]
    if any(value is None for value in values):
        raise Missing(f"non-numeric {column}")
    if not values:
        raise Missing(f"no values in {column}")
    if agg == "sd":
        if len(values) < 2:
            raise Missing("sd needs two values")
        return statistics.stdev(values)
    if agg not in AGGREGATES:
        raise ValueError(f"unknown agg {agg}")
    return AGGREGATES[agg](values)


def extract_json(data, path: Union[str, list]) -> Value:
    parts = path.split(".") if isinstance(path, str) else list(path)
    node = data
    for part in parts:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                raise Missing(f"no index {part} in {path}") from None
        elif isinstance(node, dict):
            if str(part) not in node:
                raise Missing(f"no key {part} in {path}")
            node = node[str(part)]
        else:
            raise Missing(f"{path}: {part} is below a scalar")
    if node is None:
        raise Missing(f"{path} is null")
    if isinstance(node, (dict, list)):
        raise Missing(f"{path} is not a scalar")
    if isinstance(node, bool):
        return int(node)
    return node


def extract_regex(text: str, pattern: str, group) -> Value:
    found = re.search(pattern, text)
    if not found:
        raise Missing("pattern not found")
    if isinstance(group, list):
        numerator, denominator = (_number(found.group(index)) for index in group)
        if numerator is None or not denominator:
            raise Missing("ratio groups not numeric")
        return numerator / denominator
    raw = found.group(1 if group is None else group)
    number = _number(raw)
    return raw if number is None else number


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}


def evaluate(expression: str, values: dict[str, Value]) -> float:
    """``+ - * /`` over numbers and earlier macros' raw values."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise Missing(f"needs {node.id}")
            number = _number(values[node.id])
            if number is None:
                raise Missing(f"{node.id} is not a number")
            return number
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise Missing("division by zero")
            return _OPERATORS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression {expression!r}")
    return walk(ast.parse(expression, mode="eval"))


def extract(macro: Macro, sources: Sources, values: dict[str, Value]) -> Value:
    if macro.kind == "expr":
        return evaluate(macro.spec, values)
    if macro.kind == "json":
        return extract_json(sources.json(macro.source), macro.spec)
    if macro.kind == "csv":
        return extract_csv(sources.csv(macro.source), macro.spec)
    return extract_regex(sources.text(macro.source), macro.spec, macro.group)


# Formatting


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(char, char) for char in text)


def format_value(value: Value, fmt: dict) -> str:
    """``text`` (``strip``, ``chars``), else a number: ``scale``, ``percent``,
    ``decimals`` (default 0 for integers, 2 otherwise), ``sign``."""
    if fmt.get("text"):
        text = str(value)
        prefix = fmt.get("strip")
        if prefix and text.startswith(prefix):
            text = text[len(prefix):]
        if fmt.get("chars"):
            text = text[: int(fmt["chars"])]
        return latex_escape(text)
    number = _number(value)
    if number is None:
        raise Missing(f"not a number: {value!r}")
    if not math.isfinite(number):
        raise Missing(f"not finite: {value!r}")
    scaled = number * fmt.get("scale", 1)
    if fmt.get("percent"):
        scaled *= 100
    default = 0 if isinstance(scaled, int) else 2
    decimals = int(fmt.get("decimals", default))
    quantum = decimal.Decimal(1).scaleb(-decimals)
    rounded = decimal.Decimal(repr(abs(scaled))).quantize(quantum, decimal.ROUND_HALF_UP)
    text = f"{rounded:f}"
    if rounded == 0:
        sign = ""
    elif scaled < 0:
        sign = r"\ensuremath{-}"
    else:
        sign = "+" if fmt.get("sign") else ""
    return sign + text + (r"\%" if fmt.get("percent") else "")


# Output


def describe(macro: Macro) -> str:
    if macro.kind == "expr":
        return f"= {macro.spec}"
    if macro.kind == "json":
        return f"{macro.source} : {macro.spec}"
    if macro.kind == "regex":
        group = "" if macro.group is None else f" group {macro.group}"
        return f"{macro.source} : /{macro.spec}/{group}"
    spec = macro.spec
    parts = [spec.get("agg", "value")]
    if spec.get("column"):
        parts.append(spec["column"])
    if spec.get("where"):
        parts.append("where " + json.dumps(spec["where"], separators=(",", ":")))
    if spec.get("of"):
        parts.append("of " + json.dumps(spec["of"], separators=(",", ":")))
    return f"{macro.source} : {' '.join(parts)}"


def resolve(macros: list[Macro], root: Path) -> list[Result]:
    sources = Sources(root)
    values: dict[str, Value] = {}
    results = []
    for macro in macros:
        result = Result(macro)
        try:
            result.value = extract(macro, sources, values)
            result.text = format_value(result.value, macro.fmt)
            values[macro.name] = result.value
        except Missing as error:
            result.reason = str(error)
        results.append(result)
    return results


def render(results: list[Result], root: Path, mapping: Path) -> str:
    shown = [result for result in results if result.macro.emit]
    done = sum(result.text is not None for result in shown)
    lines = [
        "% Paper numbers, generated by `python -m scripts.paper_numbers` (#130).",
        "% Do not edit by hand: re-run the script after the experiments update.",
        f"% Root: {root}",
        f"% Mapping: {mapping.name}",
        f"% {done} of {len(shown)} macros resolved; missing inputs render as {MISSING}.",
        "% Use as \\StageOneMean{} (the braces keep the following space).",
        "",
    ]
    for result in shown:
        macro = result.macro
        lines.append(f"% {macro.population} | {describe(macro)}")
        if result.text is None:
            lines.append(f"% ?? {result.reason}")
        lines.append(f"\\newcommand{{\\{macro.name}}}{{{result.text or MISSING}}}")
    return "\n".join(lines) + "\n"


def listing(macros: list[Macro]) -> str:
    lines = ["| macro | population | source |", "|---|---|---|"]
    for macro in macros:
        if macro.emit:
            source = describe(macro).replace("|", "\\|")
            lines.append(f"| `\\{macro.name}` | {macro.population} | `{source}` |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list", action="store_true",
                        help="print the macro table (Markdown) instead of writing numbers.tex")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any macro is missing")
    args = parser.parse_args(argv)
    macros = load_mapping(args.mapping)
    if args.list:
        print(listing(macros), end="")
        return 0
    results = resolve(macros, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results, args.root, args.mapping), encoding="utf-8")
    shown = [result for result in results if result.macro.emit]
    missing = [result for result in shown if result.text is None]
    for result in missing:
        print(f"?? \\{result.macro.name}: {result.reason}", file=sys.stderr)
    print(f"{args.out}: {len(shown) - len(missing)} of {len(shown)} macros resolved, "
          f"{len(missing)} missing", file=sys.stderr)
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    sys.exit(main())
