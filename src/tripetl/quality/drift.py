"""Schema drift: what a source actually holds, against what we declared.

:mod:`tripetl.schema` declares every layout this pipeline reads, and
:func:`tripetl.sources.read_raw` hands the raw one to Spark rather than letting
it be inferred. That buys less than it looks like it does. A schema given to
the Parquet reader is a *projection*, not an assertion: a column the file does
not have comes back present and entirely null, a column the file gained is
dropped without a word, and only a type Spark cannot reconcile at all raises --
from inside the scan, naming a row group rather than the column.

The all-null case is the expensive one. Every ``not_null`` rule on that column
then reports 0% and the gate blocks, so the run does stop -- but the report
says *the data is bad* when what actually happened is *the source was renamed*,
and that is an hour spent looking in the wrong place. Comparing the stored
schema against the declared one first turns it into one line naming the column
that vanished.

Drift is graded rather than pass/fail, on one question: would it change how a
row reads? A missing column and an unreconcilable type would, and block. An
added column, a safe upcast and a change of spelling would not, and are
reported so that a feed drifting under you is visible before the day it breaks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)

#: Declared in the schema, absent from the source. The declared read yields a
#: column of nulls, so the damage surfaces as a data-quality failure somewhere
#: unrelated to the actual cause.
MISSING = "missing"

#: Present in the source, not declared. Dropped by the declared read, which is
#: harmless today and is also how you find out the vendor added a column.
UNEXPECTED = "unexpected"

#: Stored as a type that reads losslessly as the declared one.
WIDENED = "widened"

#: Stored as a type that does not. Either the read fails mid-scan, or it
#: succeeds and the values are not what the transforms expect.
MISMATCHED = "mismatched"

#: Same column, different spelling. Spark resolves names case-insensitively by
#: default, so the declared read still finds it and still hands downstream code
#: the declared spelling. Worth reporting, not worth stopping over.
RECASED = "recased"

#: Kinds that change how a row reads, and so must stop a run.
BLOCKING_KINDS = frozenset({MISSING, MISMATCHED})

#: Stored types that can be read as the declared type without losing a value.
#: Spark applies these upcasts silently and they are genuinely safe, so a
#: source storing ``int`` where we declared ``double`` is worth reporting but
#: not worth blocking. ``bigint`` -> ``double`` is the one arguable entry: a
#: widening by SQL's type precedence, but lossy above 2^53. The columns it
#: could apply to here are ids, counts and money in dollars, none of which come
#: within eleven orders of magnitude of that.
WIDENINGS: dict[str, frozenset[str]] = {
    "smallint": frozenset({"tinyint"}),
    "int": frozenset({"tinyint", "smallint"}),
    "bigint": frozenset({"tinyint", "smallint", "int"}),
    "float": frozenset({"tinyint", "smallint", "int"}),
    "double": frozenset({"tinyint", "smallint", "int", "bigint", "float"}),
    "timestamp": frozenset({"date"}),
}


@dataclass(frozen=True)
class ColumnDrift:
    """One column that does not match its declaration.

    Attributes:
        expected: How the column was declared -- its type, except for
            :data:`RECASED`, where it is the declared spelling.
        actual: How the source stores it, under the same convention. ``None``
            for :data:`MISSING`, where there is nothing stored to describe.
    """

    column: str
    kind: str
    expected: str | None = None
    actual: str | None = None

    @property
    def is_blocking(self) -> bool:
        """True when this drift changes how a row reads."""
        return self.kind in BLOCKING_KINDS

    def summary(self) -> str:
        if self.kind == MISSING:
            detail = f"declared {self.expected}, absent from the source"
        elif self.kind == UNEXPECTED:
            detail = f"stored as {self.actual}, never declared"
        elif self.kind == RECASED:
            detail = f"declared {self.expected!r}, stored as {self.actual!r}"
        else:
            detail = f"declared {self.expected}, stored as {self.actual}"
        marker = "BLOCK" if self.is_blocking else "WARN "
        return f"[{marker}] {self.kind}[{self.column}]: {detail}"


@dataclass(frozen=True)
class SchemaDiff:
    """Every difference between a source's layout and the declared one.

    Made of primitives so it survives a round trip through JSON. Like a quality
    report it is written to disk before it is enforced: a run stopped by drift
    must leave behind the evidence of what drifted, or the first thing anyone
    does is re-run the job against a source that may have moved on again.
    """

    dataset: str
    declared_columns: int
    stored_columns: int
    columns: tuple[ColumnDrift, ...] = ()
    types_checked: bool = True

    @property
    def conforms(self) -> bool:
        """True when nothing found would change how a row reads."""
        return not self.blocking

    @property
    def blocking(self) -> tuple[ColumnDrift, ...]:
        return tuple(drift for drift in self.columns if drift.is_blocking)

    @property
    def advisory(self) -> tuple[ColumnDrift, ...]:
        return tuple(drift for drift in self.columns if not drift.is_blocking)

    def of_kind(self, kind: str) -> tuple[ColumnDrift, ...]:
        return tuple(drift for drift in self.columns if drift.kind == kind)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "conforms": self.conforms,
            "declared_columns": self.declared_columns,
            "stored_columns": self.stored_columns,
            "types_checked": self.types_checked,
            "blocking_count": len(self.blocking),
            "advisory_count": len(self.advisory),
            "columns": [asdict(drift) for drift in self.columns],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        verdict = "CONFORMS" if self.conforms else "DRIFTED"
        scope = "names and types" if self.types_checked else "names only"
        lines = [
            f"schema check :: {self.dataset or 'source'} :: {verdict}",
            f"  declared: {self.declared_columns}   stored: {self.stored_columns}"
            f"   checked: {scope}"
            f"   blocking: {len(self.blocking)}   advisory: {len(self.advisory)}",
        ]
        if not self.columns:
            lines.append("  no drift")
        lines.extend(f"  {drift.summary()}" for drift in self.columns)
        return "\n".join(lines)


class SchemaDriftError(RuntimeError):
    """Raised when a source's layout would change how its rows read.

    Carries the whole diff rather than only a message, so a caller -- an
    Airflow task, a test, the CLI -- can render the detail instead of reading
    the source again to find out what moved.
    """

    def __init__(self, diff: SchemaDiff) -> None:
        self.diff = diff
        named = ", ".join(f"{drift.kind}[{drift.column}]" for drift in diff.blocking)
        super().__init__(
            f"schema drift in {diff.dataset or 'the source'}: {named} "
            f"({len(diff.blocking)} blocking, {len(diff.advisory)} advisory)"
        )


def compare_schemas(
    stored: StructType,
    declared: StructType,
    *,
    dataset: str = "",
    check_types: bool = True,
    case_sensitive: bool = False,
) -> SchemaDiff:
    """Diff the layout a source carries against the one we declared.

    Args:
        stored: What the source actually holds.
        declared: What this pipeline expects to read.
        dataset: Named in the diff and in any error, so a failure says which
            source moved.
        check_types: Compare types as well as names. False for a source that
            has no types to compare -- a headered CSV read without inference is
            every column a string, and checking types against that reports
            drift on every column of a perfectly good file.
        case_sensitive: Match names exactly. The default mirrors Spark, which
            resolves column names case-insensitively unless
            ``spark.sql.caseSensitive`` is set; matching more strictly than the
            reader does would report drift the reader handles.
    """

    def key(name: str) -> str:
        return name if case_sensitive else name.casefold()

    by_key = {key(field.name): field for field in stored.fields}
    declared_keys = {key(field.name) for field in declared.fields}

    drifts: list[ColumnDrift] = []
    for field in declared.fields:
        found = by_key.get(key(field.name))
        want = field.dataType.simpleString()
        if found is None:
            drifts.append(ColumnDrift(column=field.name, kind=MISSING, expected=want))
            continue

        if found.name != field.name:
            drifts.append(
                ColumnDrift(
                    column=field.name,
                    kind=RECASED,
                    expected=field.name,
                    actual=found.name,
                )
            )

        have = found.dataType.simpleString()
        if not check_types or have == want:
            continue
        kind = WIDENED if have in WIDENINGS.get(want, frozenset()) else MISMATCHED
        drifts.append(ColumnDrift(column=field.name, kind=kind, expected=want, actual=have))

    drifts.extend(
        ColumnDrift(
            column=field.name,
            kind=UNEXPECTED,
            actual=field.dataType.simpleString(),
        )
        for field in stored.fields
        if key(field.name) not in declared_keys
    )

    return SchemaDiff(
        dataset=dataset,
        declared_columns=len(declared.fields),
        stored_columns=len(stored.fields),
        columns=tuple(drifts),
        types_checked=check_types,
    )


def enforce_schema(diff: SchemaDiff, *, strict: bool = True) -> SchemaDiff:
    """Log the diff and, when ``strict``, raise on drift that would change a read.

    Mirrors :func:`tripetl.quality.gate.enforce` deliberately: the two are the
    same decision made about different evidence, and a caller that turns one
    off for a first look at an unfamiliar extract almost always wants the other
    off too.
    """
    for advisory in diff.advisory:
        logger.warning("schema drift in %s -- %s", diff.dataset, advisory.summary())

    if diff.conforms:
        logger.info(
            "schema check passed for %s (%d columns declared, %d advisory notes)",
            diff.dataset,
            diff.declared_columns,
            len(diff.advisory),
        )
        return diff

    for blocking in diff.blocking:
        logger.error("schema drift in %s -- %s", diff.dataset, blocking.summary())

    if strict:
        raise SchemaDriftError(diff)

    logger.warning(
        "schema drift in %s not enforced (strict=False); reading anyway with %d blocking",
        diff.dataset,
        len(diff.blocking),
    )
    return diff
