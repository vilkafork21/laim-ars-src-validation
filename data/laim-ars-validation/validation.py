from typing import Any, ClassVar, Iterator, Callable
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from itertools import chain, starmap
from functools import reduce, partial
from operator import add, or_, attrgetter

from math import log10, isfinite
from pathlib import PurePath

import polars as pl

from device import Device
from common_core import Categories, Sentinel
from core import SpansDataSpec, SpanAttr, Predicate
from spec import SpansData, Recast, DType

import perf
import rules
from report import Ui, Chip, Bar, Fmt, Fig, Grid
from tui import ColorSchemeDataScience
from tui_data import ColorSchemeDataScienceCold


type Aggregate = Callable[[pl.Expr], pl.Expr]
type MetricFn = Callable[[SpansDataSpec, Resolution], Iterator[pl.Expr]]


@dataclass(frozen=True)
class Ns:
    per_ms: ClassVar[int] = 1_000_000
    per_second: ClassVar[int] = 1_000_000_000
    per_day: ClassVar[int] = 86_400 * per_second


@dataclass(frozen=True)
class Num:
    types: ClassVar[tuple[type[pl.DataType], ...]] = (pl.Int64, pl.Float32, pl.Float64)


@dataclass(frozen=True, eq=True)
class Resolution:
    percentiles: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95)
    grains: tuple[str, ...] = ("1d", "4h")
    windows: tuple[int, ...] = (10, 50, 100)


# K7 сравнивает агентов на одинаковой суточной шкале. Отдельное адаптивное
# зерно TimeGrain используется только для читаемого отображения графиков.
READINESS_GRAIN = "1d"

# Итог проверки (status): "full" — K1–K7, как в методике; "quality" — только
# качество данных K1–K6, готовность K7 рассчитывается и показывается, но в
# светофор не входит. Нужен нодам, для которых недостаточный объём данных
# не должен закрашивать красным пригодные по качеству данные.
VERDICT_SCOPES = ("full", "quality")


@dataclass(frozen=True, eq=False)
class Dim:
    name: str
    expr: pl.Expr

    def keyed(self) -> pl.Expr:
        return self.expr.alias(self.name)


@dataclass(frozen=True, eq=False)
class By:
    trace: ClassVar[Dim] = Dim("trace_id", pl.col("trace_id"))
    agent: ClassVar[Dim] = Dim("agent_id", pl.col("agent_id"))
    session: ClassVar[Dim] = Dim("session_id", pl.col("session_id"))
    ver_major: ClassVar[Dim] = Dim(
        "ver_major", pl.col("nexus_distrib_ver").str.extract(r"^(\d+)", 1)
    )
    ver_minor: ClassVar[Dim] = Dim(
        "ver_minor", pl.col("nexus_distrib_ver").str.extract(r"^(\d+\.\d+)", 1)
    )
    ver_patch: ClassVar[Dim] = Dim("ver_patch", pl.col("nexus_distrib_ver"))

    @staticmethod
    def time(grain: str) -> Dim:
        return Dim(
            "ts",
            pl.from_epoch(pl.col("start_time_ns"), time_unit="ns").dt.truncate(grain),
        )


@dataclass(frozen=True, eq=False)
class Group:
    @staticmethod
    def keyed(dims: tuple[Dim, ...]) -> tuple[pl.Expr, ...]:
        return tuple(map(Dim.keyed, dims))

    @staticmethod
    def named(dims: tuple[Dim, ...]) -> tuple[str, ...]:
        return tuple(map(attrgetter("name"), dims))

    @staticmethod
    def aggregate(
        lf: pl.LazyFrame, dims: tuple[Dim, ...], body: Iterator[pl.Expr]
    ) -> pl.LazyFrame:
        keys = Group.keyed(dims)

        return lf.select(body) if not keys else lf.group_by(keys).agg(body)


@dataclass(frozen=True, eq=False)
class FromAttr:
    @staticmethod
    def real_values(a: SpanAttr) -> pl.Expr:
        col = pl.col(a.name)

        return col if a.empty is None else col.filter(col != a.empty)

    @staticmethod
    def missing(a: SpanAttr) -> pl.Expr:
        col = pl.col(a.name)

        return col.is_null() | (
            (col == a.empty) if a.empty is not None else pl.lit(False)
        )

    @staticmethod
    def trash(a: SpanAttr) -> pl.Expr:
        return (pl.col(a.name) == a.trash) if a.trash is not None else pl.lit(False)

    @staticmethod
    def violation(a: SpanAttr) -> pl.Expr:
        return (~a.validity).fill_null(True)


@dataclass(frozen=True, eq=False)
class Across:
    @staticmethod
    def missing(
        attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64
    ) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.missing(a).cast(dtype), attrs))

    @staticmethod
    def violations(
        attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64
    ) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.violation(a).cast(dtype), attrs))

    @staticmethod
    def nulls(
        attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64
    ) -> pl.Expr:
        return reduce(add, map(lambda a: pl.col(a.name).is_null().cast(dtype), attrs))

    @staticmethod
    def trash(
        attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64
    ) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.trash(a).cast(dtype), attrs))

    @staticmethod
    def violation_rate(attrs: tuple[SpanAttr, ...]) -> pl.Expr:
        return Across.violations(attrs, pl.Float64) / len(attrs)

    @staticmethod
    def null_rate(attrs: tuple[SpanAttr, ...]) -> pl.Expr:
        return Across.nulls(attrs, pl.Float64) / len(attrs)


@dataclass(frozen=True, eq=False)
class FromSpec:
    @staticmethod
    def numeric_attrs(spec: SpansDataSpec) -> Iterator[SpanAttr]:
        return filter(lambda a: a.type_polars in Num.types, spec.attrs)

    @staticmethod
    def mandatory(spec: SpansDataSpec) -> tuple[SpanAttr, ...]:
        return tuple(filter(lambda a: a.mandatory, spec.attrs))

    @staticmethod
    def distribution_signals(spec: SpansDataSpec) -> Iterator[tuple[str, pl.Expr]]:
        return chain(
            map(
                lambda a: (a.name, FromAttr.real_values(a)),
                FromSpec.numeric_attrs(spec),
            ),
            (("duration_ns", pl.col("end_time_ns") - pl.col("start_time_ns")),),
        )

    @staticmethod
    def dynamic_signals(spec: SpansDataSpec) -> tuple[tuple[str, pl.Expr], ...]:
        return (
            ("missing", Across.missing(spec.attrs)),
            ("invalid", Across.violations(spec.attrs)),
            ("duration", pl.col("end_time_ns") - pl.col("start_time_ns")),
            ("error", (pl.col("status_code") == "STATUS_CODE_ERROR").cast(pl.Int64)),
        )


@dataclass(frozen=True, eq=False)
class Metric:
    @staticmethod
    def completeness(
        spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean
    ) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.missing(a)).alias(a.name), spec.attrs)

    @staticmethod
    def correctness(
        spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean
    ) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.violation(a)).alias(a.name), spec.attrs)

    @staticmethod
    def contamination(
        spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean
    ) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.trash(a)).alias(a.name), spec.attrs)

    @staticmethod
    def nulls(
        spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.sum
    ) -> Iterator[pl.Expr]:
        return map(lambda a: agg(pl.col(a.name).is_null()).alias(a.name), spec.attrs)

    @staticmethod
    def distribution(spec: SpansDataSpec, resolution: Resolution) -> Iterator[pl.Expr]:
        quantiles = lambda name, value: map(
            lambda q: value.quantile(q, interpolation="linear").alias(
                f"{name}__q{int(q * 100):02d}"
            ),
            resolution.percentiles,
        )

        return chain.from_iterable(
            starmap(quantiles, FromSpec.distribution_signals(spec))
        )


@dataclass(frozen=True, eq=False)
class Quality:
    scanned: pl.LazyFrame
    spec: SpansDataSpec = SpansData
    resolution: Resolution = field(default_factory=Resolution)

    invariants: ClassVar[tuple[str, ...]] = (
        "agent_id",
        "session_id",
        "service_name",
        "service_version",
        "nexus_distrib_ver",
    )

    def blocking_attrs(self) -> tuple[SpanAttr, ...]:
        return FromSpec.mandatory(self.spec)

    def advisory_attrs(self) -> tuple[SpanAttr, ...]:
        return tuple(filter(lambda a: not a.mandatory, self.spec.attrs))

    def profile(
        self, metric: MetricFn, *dims: Dim, grain: None | str = None
    ) -> pl.LazyFrame:
        graining = (By.time(grain),) if grain else ()
        keys = frozenset(Group.named(dims + graining))
        body = chain(
            (pl.len().alias("n"),),
            filter(
                lambda e: e.meta.output_name() not in keys,
                metric(self.spec, self.resolution),
            ),
        )

        return Group.aggregate(self.scanned, dims + graining, body)

    def completeness(
        self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean
    ) -> pl.LazyFrame:
        return self.profile(partial(Metric.completeness, agg=agg), *dims, grain=grain)

    def correctness(
        self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean
    ) -> pl.LazyFrame:
        return self.profile(partial(Metric.correctness, agg=agg), *dims, grain=grain)

    def contamination(
        self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean
    ) -> pl.LazyFrame:
        return self.profile(partial(Metric.contamination, agg=agg), *dims, grain=grain)

    def nulls(
        self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.sum
    ) -> pl.LazyFrame:
        return self.profile(partial(Metric.nulls, agg=agg), *dims, grain=grain)

    def distribution(self, *dims: Dim, grain: None | str = None) -> pl.LazyFrame:
        return self.profile(Metric.distribution, *dims, grain=grain)

    def tagged(self, over: Dim = By.trace) -> pl.LazyFrame:
        rejected = reduce(or_, map(FromAttr.violation, FromSpec.mandatory(self.spec)))
        flags = self.scanned.group_by(over.name).agg(rejected.any().alias("rejected"))

        # Polars объединяет NULL trace_id в одну группу. Для join нужна та же
        # семантика ключа, иначе строки теряют флаг отбраковки и нарушается
        # инвариант K5/K7: n = dropped + kept.
        return self.scanned.join(flags, on=over.name, how="left", nulls_equal=True)

    def rejection(
        self,
        *dims: Dim,
        grain: None | str = None,
        over: Dim = By.trace,
        agg: Aggregate = pl.Expr.mean,
    ) -> pl.LazyFrame:
        graining = (By.time(grain),) if grain else ()
        trace_level = (
            self.tagged(over)
            .group_by(Group.keyed((over,) + dims + graining))
            .agg(pl.col("rejected").max().alias("rejected"))
        )
        names = Group.named(dims + graining)
        rejected = pl.col("rejected")
        body = (
            pl.len().alias("n"),
            agg(rejected).alias("dropped_rate"),
            rejected.sum().cast(pl.UInt64).alias("dropped"),
            (~rejected).sum().cast(pl.UInt64).alias("kept"),
        )

        return (
            trace_level.select(body)
            if not names
            else trace_level.group_by(names).agg(body)
        )

    def trace_status(self) -> pl.LazyFrame:
        canon = map(lambda c: pl.first(c).alias(c), ("session_id", "nexus_distrib_ver"))
        agreed = (
            reduce(
                add,
                map(
                    lambda c: (pl.col(c).n_unique() > 1).cast(pl.Int64), self.invariants
                ),
            )
            == 0
        )
        body = chain(
            canon,
            (
                pl.len().alias("spans"),
                pl.col("rejected").max().alias("rejected"),
                agreed.alias("consistent"),
            ),
        )

        return self.tagged().group_by("trace_id").agg(body)

    def consistency(self, *cols: str) -> pl.LazyFrame:
        targets = cols or self.invariants
        counts = map(lambda c: pl.col(c).n_unique().alias(c), targets)

        return self.scanned.group_by("trace_id").agg(
            chain((pl.len().alias("spans"),), counts)
        )

    def trace_consistency(self, *cols: str) -> pl.LazyFrame:
        targets = cols or self.invariants
        broken = map(lambda c: (pl.col(c).n_unique() > 1).alias(c), targets)
        rolled = self.scanned.group_by("trace_id").agg(broken)

        return rolled.select(
            map(lambda c: pl.col(c).sum().cast(pl.UInt64).alias(c), targets)
        )

    def dynamics(self, window: None | int = None) -> pl.LazyFrame:
        w = window or self.resolution.windows[0]
        order = "start_time_ns"
        seq = pl.col("span_id").cum_count().over("trace_id", order_by=order)

        def running(name: str, signal: pl.Expr) -> tuple[pl.Expr, ...]:
            cumulative = signal.cum_sum().over("trace_id", order_by=order)

            return (
                signal.alias(name),
                cumulative.alias(f"{name}__cum"),
                (cumulative / seq).alias(f"{name}__mean"),
                signal.rolling_mean(window_size=w, min_samples=1)
                .over("trace_id", order_by=order)
                .alias(f"{name}__roll{w}"),
            )

        columns = chain.from_iterable(
            starmap(running, FromSpec.dynamic_signals(self.spec))
        )

        return self.scanned.with_columns(chain((seq.alias("span_seq"),), columns))

    def verdict(
        self,
        *,
        max_nulls: int = 0,
        max_rules: int = 0,
        max_dups: int = 0,
        max_breaks: int = 0,
    ) -> pl.LazyFrame:
        attrs = self.spec.attrs
        blocking_attrs = self.blocking_attrs()
        advisory_attrs = self.advisory_attrs()
        nulls = Across.nulls(attrs).sum().cast(pl.UInt64)
        miss = Across.missing(attrs).sum().cast(pl.UInt64)
        trash = Across.trash(attrs).sum().cast(pl.UInt64)
        rules = Across.violations(attrs).sum().cast(pl.UInt64)
        # K2 ограничивает число спанов, нарушивших правило одного атрибута.
        # Сумма по атрибутам считала один спан несколько раз и могла давать
        # значение больше общего числа спанов.
        blocking_rules = pl.max_horizontal(
            tuple(FromAttr.violation(a).sum() for a in blocking_attrs)
        ).cast(pl.UInt64)
        advisory_rules = Across.violations(advisory_attrs).sum().cast(pl.UInt64)
        dups = (pl.len() - pl.struct("trace_id", "span_id").n_unique()).cast(pl.UInt64)
        broken = pl.any_horizontal(
            tuple(pl.col(c).n_unique().over("trace_id") > 1 for c in self.invariants)
        )
        breaks = pl.col("trace_id").filter(broken).n_unique().cast(pl.UInt64)

        return self.scanned.select(
            nulls.alias("null_violations"),
            miss.alias("missing"),
            trash.alias("trash_cells"),
            rules.alias("rule_violations"),
            blocking_rules.alias("blocking_rule_violations"),
            advisory_rules.alias("advisory_rule_violations"),
            dups.alias("duplicate_keys"),
            breaks.alias("trace_breaks"),
            (
                (nulls <= max_nulls)
                & (blocking_rules <= max_rules)
                & (dups <= max_dups)
                & (breaks <= max_breaks)
            ).alias("valid"),
        )

    def is_valid(self, **thresholds: int) -> bool:
        return bool(self.verdict(**thresholds).collect().get_column("valid").item())

    def flow(self, grain: str = "1d") -> pl.LazyFrame:
        blocking = Across.violation_rate(self.blocking_attrs())
        overall = Across.violation_rate(self.spec.attrs)

        return (
            self.scanned.group_by(By.time(grain).keyed())
            .agg(
                pl.col("trace_id").n_unique().alias("traces"),
                (1 - blocking.mean()).alias("validity"),
                overall.mean().alias("defect"),
            )
            .sort("ts")
        )


def scan(path: PurePath, spec: SpansDataSpec = SpansData) -> pl.LazyFrame:
    lazy = pl.scan_parquet(path.as_posix(), rechunk=True)

    return Recast.frame(lazy, lazy.collect_schema(), spec)


@dataclass(frozen=True, eq=False)
class SchemaState:
    present: ClassVar[str] = "ок"
    converted: ClassVar[str] = "приведён"
    absent: ClassVar[str] = "отсутствует"
    incompatible: ClassVar[str] = "несовместим"
    broken: ClassVar[tuple[str, ...]] = (absent, incompatible)


@dataclass(frozen=True, eq=False)
class SchemaCheck:
    @staticmethod
    def name(dtype: None | DType) -> str:
        return "—" if dtype is None else str(dtype)

    @staticmethod
    def state(source: None | DType, target: DType) -> str:
        return (
            SchemaState.absent
            if source is None
            else SchemaState.incompatible
            if not Recast.castable(source, target)
            else SchemaState.present
            if source == target
            else SchemaState.converted
        )

    @staticmethod
    def row(a: SpanAttr, mandatory: frozenset[str], schema: pl.Schema) -> tuple:
        source = schema.get(a.name)

        return (
            a.name,
            a.name in mandatory,
            SchemaCheck.name(source),
            SchemaCheck.name(a.type_polars),
            SchemaCheck.state(source, a.type_polars),
        )

    @staticmethod
    def diff(schema: pl.Schema, spec: SpansDataSpec, preset: "Preset") -> dict:
        names = frozenset(spec.column_names)
        mandatory = spec.mandatory_names
        rows = tuple(map(lambda a: SchemaCheck.row(a, mandatory, schema), spec.attrs))
        broken = lambda r: r[4] in SchemaState.broken

        crit = tuple(filter(lambda r: r[1], rows))
        soft = tuple(filter(lambda r: not r[1], rows))
        crit_bad = tuple(filter(broken, crit))
        soft_bad = tuple(filter(broken, soft))
        extra = tuple(filter(lambda c: c not in names, schema.names()))
        critical_ok = len(crit_bad) == 0
        ceiling = max(0.0, min(1.0, 1.0 - preset.ceiling_penalty * len(soft_bad)))
        # Безопасное приведение типа — штатный путь для dataframe-входа
        # платформы, где колонки часто приходят как String. Оно фиксируется в
        # отчёте, но само по себе не понижает итоговый статус.
        # Колонки вне спецификации сохраняются в отчёте, но не являются
        # дефектом K1: они не участвуют ни в приведении, ни в проверках.
        degraded = (not critical_ok) or bool(soft_bad)

        return {
            "rows": rows,
            "crit": crit,
            "soft_bad": soft_bad,
            "extra": extra,
            "critical_ok": critical_ok,
            "ceiling": ceiling,
            "degraded": degraded,
            "counts": {
                "колонок по спецификации": len(spec.column_names),
                "из них присутствует": len(frozenset(schema.names()) & names),
                "критичных (красных)": len(mandatory),
                "критичных нарушено": len(crit_bad),
                "некритичных с дефектом": len(soft_bad),
                "лишних колонок": len(extra),
                "потолок качества, %": round(ceiling * 100),
            },
        }


@dataclass(frozen=True, eq=False)
class Preset:
    volume_target: float
    min_eff_traces: float
    max_loss: float
    min_quality: float
    ready_at: float
    pilot_at: float
    trend_gain: float
    rate_target: float
    stability_tol: float
    ceiling_penalty: float


@dataclass(frozen=True, eq=False)
class Significance:
    A: ClassVar[Preset] = Preset(1e9, 1e6, 0.02, 0.98, 0.92, 0.80, 4.0, 1e5, 0.05, 0.30)
    B: ClassVar[Preset] = Preset(1e7, 1e5, 0.07, 0.93, 0.85, 0.70, 3.5, 1e4, 0.07, 0.20)
    C: ClassVar[Preset] = Preset(1e5, 1e4, 0.30, 0.80, 0.70, 0.50, 3.0, 1e3, 0.10, 0.12)
    D: ClassVar[Preset] = Preset(1e4, 1e3, 0.45, 0.65, 0.55, 0.35, 2.5, 1e2, 0.15, 0.07)
    E: ClassVar[Preset] = Preset(1e3, 1e2, 0.60, 0.50, 0.40, 0.25, 2.0, 1e1, 0.25, 0.04)

    levels: ClassVar[tuple[str, ...]] = ("A", "B", "C", "D", "E")

    @staticmethod
    def of(level: str) -> Preset:
        return getattr(Significance, level if level in Significance.levels else "C")


@dataclass(frozen=True, eq=True)
class Policy:
    window_days: int = 30

    volume_target: float = 1e5
    volume_steepness: float = 1.5

    w_validity: float = 0.40
    w_presence: float = 0.20
    w_retention: float = 0.40

    trend_gain: float = 3.0
    rate_target: float = 1e3
    stability_tol: float = 0.10

    w_trend: float = 0.40
    w_stability: float = 0.40
    w_rate: float = 0.20

    w_volume: float = 0.40
    w_quality: float = 0.40
    w_accel: float = 0.20

    min_eff_traces: float = 1e4
    max_loss: float = 0.30
    min_quality: float = 0.80

    ready_at: float = 0.70
    pilot_at: float = 0.50

    @staticmethod
    def of(significance: str, **overrides: float) -> "Policy":
        p = Significance.of(significance)
        days = int(overrides.get("window_days", 30))
        rest = dict(filter(lambda kv: kv[0] != "window_days", overrides.items()))
        base = dict(
            volume_target=p.volume_target,
            min_eff_traces=p.min_eff_traces,
            max_loss=p.max_loss,
            min_quality=p.min_quality,
            ready_at=p.ready_at,
            pilot_at=p.pilot_at,
            trend_gain=p.trend_gain,
            rate_target=p.rate_target,
            stability_tol=p.stability_tol,
        )

        return Policy(window_days=days, **{**base, **rest})

    @staticmethod
    def logistic(x: pl.Expr) -> pl.Expr:
        return 1 / (1 + (-x).exp())

    def volume(self, eff_traces: pl.Expr) -> pl.Expr:
        return self.logistic(
            self.volume_steepness
            * ((eff_traces + 1).log10() - log10(self.volume_target))
        )

    def quality_sum(
        self, validity: pl.Expr, presence: pl.Expr, retention: pl.Expr
    ) -> pl.Expr:
        return (
            self.w_validity * validity
            + self.w_presence * presence
            + self.w_retention * retention
        )

    def quality_mul(
        self, validity: pl.Expr, presence: pl.Expr, retention: pl.Expr
    ) -> pl.Expr:
        return (
            validity**self.w_validity
            * presence**self.w_presence
            * retention**self.w_retention
        )

    def acceleration(
        self, trend: pl.Expr, stability: pl.Expr, rate: pl.Expr
    ) -> pl.Expr:
        return (
            self.w_trend * self.logistic(self.trend_gain * trend)
            + self.w_stability * stability
            + self.w_rate * (1 - (-rate / self.rate_target).exp())
        )

    def readiness(self, volume: pl.Expr, quality: pl.Expr, accel: pl.Expr) -> pl.Expr:
        return self.w_volume * volume + self.w_quality * quality + self.w_accel * accel

    def verdict(
        self, eff_traces: pl.Expr, loss: pl.Expr, quality: pl.Expr, readiness: pl.Expr
    ) -> pl.Expr:
        gated = (
            (eff_traces >= self.min_eff_traces)
            & (loss <= self.max_loss)
            & (quality >= self.min_quality)
        )

        return (
            pl.when(~gated)
            .then(pl.lit("not_ready"))
            .when(readiness >= self.ready_at)
            .then(pl.lit("ready"))
            .when(readiness >= self.pilot_at)
            .then(pl.lit("pilot"))
            .otherwise(pl.lit("not_ready"))
        )

    def reason(
        self,
        eff_traces: pl.Expr,
        loss: pl.Expr,
        quality: pl.Expr,
        readiness: pl.Expr,
        worst: pl.Expr,
        trend: pl.Expr,
    ) -> pl.Expr:
        return (
            pl.when(eff_traces < self.min_eff_traces)
            .then(pl.lit("недостаточно трейсов"))
            .when(loss > self.max_loss)
            .then(
                pl.concat_str(
                    (
                        pl.lit(
                            "отбраковка выше допуска; максимальная доля нарушений: "
                        ),
                        worst,
                    )
                )
            )
            .when(quality < self.min_quality)
            .then(
                pl.concat_str(
                    (pl.lit("Q ниже порога; максимальная доля нарушений: "), worst)
                )
            )
            .when(trend < 0)
            .then(pl.lit("приток данных снижается"))
            .when(readiness >= self.ready_at)
            .then(pl.lit("готов к мониторингу"))
            .when(readiness >= self.pilot_at)
            .then(pl.lit("кандидат в пилот"))
            .otherwise(pl.lit("накапливаем данные"))
        )


@dataclass(frozen=True, eq=False)
class Readiness:
    quality: Quality
    policy: Policy = field(default_factory=Policy)

    def windowed(self) -> pl.LazyFrame:
        span_ns = self.policy.window_days * Ns.per_day

        return self.quality.tagged().filter(
            pl.col("start_time_ns") >= pl.col("start_time_ns").max() - span_ns
        )

    def components(self, win: pl.LazyFrame, *dims: Dim) -> pl.LazyFrame:
        attrs = self.quality.blocking_attrs()
        mand = FromSpec.mandatory(self.quality.spec)
        viol = Across.violation_rate(attrs)
        gaps = Across.null_rate(mand)
        base = (
            pl.col("trace_id").n_unique().alias("traces"),
            pl.col("trace_id")
            .filter(~pl.col("rejected"))
            .n_unique()
            .alias("eff_traces"),
            pl.len().alias("spans"),
            (1 - viol.mean()).alias("validity"),
            (1 - gaps.mean()).alias("presence"),
            (1 - pl.col("rejected").mean()).alias("retention"),
        )
        blame = map(
            lambda m: FromAttr.violation(m).mean().alias(f"_viol__{m.name}"), mand
        )
        keys = Group.keyed(dims)
        body = chain(base, blame)
        frame = win.select(body) if not keys else win.group_by(keys).agg(body)

        return self.scored(frame)

    def scored(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        policy = self.policy
        mand = FromSpec.mandatory(self.quality.spec)
        worst_rate = pl.max_horizontal(map(lambda m: pl.col(f"_viol__{m.name}"), mand))
        worst_name = reduce(
            lambda acc, m: (
                pl.when(pl.col(f"_viol__{m.name}") >= worst_rate)
                .then(pl.lit(m.name))
                .otherwise(acc)
            ),
            mand,
            pl.lit("—"),
        )

        return (
            frame.with_columns(
                (1 - pl.col("eff_traces") / pl.col("traces")).alias("loss_rate"),
                worst_rate.alias("worst_rule_rate"),
                pl.when(worst_rate <= 0)
                .then(pl.lit("—"))
                .otherwise(worst_name)
                .alias("worst_rule_attr"),
            )
            .with_columns(
                policy.volume(pl.col("eff_traces")).alias("volume_score"),
                policy.quality_sum(
                    pl.col("validity"), pl.col("presence"), pl.col("retention")
                ).alias("quality_score"),
            )
            .select(pl.exclude(r"^_viol__.*$"))
        )

    def acceleration(
        self, win: pl.LazyFrame, *dims: Dim, grain: str = "1d"
    ) -> pl.LazyFrame:
        policy = self.policy
        attrs = self.quality.blocking_attrs()
        viol = Across.violation_rate(attrs)
        bkeys = Group.keyed(dims) + (By.time(grain).keyed(),)
        bucket = win.group_by(bkeys).agg(
            pl.col("trace_id").n_unique().alias("b_traces"),
            (1 - viol.mean()).alias("b_validity"),
        )
        names = Group.named(dims)
        order = pl.col("ts").dt.epoch("s")
        body = (
            pl.col("b_traces").mean().alias("rate"),
            pl.corr(order, pl.col("b_traces")).alias("trend"),
            pl.col("b_validity").std().alias("_qual_vol"),
        )
        slices = bucket.group_by(names).agg(body) if names else bucket.select(body)

        return (
            slices.with_columns(
                (1 - pl.col("_qual_vol") / policy.stability_tol)
                .clip(0, 1)
                .fill_nan(1.0)
                .fill_null(1.0)
                .alias("stability"),
                pl.col("trend").fill_nan(0.0).fill_null(0.0).alias("trend"),
            )
            .with_columns(
                policy.acceleration(
                    pl.col("trend"), pl.col("stability"), pl.col("rate")
                ).alias("accel_score")
            )
            .select(pl.exclude(r"^_qual_vol$"))
        )

    def assess(self, *dims: Dim, grain: str = "1d") -> pl.LazyFrame:
        policy = self.policy
        names = Group.named(dims)
        win = self.windowed()
        comp = self.components(win, *dims)
        accel = self.acceleration(win, *dims, grain=grain)
        joined = (
            comp.join(accel, on=names, how="left")
            if names
            else comp.join(accel, how="cross")
        )
        # Полный набор слагаемых сохраняется в машинном результате. HTML
        # показывает только итоговые показатели, а CSV партии получает
        # validity / presence / retention и составляющие динамики без
        # повторного расчёта и без расхождения с K7.
        report = (
            "traces",
            "eff_traces",
            "loss_rate",
            "worst_rule_rate",
            "worst_rule_attr",
            "validity",
            "presence",
            "retention",
            "volume_score",
            "quality_score",
            "rate",
            "trend",
            "stability",
            "accel_score",
            "readiness",
            "verdict",
            "reason",
        )

        return (
            joined.with_columns(
                policy.readiness(
                    pl.col("volume_score"),
                    pl.col("quality_score"),
                    pl.col("accel_score"),
                ).alias("readiness")
            )
            .with_columns(
                policy.verdict(
                    pl.col("eff_traces"),
                    pl.col("loss_rate"),
                    pl.col("quality_score"),
                    pl.col("readiness"),
                ).alias("verdict"),
                policy.reason(
                    pl.col("eff_traces"),
                    pl.col("loss_rate"),
                    pl.col("quality_score"),
                    pl.col("readiness"),
                    pl.col("worst_rule_attr"),
                    pl.col("trend"),
                ).alias("reason"),
            )
            .select(names + report)
            .sort("readiness", descending=True)
        )

    def blame(self, *dims: Dim) -> pl.LazyFrame:
        names = Group.named(dims)
        report = self.quality.correctness(*dims)

        return (
            report.unpivot(
                index=names + ("n",), variable_name="attr", value_name="violation_rate"
            )
            .filter(pl.col("violation_rate") > 0)
            .sort(("violation_rate", "attr"), descending=(True, False))
        )


@dataclass(frozen=True, eq=False)
class Landscape:
    scanned: pl.LazyFrame
    spec: SpansDataSpec = SpansData

    # наличие input_request используется как формальный признак диалогового трейса
    dialog_kind: ClassVar[str] = "input_request"
    # топ повторяющихся групп: хватает, чтобы увидеть доминирующий служебный повтор, не раздувая отчёт
    repeats_top: ClassVar[int] = 8

    def is_dialog(self) -> pl.Expr:
        return (pl.col("aef_kind") == self.dialog_kind).any().over("trace_id")

    def overview(self) -> pl.LazyFrame:
        return self.scanned.select(
            pl.len().alias("spans"),
            pl.col("trace_id").n_unique().alias("traces"),
            pl.col("session_id").n_unique().alias("sessions"),
            pl.col("agent_id").n_unique().alias("agents"),
            pl.col("agent_id").first().alias("agent"),
            pl.col("nexus_distrib_ver").n_unique().alias("versions"),
            pl.col("nexus_distrib_ver").first().alias("version"),
            pl.struct("aef_kind", "span_name", "input_text", "output_text")
            .n_unique()
            .alias("unique_records"),
            pl.col("start_time_ns").min().alias("t0_ns"),
            pl.col("end_time_ns").max().alias("t1_ns"),
            Predicate.is_error.mean().alias("error_rate"),
        )

    def trace_shape(self) -> pl.LazyFrame:
        per = self.scanned.group_by("trace_id").agg(
            pl.len().alias("spans"),
            (
                (pl.col("end_time_ns").max() - pl.col("start_time_ns").min())
                / Ns.per_ms
            ).alias("dur_ms"),
            (pl.col("aef_kind") == self.dialog_kind).any().alias("dialog"),
        )

        return per.select(
            pl.len().alias("traces"),
            (pl.col("spans") == 1).sum().cast(pl.UInt64).alias("singles"),
            (pl.col("spans") == 1).mean().alias("single_share"),
            pl.col("spans").median().alias("spans_median"),
            pl.col("spans").max().alias("spans_max"),
            pl.col("dialog").mean().alias("dialog_share"),
            pl.col("dialog").sum().cast(pl.UInt64).alias("dialog_traces"),
            pl.col("dur_ms").median().alias("dur_ms_median"),
            pl.col("dur_ms").quantile(0.95).alias("dur_ms_p95"),
        )

    def composition(self) -> pl.LazyFrame:
        return (
            self.scanned.group_by("aef_kind")
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / pl.col("n").sum()).alias("share"))
            .sort(("n", "aef_kind"), descending=(True, False))
        )

    def degeneracy(self) -> pl.LazyFrame:
        pairs = self.scanned.group_by("input_text", "output_text").agg(
            pl.len().alias("n")
        )

        return pairs.select(
            pl.col("n").max().cast(pl.UInt64).alias("dominant_n"),
            (pl.col("n").max() / pl.col("n").sum()).alias("dominant_share"),
            pl.len().cast(pl.UInt64).alias("unique_pairs"),
        )

    def repeats(self) -> pl.LazyFrame:
        return (
            self.scanned.group_by("aef_kind", "span_name")
            .agg(
                pl.len().alias("n"),
                pl.col("input_text").n_unique().alias("uniq_in"),
                pl.col("output_text").n_unique().alias("uniq_out"),
            )
            .sort(("n", "aef_kind", "span_name"), descending=(True, False, False))
            .head(self.repeats_top)
        )

    def coverage(self) -> pl.LazyFrame:
        no_http = pl.col("http_method") == Sentinel.domains["none"]
        no_name = pl.col("llm_model") == Sentinel.empties["BYTE_ARRAY (UTF8)"]
        no_toks = pl.col("llm_total_tokens") == Sentinel.empties["INT64"]

        return self.scanned.select(
            Predicate.is_http.sum().cast(pl.UInt64).alias("http_spans"),
            (Predicate.is_http & no_http).sum().cast(pl.UInt64).alias("http_gap"),
            Predicate.is_llm.sum().cast(pl.UInt64).alias("llm_spans"),
            (Predicate.is_llm & no_name).sum().cast(pl.UInt64).alias("llm_no_model"),
            (Predicate.is_llm & no_toks).sum().cast(pl.UInt64).alias("llm_no_tokens"),
        )

    def traffic(self, grain: str) -> pl.LazyFrame:
        stamped = self.scanned.with_columns(
            By.time(grain).keyed(), self.is_dialog().alias("dialog")
        )

        return (
            stamped.group_by("ts")
            .agg(
                pl.col("dialog").sum().cast(pl.UInt64).alias("dialog_spans"),
                (~pl.col("dialog")).sum().cast(pl.UInt64).alias("tech_spans"),
                Predicate.is_error.mean().alias("error_rate"),
            )
            .sort("ts")
        )


@dataclass(frozen=True)
class TimeGrain:
    # Зерно подбирается по фактической длительности данных, а не по календарю:
    # выгрузка за час должна давать минутные корзины, за месяц — дневные.
    # Берётся самое мелкое зерно, дающее не больше bucket_cap корзин: выше
    # потолка точки сливаются в шум и подписи оси времени нечитаемы.
    grains: ClassVar[tuple[tuple[str, int], ...]] = (
        ("1m", 60),
        ("5m", 300),
        ("15m", 900),
        ("1h", 3_600),
        ("4h", 14_400),
        ("1d", 86_400),
    )
    bucket_cap: ClassVar[int] = 72

    @staticmethod
    def pick(span_days: float) -> str:
        span_seconds = max(0.0, span_days) * 86_400
        fitting = filter(
            lambda g: span_seconds / g[1] <= TimeGrain.bucket_cap, TimeGrain.grains
        )

        return next(fitting, TimeGrain.grains[-1])[0]


@dataclass(frozen=True, eq=False)
class Criteria:
    method: ClassVar[str] = "методика валидации качества данных телеметрии, v0.3.0"
    method_codes: ClassVar[tuple[str, ...]] = ("K1", "K2", "K3", "K4", "K5", "K6", "K7")
    diagnostic_code: ClassVar[str] = "K8"

    titles: ClassVar[dict[str, str]] = {
        "K1": "Соответствие схеме спецификации",
        "K2": "Значения атрибутов",
        "K3": "Инварианты трейса",
        "K4": "Уникальность ключей",
        "K5": "Отбраковка трейсов",
        "K6": "Сводная проверка целостности",
        "K7": "Готовность к мониторингу",
        "K8": "Диагностика структуры телеметрии",
    }

    # краткие описания для мини-таблиц «Проверяемые критерии» в разделах отчёта
    briefs: ClassVar[dict[str, str]] = {
        "K1": "наличие атрибутов и совместимость типов со спецификацией",
        "K2": "соответствие значений правилам атрибутов",
        "K3": "постоянство контекстных атрибутов внутри трейса",
        "K4": "уникальность пары (trace_id, span_id)",
        "K5": "доля трейсов, отбракованных по обязательным атрибутам",
        "K6": "пустые значения, блокирующие нарушения, дубли и расхождения инвариантов",
        "K7": "объём, качество и динамика относительно порогов политики",
        "K8": "структура трейсов, доля input_request, повторяемость и состав aef_kind",
    }

    # в каком разделе отчёта лежат детали проверки каждого критерия
    details: ClassVar[dict[str, str]] = {
        "K1": "schema",
        "K2": "values",
        "K3": "values",
        "K4": "values",
        "K5": "readiness",
        "K6": "values",
        "K7": "readiness",
        "K8": "kinds",
    }

    @staticmethod
    def anchor(code: str) -> str:
        return f"crit-{code.lower()}"

    @staticmethod
    def affects_verdict(problem: dict) -> bool:
        linked = str(problem.get("crit", ""))

        return any(
            map(lambda code: Criteria.anchor(code) in linked, Criteria.method_codes)
        )

    @staticmethod
    def outcomes(
        frames: dict,
        diff: dict,
        problems: tuple[dict, ...],
        policy: Policy,
        gates: dict,
    ) -> dict:
        verdict = frames["verdict"].to_dicts()[0]
        rejection = frames["rejection"]
        assess = frames["assess"]
        dropped = int(rejection.get_column("dropped").sum() or 0)
        total = dropped + int(rejection.get_column("kept").sum() or 0)
        loss = dropped / total if total else 0.0
        readiness = (
            tuple(map(str, assess.get_column("verdict").to_list()))
            if assess.height
            else ("not_ready",)
        )
        touching = lambda code: tuple(
            filter(lambda p: Criteria.anchor(code) in p.get("crit", ""), problems)
        )
        k8 = touching("K8")

        passed = lambda ok: ("пройден", "good") if ok else ("нарушен", "bad")
        # счётчик выше допуска — нарушен; ненулевой в пределах допуска — warn
        gated = lambda count, key: (
            ("нарушен", "bad")
            if count > gates.get(key, 0)
            else ("в пределах допуска", "warn")
            if count > 0
            else ("пройден", "good")
        )
        tolerated_any = any(
            0 < verdict[c] <= gates.get(k, 0)
            for c, k in (
                ("null_violations", "max_nulls"),
                ("blocking_rule_violations", "max_rules"),
                ("duplicate_keys", "max_dups"),
                ("trace_breaks", "max_breaks"),
            )
        )

        return {
            "K1": (
                ("нарушен", "bad")
                if not diff["critical_ok"]
                else ("есть замечания", "warn")
                if diff["degraded"]
                else ("пройден", "good")
            ),
            "K2": (
                passed(False)
                if verdict["blocking_rule_violations"] > gates.get("max_rules", 0)
                else ("блокирующие нарушения в пределах допуска", "warn")
                if verdict["blocking_rule_violations"] > 0
                else ("есть неблокирующие нарушения", "warn")
                if verdict["advisory_rule_violations"] > 0
                else passed(True)
            ),
            "K3": gated(verdict["trace_breaks"], "max_breaks"),
            "K4": gated(verdict["duplicate_keys"], "max_dups"),
            "K5": passed(loss <= policy.max_loss),
            "K6": (
                ("нарушен", "bad")
                if not verdict["valid"]
                else ("в пределах допусков", "warn")
                if tolerated_any
                else ("пройден", "good")
            ),
            "K7": (
                ("пройден · ready", "good")
                if all(map(lambda v: v == "ready", readiness))
                else ("частично", "warn")
                if any(map(lambda v: v in ("ready", "pilot"), readiness))
                else ("нарушен · not_ready", "bad")
            ),
            "K8": (
                ("критично", "bad")
                if any(map(lambda p: p["sev"] == "критично", k8))
                else ("важно", "warn")
                if any(map(lambda p: p["sev"] == "важно", k8))
                else ("умеренно", "muted")
                if k8
                else ("отклонений нет", "good")
            ),
        }

    @staticmethod
    def ref(code: str) -> str:
        return Ui.link(Criteria.anchor(code), f"{code} · {Criteria.titles[code]}")

    @staticmethod
    def payload(outcomes: dict) -> dict:
        return dict(
            map(
                lambda item: (
                    item[0],
                    {
                        "result": item[1][0],
                        "tone": item[1][1],
                        "title": Criteria.titles[item[0]],
                    },
                ),
                outcomes.items(),
            )
        )

    @staticmethod
    def registry(level: str, policy: Policy, gates: dict) -> tuple[dict, ...]:
        entry = lambda code, check, breach, source: {
            "anchor": Criteria.anchor(code),
            "code": code,
            "title": Criteria.titles[code],
            "check": check,
            "breach": breach,
            "source": source,
        }

        return (
            entry(
                "K1",
                "Для каждого атрибута спецификации проверяются наличие колонки и совместимость физического типа.",
                "Отсутствие или несовместимый тип обязательного атрибута останавливает анализ значений. "
                "Отклонение необязательного атрибута фиксируется отдельно.",
                f"{Criteria.method}, раздел 05 «Проверка целостности»; спецификация телеметрии УМР",
            ),
            entry(
                "K2",
                "Каждое значение проверяется по правилу своего атрибута. K2 блокируют только обязательные "
                "атрибуты; нарушения необязательных атрибутов учитываются отдельно.",
                "K2 не пройден, если число спанов, нарушивших правило хотя бы одного отдельного обязательного "
                "атрибута, превышает допуск. Такой спан также приводит к отбраковке трейса по K5.",
                f"{Criteria.method}, раздел 02 «Атрибутные профили»; спецификация телеметрии УМР",
            ),
            entry(
                "K3",
                "Внутри каждого trace_id проверяется постоянство идентификаторов агента и сессии, "
                "версии дистрибутива и служебного контекста.",
                "Если внутри одного trace_id меняется хотя бы один контекстный атрибут, K3 не пройден.",
                f"{Criteria.method}, раздел 04 «Согласованность инвариантов трассы»",
            ),
            entry(
                "K4",
                "Пара (trace_id, span_id) уникальна во всей выгрузке.",
                "Повтор пары ключей — дубликат спана.",
                f"{Criteria.method}, раздел 05 «Проверка целостности»",
            ),
            entry(
                "K5",
                "Трейс отбраковывается целиком, если хотя бы в одном его спане нарушено правило хотя бы одного "
                f"обязательного атрибута ({len(SpansData.mandatory_names)} из {len(SpansData.attrs)}).",
                f"Доля отбракованных трейсов выше допуска L_max = {Fmt.pct(policy.max_loss, 0)} "
                f"(из политики значимости «{level}»).",
                f"{Criteria.method}, разделы 03 «Отбраковка трасс» и 11 «Уровень значимости»",
            ),
            entry(
                "K6",
                "С допусками сравниваются четыре счётчика: пустые значения, максимальное число спанов с "
                "нарушением правила одного обязательного атрибута, дубли пары (trace_id, span_id) и трейсы "
                "с расхождениями инвариантов.",
                "K6 не пройден, если хотя бы один счётчик превышает свой допуск: "
                f"пустых значений ≤ {gates.get('max_nulls', 0)}, спанов с нарушением одного обязательного "
                f"атрибута ≤ {gates.get('max_rules', 0)}, "
                f"дублей ≤ {gates.get('max_dups', 0)}, рассогласований ≤ {gates.get('max_breaks', 0)}.",
                f"{Criteria.method}, раздел 05 «Проверка целостности»",
            ),
            entry(
                "K7",
                f"Обязательные условия допуска: эффективных трейсов ≥ {Fmt.num(policy.min_eff_traces)}, "
                f"потери ≤ {Fmt.pct(policy.max_loss, 0)}, качество Q ≥ {policy.min_quality}. Затем показатель "
                "готовности R, рассчитанный из объёма, качества и динамики притока, сравнивается с порогами: "
                f"R ≥ {policy.ready_at} — ready (мониторинг), R ≥ {policy.pilot_at} — pilot, иначе not_ready.",
                "Если обязательное условие не выполнено или R ниже порога, результат K7 — not_ready; "
                "расчётная причина указывается в таблице.",
                f"{Criteria.method}, разделы 06–09 и 11; параметры — политика значимости «{level}»",
            ),
            entry(
                "K8",
                "Рассчитываются доля односпановых трейсов, доля трейсов с aef_kind=input_request, максимальная доля "
                "одной пары input/output и количество спанов каждого aef_kind.",
                f"Односпановых трейсов ≥ {Fmt.pct(Triage.single_cap, 0)} — критично; диалоговых трейсов < "
                f"{Fmt.pct(Triage.dialog_floor, 0)} — критично; одна пара input/output встречается не менее чем в "
                f"{Fmt.pct(Triage.dominant_cap, 0)} спанов — критично; отсутствие aef_kind=input_request, "
                "aef_kind=output_request или aef_kind=start_agent — критично; отсутствие aef_kind=llm или его "
                "недостаточное количество — важно; отсутствие остальных типов — умеренно.",
                "дополнительная диагностика вне методики v0.3.0; пороги K8 не влияют на итог K1–K7",
            ),
        )


@dataclass(frozen=True, eq=False)
class Triage:
    # Пороги применяются к долям по всей выборке и отделяют системное
    # отклонение структуры телеметрии от единичного наблюдения.
    single_cap: ClassVar[float] = 0.90
    dialog_floor: ClassVar[float] = 0.05
    dominant_cap: ClassVar[float] = 0.50
    systemic_cap: ClassVar[float] = 0.50
    trend_floor: ClassVar[
        float
    ] = -0.30  # выраженное снижение притока: corr(время, объём) ниже порога
    trend_buckets: ClassVar[int] = 5  # минимум корзин, чтобы тренд не был шумом

    severity_order: ClassVar[dict[str, int]] = {
        "критично": 0,
        "важно": 1,
        "умеренно": 2,
    }

    @staticmethod
    def item(
        sev: str, title: str, scale: str, impact: str, action: str, crit: str
    ) -> dict:
        return {
            "sev": sev,
            "title": title,
            "scale": scale,
            "impact": impact,
            "action": action,
            "crit": crit,
        }

    @staticmethod
    def rule_ref(attr: str) -> str:
        return f"{Criteria.ref('K2')}, {Ui.link(f'rule-{attr}', f'правило {attr}')}"

    @staticmethod
    def structural(shape: dict, overview: dict, degeneracy: dict) -> tuple[dict, ...]:
        found = ()
        if shape["single_share"] >= Triage.single_cap:
            found += (
                Triage.item(
                    "критично",
                    "Доля односпановых трейсов превышает порог",
                    f"{Fmt.num(shape['singles'])} из {Fmt.num(shape['traces'])} трейсов ({Fmt.pct(shape['single_share'])}) "
                    f"состоят из одного спана; порог — менее {Fmt.pct(Triage.single_cap, 0)}",
                    "Для этих трейсов нельзя восстановить последовательность дочерних вызовов.",
                    "Проверьте parent_span_id дочерних спанов и наличие корневого спана aef_kind=start_agent.",
                    Criteria.ref("K8"),
                ),
            )
        if shape["dialog_share"] < Triage.dialog_floor:
            found += (
                Triage.item(
                    "критично",
                    "Доля трейсов с aef_kind=input_request ниже порога",
                    f"{Fmt.num(shape['dialog_traces'])} из {Fmt.num(shape['traces'])} трейсов ({Fmt.pct(shape['dialog_share'], 2)}) "
                    f"содержат aef_kind=input_request; порог — не менее {Fmt.pct(Triage.dialog_floor, 0)}",
                    "Выборка содержит недостаточно трейсов пользовательских запросов для оценки диалогового трафика.",
                    "Проверьте формирование спанов aef_kind=input_request для пользовательских запросов.",
                    Criteria.ref("K8"),
                ),
            )
        if degeneracy["dominant_share"] >= Triage.dominant_cap:
            found += (
                Triage.item(
                    "критично",
                    "Доля повторяющихся input/output превышает порог",
                    f"{Fmt.num(degeneracy['dominant_n'])} из {Fmt.num(overview['spans'])} спанов ({Fmt.pct(degeneracy['dominant_share'])}) "
                    f"содержат одну пару input/output при пороге менее {Fmt.pct(Triage.dominant_cap, 0)}; "
                    f"уникальных записей — {Fmt.num(overview['unique_records'])} из {Fmt.num(overview['spans'])} "
                    f"({Fmt.pct(overview['unique_records'] / overview['spans'])})",
                    "Повторяющиеся записи непропорционально влияют на агрегированные показатели выборки.",
                    "Определите источник повторяющихся input/output и исключите служебные повторы до формирования витрины.",
                    Criteria.ref("K8"),
                ),
            )

        return found

    # Единая шкала K8: вход, выход и корневой спан критичны для восстановления
    # обращения; llm важен для оценки вызовов модели; остальные типы дополняют
    # картину и имеют умеренный приоритет.
    kind_priority: ClassVar[dict[str, str]] = {
        "input_request": "критично",
        "output_request": "критично",
        "start_agent": "критично",
        "llm": "важно",
        "chain": "умеренно",
        "tool": "умеренно",
        "retriever": "умеренно",
        "kafka_produce": "умеренно",
        "kafka_consume": "умеренно",
        "other": "умеренно",
        "guard": "умеренно",
    }
    kind_order: ClassVar[tuple[str, ...]] = (
        "input_request",
        "output_request",
        "start_agent",
        "llm",
        "chain",
        "tool",
        "retriever",
        "kafka_produce",
        "kafka_consume",
        "other",
        "guard",
    )
    # input_request уже проверяется порогом доли диалоговых трейсов; отдельная
    # карточка об отсутствии дублировала бы то же отклонение.
    kind_missing_order: ClassVar[tuple[str, ...]] = tuple(
        filter(lambda kind: kind != "input_request", kind_order)
    )

    kind_impact: ClassVar[dict[str, str]] = {
        "llm": "Цепочка вызовов модели не представлена в телеметрии.",
        "start_agent": "Корневой шаг агента не представлен отдельным спаном.",
        "output_request": "Завершение пользовательского обращения не представлено отдельным спаном.",
        "tool": "Вызовы tool не представлены; влияние зависит от того, использует ли их агент.",
        "retriever": "Вызовы retriever не представлены; влияние зависит от того, использует ли их агент.",
    }

    @staticmethod
    def kinds(
        composition: "pl.DataFrame", shape: dict, overview: dict
    ) -> tuple[dict, ...]:
        counts = dict(
            map(lambda r: (str(r["aef_kind"]), r["n"]), composition.to_dicts())
        )
        absent = lambda kind: Triage.item(
            Triage.kind_priority[kind],
            f"aef_kind={kind} отсутствует",
            f"aef_kind={kind}: 0 из {Fmt.num(overview['spans'])} спанов",
            Triage.kind_impact.get(
                kind, f"Спаны aef_kind={kind} не представлены в выборке."
            ),
            f"Проверьте формирование спанов aef_kind={kind} на стороне источника телеметрии.",
            Criteria.ref("K8"),
        )
        found = tuple(
            map(absent, filter(lambda k: not counts.get(k), Triage.kind_missing_order))
        )

        llm_n = counts.get("llm", 0)
        if llm_n and shape["dialog_traces"] and llm_n < shape["dialog_traces"]:
            found += (
                Triage.item(
                    "важно",
                    "Спанов aef_kind=llm меньше, чем диалоговых трейсов",
                    f"aef_kind=llm: {Fmt.num(llm_n)} спанов на {Fmt.num(shape['dialog_traces'])} диалоговых трейсов "
                    f"({llm_n / shape['dialog_traces']:.2f} на трейс)",
                    "Количество вызовов модели в выборке ниже количества трейсов с input_request.",
                    "Сопоставьте вызовы модели со спанами aef_kind=llm и обеспечьте их передачу при экспорте.",
                    Criteria.ref("K8"),
                ),
            )

        return found

    @staticmethod
    def extraction(coverage: dict) -> tuple[dict, ...]:
        found = ()
        http_gap = (
            coverage["http_gap"] / coverage["http_spans"]
            if coverage["http_spans"]
            else 0.0
        )
        llm_gap = (
            coverage["llm_no_model"] / coverage["llm_spans"]
            if coverage["llm_spans"]
            else 0.0
        )
        if http_gap >= Triage.systemic_cap:
            found += (
                Triage.item(
                    "умеренно",
                    "Не заполнены атрибуты HTTP",
                    f"{Fmt.num(coverage['http_gap'])} из {Fmt.num(coverage['http_spans'])} спанов input_request/output_request "
                    f"({Fmt.pct(http_gap)}) без http_method, http_path или http_status_code",
                    "Без этих атрибутов нельзя полностью анализировать HTTP-обмен; K2 не блокируется.",
                    "Заполните http_method, http_path и http_status_code для input_request и output_request.",
                    Triage.rule_ref("http_method"),
                ),
            )
        if llm_gap >= Triage.systemic_cap:
            found += (
                Triage.item(
                    "умеренно",
                    "Не заполнены атрибуты aef_kind=llm",
                    f"пустой llm_model: {Fmt.num(coverage['llm_no_model'])} из {Fmt.num(coverage['llm_spans'])} спанов aef_kind=llm "
                    f"({Fmt.pct(llm_gap)}); llm_total_tokens = -1: {Fmt.num(coverage['llm_no_tokens'])}",
                    "Без этих атрибутов нельзя полностью анализировать модель и токены; K2 не блокируется.",
                    "Заполните llm_model, llm_prompt_tokens, llm_completion_tokens, llm_total_tokens "
                    "и llm_temperature для aef_kind=llm.",
                    Triage.rule_ref("llm_model"),
                ),
            )

        return found

    @staticmethod
    def worst_rows(blame: "pl.DataFrame", spec: SpansDataSpec) -> "pl.DataFrame":
        return blame.filter(
            pl.col("attr").is_in(tuple(spec.mandatory_names))
            & (pl.col("violation_rate") >= Triage.systemic_cap)
        ).sort(("violation_rate", "attr"), descending=(True, False))

    @staticmethod
    def integrity(
        verdict: dict,
        shape: dict,
        rejection: "pl.DataFrame",
        blame: "pl.DataFrame",
        policy: Policy,
        spec: SpansDataSpec,
        gates: dict,
    ) -> tuple[dict, ...]:
        found = ()
        dropped = int(rejection.get_column("dropped").sum() or 0)
        total = dropped + int(rejection.get_column("kept").sum() or 0)
        loss = dropped / total if total else 0.0
        worst = Triage.worst_rows(blame, spec)
        named = lambda row: (
            f"{row['attr']} — {Fmt.num(round(row['violation_rate'] * row['n']))} из "
            f"{Fmt.num(row['n'])} спанов ({Fmt.pct(row['violation_rate'])})"
        )
        if loss > policy.max_loss:
            leading = (
                f"; наибольший вклад — {named(worst.to_dicts()[0])}"
                if worst.height
                else ""
            )
            found += (
                Triage.item(
                    "критично",
                    "Доля отбракованных трейсов превышает допуск",
                    f"{Fmt.num(dropped)} из {Fmt.num(total)} трейсов ({Fmt.pct(loss)}) отбраковано при допуске ≤ "
                    f"{Fmt.pct(policy.max_loss, 0)}{leading}",
                    "Доля отбракованных трейсов превышает допуск K5.",
                    "Устраните нарушения обязательных атрибутов, перечисленные в разделе «Качество значений».",
                    Criteria.ref("K5"),
                ),
            )
        if worst.height:
            listed = "; ".join(map(named, worst.to_dicts()))
            refs = ", ".join(
                map(lambda r: Triage.rule_ref(r["attr"]), worst.to_dicts())
            )
            found += (
                Triage.item(
                    "важно",
                    "Правила обязательных атрибутов нарушены не менее чем в 50% спанов",
                    listed,
                    "Каждое нарушение обязательного атрибута отбраковывает весь трейс.",
                    "Приведите значения этих атрибутов к формату спецификации в источнике телеметрии.",
                    refs,
                ),
            )
        blocking = blame.filter(pl.col("attr").is_in(tuple(spec.mandatory_names)))
        if (
            verdict["blocking_rule_violations"] > gates.get("max_rules", 0)
            and blocking.height
        ):
            leading = blocking.sort(
                ("violation_rate", "attr"), descending=(True, False)
            ).to_dicts()[0]
            found += (
                Triage.item(
                    "важно",
                    "Доля спанов с нарушением обязательного атрибута превышает допуск K2",
                    f"{named(leading)} при допуске {Fmt.num(gates.get('max_rules', 0))} спанов",
                    "K2 сравнивает допуск с наибольшим числом нарушивших спанов по одному обязательному атрибуту.",
                    f"Исправьте значения атрибута {leading['attr']} в источнике телеметрии.",
                    f"{Criteria.ref('K2')} · {Triage.rule_ref(leading['attr'])}",
                ),
            )
        if verdict["duplicate_keys"] > 0:
            found += (
                Triage.item(
                    "важно",
                    "Дубликаты ключей (trace_id, span_id)",
                    f"{Fmt.num(verdict['duplicate_keys'])} повторов пары ключей при допуске {gates.get('max_dups', 0)}",
                    "Повторные записи изменяют счётчики и структуру трейсов.",
                    "Проверьте экспорт телеметрии на повторную отправку одних и тех же спанов.",
                    Criteria.ref("K4"),
                ),
            )
        if verdict["trace_breaks"] > 0:
            found += (
                Triage.item(
                    "важно",
                    "Инварианты расходятся внутри трейса",
                    f"{Fmt.num(verdict['trace_breaks'])} из {Fmt.num(shape['traces'])} трейсов ({Fmt.pct(verdict['trace_breaks'] / shape['traces'])}) "
                    "содержат изменения контекстных атрибутов",
                    "Расхождение не позволяет однозначно отнести спаны к одному контексту агента и версии.",
                    "Проверьте постоянство контекстных атрибутов во всех спанах трейса.",
                    Criteria.ref("K3"),
                ),
            )
        if verdict["null_violations"] > gates.get("max_nulls", 0):
            found += (
                Triage.item(
                    "важно",
                    "Количество пустых значений превышает допуск",
                    f"{Fmt.num(verdict['null_violations'])} пустых значений при допуске {gates.get('max_nulls', 0)}",
                    "Пустое значение учитывается как нарушение K6.",
                    "Заполните значения в источнике телеметрии.",
                    Criteria.ref("K6"),
                ),
            )

        return found

    @staticmethod
    def schema(diff: dict) -> tuple[dict, ...]:
        defects = len(diff["soft_bad"])

        return (
            ()
            if not defects
            else (
                Triage.item(
                    "умеренно",
                    "В схеме есть отклонения по необязательным атрибутам",
                    f"необязательных атрибутов с дефектом: {len(diff['soft_bad'])}; колонок вне спецификации: {len(diff['extra'])}; "
                    f"атрибутов совместимого типа: {sum(r[4] not in SchemaState.broken for r in diff['rows'])} из {len(diff['rows'])}",
                    "Часть необязательных атрибутов недоступна для последующих проверок.",
                    "Приведите состав и типы колонок к спецификации телеметрии.",
                    Criteria.ref("K1"),
                ),
            )
        )

    @staticmethod
    def dynamics(flow: "pl.DataFrame") -> tuple[dict, ...]:
        if flow.height < Triage.trend_buckets:
            return ()
        raw = flow.with_row_index("i").select(pl.corr("i", "traces")).item()
        trend = float(raw) if raw is not None and isfinite(float(raw)) else 0.0

        return (
            ()
            if trend >= Triage.trend_floor
            else (
                Triage.item(
                    "умеренно",
                    "Динамика притока ниже порога",
                    f"корреляция количества трейсов со временем {trend:+.2f} по {flow.height} корзинам; "
                    f"порог — не ниже {Triage.trend_floor:+.2f}",
                    "Снижение притока уменьшает объём данных в расчётном окне K7.",
                    "Проверьте полноту и стабильность экспорта телеметрии за период.",
                    Criteria.ref("K7"),
                ),
            )
        )

    @staticmethod
    def problems(
        frames: dict, diff: dict, policy: Policy, spec: SpansDataSpec, gates: dict
    ) -> tuple[dict, ...]:
        single = lambda name: frames[name].to_dicts()[0]
        found = (
            Triage.structural(single("shape"), single("overview"), single("degeneracy"))
            + Triage.kinds(frames["composition"], single("shape"), single("overview"))
            + Triage.extraction(single("coverage"))
            + Triage.integrity(
                single("verdict"),
                single("shape"),
                frames["rejection"],
                frames["blame"],
                policy,
                spec,
                gates,
            )
            + Triage.schema(diff)
            + Triage.dynamics(frames["flow"])
        )

        return tuple(sorted(found, key=lambda p: Triage.severity_order[p["sev"]]))


def _checked_kinds() -> None:
    """Роли типов спанов обязаны существовать в спецификации: переименование
    категории в Categories.aef_kind не должно молча отключать проверки K8."""
    known = frozenset(Categories.aef_kind)
    used = frozenset(Triage.kind_priority) | {Landscape.dialog_kind}
    strange = tuple(sorted(used - known))
    missing = tuple(sorted(known - used))

    if strange or missing:
        raise ValueError(
            f"приоритеты aef_kind не согласованы со спецификацией: лишние={strange}, отсутствуют={missing}"
        )


_checked_kinds()


@dataclass(frozen=True, eq=False)
class Args:
    ints: ClassVar[tuple[str, ...]] = ("window_days",)
    gates: ClassVar[tuple[str, ...]] = (
        "max_nulls",
        "max_rules",
        "max_dups",
        "max_breaks",
    )
    floats: ClassVar[tuple[str, ...]] = (
        "volume_target",
        "volume_steepness",
        "rate_target",
        "min_eff_traces",
        "max_loss",
        "min_quality",
        "ready_at",
        "pilot_at",
        "trend_gain",
        "stability_tol",
        "w_validity",
        "w_presence",
        "w_retention",
        "w_trend",
        "w_stability",
        "w_rate",
        "w_volume",
        "w_quality",
        "w_accel",
    )

    @staticmethod
    def given(params: dict, key: str) -> bool:
        return key in params and params[key] not in (None, "")

    @staticmethod
    def overrides(params: dict) -> dict:
        pick = lambda keys, fn: map(
            lambda k: (k, fn(params[k])), filter(partial(Args.given, params), keys)
        )

        return dict(chain(pick(Args.ints, int), pick(Args.floats, float)))

    # знаменатель относительного допуска: спаны или трейсы
    gate_basis: ClassVar[dict[str, str]] = {
        "max_nulls": "spans",
        "max_rules": "spans",
        "max_dups": "spans",
        "max_breaks": "traces",
    }

    @staticmethod
    def gate(key: str, raw: Any) -> int | str:
        """Допуск — целое число (штуки) или доля со знаком %, например 0.1%.

        Дробь без % отклоняется: неясно, штуки это или доля.
        """
        text = str(raw).strip()
        if text.endswith("%"):
            body = text[:-1].strip()
            try:
                share = float(body)
            except ValueError as error:
                raise ValueError(
                    f"{key}: доля должна быть числом со знаком %, получено {text!r}"
                ) from error
            if not (0.0 <= share <= 100.0):
                raise ValueError(f"{key}: доля должна быть в пределах 0%..100%")
            return f"{body}%"
        try:
            return int(text)
        except ValueError as error:
            raise ValueError(
                f"{key}: допуск должен быть целым числом (штуки) или долей со знаком %, "
                f"например 0.1%; получено {text!r}"
            ) from error

    @staticmethod
    def thresholds(params: dict) -> dict:
        return dict(
            map(
                lambda k: (k, Args.gate(k, params[k])),
                filter(partial(Args.given, params), Args.gates),
            )
        )

    @staticmethod
    def resolve_gates(gates: dict, spans: int, traces: int) -> dict:
        """Доли округляет вверх по фактическому объёму; штуки оставляет как есть."""
        basis = {"spans": int(spans or 0), "traces": int(traces or 0)}

        def resolve(key: str, value: int | str) -> int:
            if not isinstance(value, str):
                return int(value)
            raw = Decimal(value[:-1]) * Decimal(basis[Args.gate_basis[key]]) / 100

            return int(raw.to_integral_value(rounding=ROUND_CEILING))

        return dict(map(lambda kv: (kv[0], resolve(*kv)), gates.items()))

    @staticmethod
    def gate_note(name: str, gates: dict, spec: dict, spans: int, traces: int) -> str:
        """«0.1% от 4 400 спанов = 5 спанов» для доли, иначе «5»."""
        shown = Fmt.num(gates.get(name, 0))
        raw = spec.get(name)
        if not isinstance(raw, str):
            return shown
        basis = Args.gate_basis[name]
        total = spans if basis == "spans" else traces
        unit = "спанов" if basis == "spans" else "трейсов"

        return f"{raw} от {Fmt.num(total)} {unit} = {shown} {unit}"


@dataclass(frozen=True, eq=False)
class Report:
    title: ClassVar[str] = "Отчёт о качестве телеметрии агента"
    # верхние строки таблицы атрибутов: дальше идут меньшие доли нарушений
    blame_top: ClassVar[int] = 12
    # Эти служебные поля участвуют в расчёте K3, но не выводятся владельцу.
    hidden_attributes: ClassVar[frozenset[str]] = frozenset(
        ("service_name", "service_version")
    )
    significance_hint: ClassVar[str] = "характеристика агента по методике"

    @staticmethod
    def kv(mapping: dict) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "показатель": map(str, mapping.keys()),
                "значение": map(str, mapping.values()),
            }
        )

    @staticmethod
    def action_items(problems: tuple[dict, ...]) -> tuple[dict, ...]:
        return tuple(filter(Criteria.affects_verdict, problems))

    @staticmethod
    def diagnostic_items(problems: tuple[dict, ...]) -> tuple[dict, ...]:
        return tuple(
            filter(lambda problem: not Criteria.affects_verdict(problem), problems)
        )

    @staticmethod
    def when(ns: None | int) -> str:
        from datetime import datetime, timezone

        return (
            "—"
            if ns is None
            else datetime.fromtimestamp(ns / Ns.per_second, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
        )

    @staticmethod
    def moment(ns: None | float) -> str:
        from datetime import datetime, timezone

        return (
            "—"
            if ns is None
            else datetime.fromtimestamp(ns / Ns.per_second, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    @staticmethod
    def type_name(shown: str) -> str:
        return (
            "категория из списка (см. правило атрибута)"
            if shown.startswith("Enum")
            else shown
        )

    @staticmethod
    def status_title(light: str, valid: bool, scope: str = "full") -> str:
        if scope == "quality":
            return {
                "green": "качество данных подтверждено",
                "amber": "качество данных с отклонениями",
            }.get(light, "качество данных не пройдено")

        return (
            "готов к мониторингу"
            if light == "green"
            else "готов с отклонениями"
            if light == "amber"
            else "не пройдены проверки качества данных"
            if not valid
            else "не готов к мониторингу"
        )

    @staticmethod
    def probes(codes: tuple[str, ...], outcomes: dict) -> str:
        label = "Диагностика" if codes == (Criteria.diagnostic_code,) else "Критерии"
        links = " · ".join(map(lambda c: Criteria.ref(c), codes))

        return f'<p class="criterion-links"><b>{label}:</b> {links}</p>'

    @staticmethod
    def verdict_section(frames: dict, outcomes: dict, scope: str = "full") -> str:
        verdict = frames["verdict"].to_dicts()[0]
        rejection = frames["rejection"]
        assess = frames["assess"]
        dropped = int(rejection.get_column("dropped").sum() or 0)
        kept = int(rejection.get_column("kept").sum() or 0)
        loss = dropped / (dropped + kept) if (dropped + kept) else 0.0
        best = (
            assess.sort("readiness", descending=True).to_dicts()[0]
            if assess.height
            else {}
        )
        quality_codes = Criteria.method_codes[:-1]
        quality_tones = tuple(map(lambda code: outcomes[code][1], quality_codes))
        quality_tone = (
            "bad"
            if "bad" in quality_tones
            else "warn"
            if "warn" in quality_tones
            else "good"
        )
        quality_state = (
            "Проверки не пройдены"
            if quality_tone == "bad"
            else "Есть неблокирующие отклонения"
            if quality_tone == "warn"
            else "Проверки пройдены"
        )
        quality_flags = tuple(
            filter(lambda code: outcomes[code][1] != "good", quality_codes)
        )
        failed_base = tuple(
            filter(
                lambda code: outcomes[code][1] == "bad" and code != "K6", quality_codes
            )
        )
        quality_reason = (
            f"Не пройдены: {', '.join(failed_base)}; K6 также не пройден."
            if failed_base and outcomes["K6"][1] == "bad"
            else f"Не пройден: {quality_flags[0]}."
            if len(quality_flags) == 1 and quality_tone == "bad"
            else f"Отклонения: {', '.join(quality_flags)}."
            if quality_flags
            else ""
        )
        quality_facts = (
            f"K2: до {Fmt.num(verdict['blocking_rule_violations'])} спанов по одному обязательному атрибуту, "
            f"{Fmt.num(verdict['advisory_rule_violations'])} нарушений необязательных атрибутов · "
            f"K3: {Fmt.num(verdict['trace_breaks'])} трейсов · K4: {Fmt.num(verdict['duplicate_keys'])} · "
            f"K5: {Fmt.pct(loss)} · K6: {Fmt.num(verdict['null_violations'])} пустых значений"
        )
        quality_links = "<span>Критерии:</span> " + " · ".join(
            map(lambda code: Ui.link(Criteria.anchor(code), code), quality_codes)
        )

        readiness = best.get("readiness")
        readiness_raw = str(best.get("verdict", "not_ready"))
        readiness_tone = Chip.readiness.get(readiness_raw, "bad")
        readiness_state = {
            "ready": "Готов к мониторингу",
            "pilot": "Пилотный режим",
            "not_ready": "Не готов к мониторингу",
        }.get(readiness_raw, readiness_raw)
        readiness_reason = (
            "Причина: выполнены условия допуска и порог ready."
            if readiness_raw == "ready"
            else "Причина: выполнены условия pilot, но не достигнут порог ready."
            if readiness_raw == "pilot"
            else f"Причина: {best.get('reason', 'нет расчёта')}."
        )
        readiness_facts = (
            f"Эффективных трейсов: {Fmt.num(best.get('eff_traces'))} · "
            f"потери: {Fmt.pct(best.get('loss_rate'))} · Q = {Fmt.score(best.get('quality_score'))}"
        )
        readiness_links = (
            Criteria.ref("K7") + " · " + Ui.link("k7-method", "Параметры методики K7")
        )

        signals = Ui.signal(
            "Качество данных · K1–K6",
            quality_state,
            quality_tone,
            "",
            quality_reason,
            quality_facts,
            quality_links,
        ) + Ui.signal(
            "Готовность к мониторингу · K7",
            readiness_state,
            readiness_tone,
            f"R = {Fmt.score(readiness)}",
            readiness_reason,
            readiness_facts,
            readiness_links,
        )

        lead = (
            "Качество данных и готовность к мониторингу оцениваются отдельно. "
            "Итоговый светофор рассчитан по качеству данных K1–K6; "
            "K7 не учитывается в итоге и приведён справочно."
            if scope == "quality"
            else "Качество данных и готовность к мониторингу оцениваются отдельно."
        )

        return Ui.section(
            "Итог проверки",
            lead,
            f'<div class="signal-grid">{signals}</div>',
            "verdict",
        )

    @staticmethod
    def meta_section(level: str, policy: Policy, frames: dict, grain: str) -> str:
        overview = frames["overview"].to_dicts()[0]
        shape = frames["shape"].to_dicts()[0]
        days = (
            (overview["t1_ns"] - overview["t0_ns"]) / Ns.per_day
            if overview["t0_ns"] is not None and overview["t1_ns"] is not None
            else 0.0
        )
        agent = (
            str(overview["agent"] or "—")
            if overview["agents"] == 1
            else f"несколько ({overview['agents']})"
        )
        version = (
            str(overview["version"] or "—")
            if overview["versions"] <= 1
            else f"несколько ({overview['versions']})"
        )
        cards = (
            ("agent_id", agent, ""),
            ("версия дистрибутива", version, ""),
            ("степень значимости", level, Report.significance_hint),
            (
                "спанов",
                Fmt.num(overview["spans"]),
                f"уникальных записей: {Fmt.pct(overview['unique_records'] / overview['spans'])}"
                if overview["spans"]
                else "",
            ),
            (
                "трейсов",
                Fmt.num(overview["traces"]),
                f"сессий: {Fmt.num(overview['sessions'])}",
            ),
            (
                "диалоговых трейсов",
                Fmt.pct(shape["dialog_share"]),
                f"{Fmt.num(shape['dialog_traces'])} с aef_kind=input_request",
            ),
            (
                "период данных",
                f"{days:.0f} дн.",
                f"{Report.when(overview['t0_ns'])} … {Report.when(overview['t1_ns'])}",
            ),
            (
                "окно K7",
                f"{policy.window_days} дн.",
                f"корзина расчёта A: {READINESS_GRAIN}; корзина графиков: {grain}",
            ),
        )

        return Ui.section(
            "Объект и охват проверки",
            "Идентификаторы агента и фактический состав выборки.",
            Ui.cards(cards),
            "meta",
        )

    @staticmethod
    def problems_section(problems: tuple[dict, ...]) -> str:
        actions = Report.action_items(problems)
        if not actions:
            body = Ui.callout("good", "<b>Отклонения K1–K7 не выявлены.</b>")

            return Ui.section(
                "Отклонения и действия",
                "Только отклонения K1–K7, которые учитываются в итоговом результате.",
                body,
                "problems",
            )

        cards = str().join(
            starmap(lambda i, p: Ui.problem(i + 1, p), enumerate(actions))
        )
        return Ui.section(
            "Отклонения и действия",
            "Отклонения K1–K7 в порядке приоритета: критично, важно, умеренно.",
            cards,
            "problems",
        )

    @staticmethod
    def diagnostics_section(problems: tuple[dict, ...]) -> str:
        diagnostics = Report.diagnostic_items(problems)
        lead = "Отклонения K8 отсортированы по приоритету и не влияют на итог K1–K7."
        if not diagnostics:
            body = Ui.callout("good", "<b>Отклонения K8 не выявлены.</b>")
        else:
            rows = pl.DataFrame(
                tuple(
                    map(
                        lambda problem: {
                            "приоритет": Chip.of(problem["sev"], Chip.severity),
                            "отклонение": problem["title"],
                            "факт": problem["scale"],
                            "влияние": problem["impact"],
                            "действие": problem["action"],
                            "критерий": problem["crit"],
                        },
                        diagnostics,
                    )
                )
            )
            body = Ui.table(
                rows,
                renderers={
                    "приоритет": lambda value: value,
                    "критерий": lambda value: value,
                },
                wrap=("отклонение", "факт", "влияние", "действие"),
            )

        return Ui.section("Диагностика K8", lead, body, "k8-diagnostics")

    @staticmethod
    def deviation_note(state: str, source: str, target: str) -> str:
        return (
            "атрибут не передаётся в выгрузке"
            if state == SchemaState.absent
            else f"тип {source} нельзя привести к ожидаемому {target}"
            if state == SchemaState.incompatible
            else f"тип приведён автоматически: {source} → {Report.type_name(target)}"
        )

    @staticmethod
    def schema_frame(rows: tuple, deviations_only: bool) -> pl.DataFrame:
        visible = tuple(filter(lambda r: r[0] not in Report.hidden_attributes, rows))
        chosen = (
            tuple(filter(lambda r: r[4] != SchemaState.present, visible))
            if deviations_only
            else visible
        )
        record = lambda r: {
            "атрибут": r[0],
            "обязательный": "да" if r[1] else "нет",
            "статус": r[4],
            "пояснение": (
                Report.deviation_note(r[4], r[2], r[3])
                if r[4] != SchemaState.present
                else "тип совпадает со спецификацией"
            ),
        }

        return pl.DataFrame(tuple(map(record, chosen)))

    @staticmethod
    def schema_table(rows: tuple, deviations_only: bool) -> str:
        frame = Report.schema_frame(rows, deviations_only)

        return (
            Ui.table(
                frame,
                renderers={"статус": lambda v: Chip.of(str(v), Chip.schema)},
                wrap=("пояснение",),
            )
            if frame.height
            else ""
        )

    @staticmethod
    def schema_section(diff: dict, outcomes: dict) -> str:
        counts = diff["counts"]
        rows = diff["rows"]
        compatible = sum(map(lambda r: r[4] not in SchemaState.broken, rows))
        mandatory = tuple(filter(lambda r: r[1], rows))
        mandatory_ok = sum(map(lambda r: r[4] not in SchemaState.broken, mandatory))
        optional = tuple(filter(lambda r: not r[1], rows))
        optional_ok = sum(map(lambda r: r[4] not in SchemaState.broken, optional))
        ratio = lambda numerator, denominator: (
            numerator / denominator if denominator else 0.0
        )
        fine = sum(
            map(
                lambda r: (
                    r[4] == SchemaState.present and r[0] not in Report.hidden_attributes
                ),
                rows,
            )
        )
        cards = (
            (
                "атрибутов соответствует схеме",
                Fmt.pct(ratio(compatible, len(rows)), 0),
                f"{compatible} из {len(rows)}",
            ),
            (
                "обязательных соответствует",
                Fmt.pct(ratio(mandatory_ok, len(mandatory)), 0),
                f"{mandatory_ok} из {len(mandatory)}",
            ),
            (
                "необязательных соответствует",
                Fmt.pct(ratio(optional_ok, len(optional)), 0),
                f"{optional_ok} из {len(optional)}",
            ),
            (
                "колонок вне спецификации",
                Fmt.num(counts["лишних колонок"]),
                "в проверке не участвуют",
            ),
        )

        verdict = (
            Ui.callout(
                "bad",
                "<b>K1 не пройден.</b> Обязательные атрибуты отсутствуют или имеют "
                "несовместимый тип; анализ значений остановлен.",
            )
            if not diff["critical_ok"]
            else Ui.callout(
                "warn" if diff["degraded"] else "good",
                f"<b>K1 пройден.</b> Обязательным требованиям соответствуют "
                f"{mandatory_ok} из {len(mandatory)} атрибутов.",
            )
        )

        deviations = Report.schema_table(rows, deviations_only=True)
        listed = (
            deviations
            + Ui.note(f"Остальные {fine} атрибутов: тип совпадает со спецификацией.")
            if deviations
            else Ui.note("Все атрибуты спецификации переданы, типы совпадают.")
        )
        extra = (
            Ui.note(
                "Колонки вне спецификации (в проверке не участвуют): "
                + ", ".join(diff["extra"])
            )
            if diff["extra"]
            else ""
        )
        return Ui.section(
            "Проверка схемы",
            "K1: наличие атрибутов и совместимость физических типов со спецификацией телеметрии УМР.",
            Report.probes(("K1",), outcomes)
            + Ui.cards(cards)
            + verdict
            + listed
            + extra,
            "schema",
        )

    @staticmethod
    def culprits_frame(frames: dict) -> pl.DataFrame:
        rates = frames["correctness"].to_dicts()[0]
        counts = frames["correctness_n"].to_dicts()[0]
        described = dict(map(lambda a: (a.name, rules.describe(a)), SpansData.attrs))
        mand = SpansData.mandatory_names
        record = lambda a: {
            "группа": "блокирующие" if a.mandatory else "неблокирующие",
            "атрибут": a.name,
            "обязательный": "да" if a.name in mand else "нет",
            "влияние": "блокирует K2" if a.mandatory else "не блокирует K2",
            "нарушений": int(counts.get(a.name) or 0),
            "доля нарушений": float(rates.get(a.name) or 0.0),
            "причина: нарушено правило": described[a.name],
            "критерий": f"{Ui.link(Criteria.anchor('K2'), 'K2')} · {Ui.link(f'rule-{a.name}', 'правило')}",
        }
        shown_attrs = tuple(
            filter(lambda a: a.name not in Report.hidden_attributes, SpansData.attrs)
        )
        rows = tuple(filter(lambda r: r["нарушений"] > 0, map(record, shown_attrs)))

        return (
            pl.DataFrame(rows).sort("доля нарушений", descending=True)
            if rows
            else pl.DataFrame()
        )

    @staticmethod
    def attribute_summary_frame(frames: dict) -> pl.DataFrame:
        violations = frames["correctness"].to_dicts()[0]
        violation_counts = frames["correctness_n"].to_dicts()[0]
        empty_rates = frames["nulls"].to_dicts()[0]
        empty_counts = frames["nulls_n"].to_dicts()[0]
        schema = dict(map(lambda row: (row[0], row[4]), frames["_diff"]["rows"]))
        mandatory = SpansData.mandatory_names
        record = lambda a: {
            "атрибут": a.name,
            "обязательный": "да" if a.name in mandatory else "нет",
            "статус схемы": schema[a.name],
            "нарушений правил": int(violation_counts.get(a.name) or 0),
            "доля нарушений": float(violations.get(a.name) or 0.0),
            "пустых значений": int(empty_counts.get(a.name) or 0),
            "доля пустых значений": float(empty_rates.get(a.name) or 0.0),
            "критерии": (
                f"{Ui.link(Criteria.anchor('K1'), 'K1')} · "
                f"{Ui.link(Criteria.anchor('K2'), 'K2')} · "
                f"{Ui.link(f'rule-{a.name}', 'правило')}"
            ),
        }
        shown_attrs = tuple(
            filter(lambda a: a.name not in Report.hidden_attributes, SpansData.attrs)
        )

        return pl.DataFrame(tuple(map(record, shown_attrs)))

    @staticmethod
    def values_section(frames: dict, gates: dict, outcomes: dict) -> str:
        verdict = frames["verdict"].to_dicts()[0]
        spec = frames.get("_gates_spec", {})
        spans = int(frames["overview"].to_dicts()[0].get("spans", 0) or 0)
        traces = int(frames["shape"].to_dicts()[0].get("traces", 0) or 0)
        tolerance = lambda name: (
            f"допуск: {Args.gate_note(name, gates, spec, spans, traces)}"
        )

        rule_cards = (
            (
                "K2 · максимум нарушивших спанов по обязательному атрибуту",
                Fmt.num(verdict["blocking_rule_violations"]),
                tolerance("max_rules"),
            ),
            (
                "K2 · нарушения необязательных атрибутов",
                Fmt.num(verdict["advisory_rule_violations"]),
                "не блокируют результат K2",
            ),
            ("неприводимые значения", Fmt.num(verdict["trash_cells"]), "учтены в K2"),
        )
        integrity_cards = (
            (
                "K6 · пустые значения",
                Fmt.num(verdict["null_violations"]),
                tolerance("max_nulls"),
            ),
            (
                "K4 · дубли ключей",
                Fmt.num(verdict["duplicate_keys"]),
                tolerance("max_dups"),
            ),
            (
                "K3 · расхождения",
                Fmt.num(verdict["trace_breaks"]),
                tolerance("max_breaks"),
            ),
        )
        integrity_note = Ui.note(
            "K6 объединяет четыре блокирующих счётчика: пустые значения, нарушения K2, дубли K4 "
            "и расхождения K3. Каждый счётчик сравнивается со своим допуском."
        )

        culprits = Report.culprits_frame(frames)
        blocking = (
            culprits.filter(pl.col("группа") == "блокирующие")
            if culprits.height
            else culprits
        )
        advisory_rows = (
            culprits.filter(pl.col("группа") == "неблокирующие")
            if culprits.height
            else culprits
        )
        violation_table = lambda frame: Ui.table(
            frame.drop("группа"),
            renderers={
                "нарушений": lambda v: Fmt.num(v),
                "доля нарушений": lambda v: Bar.defect(v),
                "критерий": lambda v: v,
            },
            headers={"причина: нарушено правило": "правило"},
            wrap=("причина: нарушено правило",),
        )
        shown = blocking.head(Report.blame_top) if blocking.height else blocking
        trimmed = (
            Ui.note(
                f"Показаны {Report.blame_top} из {blocking.height} блокирующих атрибутов; "
                "полный свод правил — в разделе «Результаты по критериям»."
            )
            if blocking.height > Report.blame_top
            else ""
        )
        blocking_block = (
            ("<h3>Блокирующие нарушения K2</h3>" + violation_table(shown) + trimmed)
            if blocking.height
            else Ui.callout("good", "<b>Блокирующие нарушения K2 не выявлены.</b>")
        )
        advisory_block = (
            Ui.details(
                f"Необязательные атрибуты: неблокирующие нарушения · {advisory_rows.height}",
                violation_table(advisory_rows),
            )
            if advisory_rows.height
            else ""
        )
        blamed = blocking_block + advisory_block

        k3_result, k3_tone = outcomes["K3"]
        inv_block = (
            Ui.callout(
                k3_tone,
                f"<b>K3: {k3_result}.</b> Трейсов с расхождением хотя бы одного "
                f"контекстного атрибута: {Fmt.num(verdict['trace_breaks'])}.",
            )
            if verdict["trace_breaks"]
            else Ui.note("Контекстные атрибуты постоянны внутри каждого трейса (K3).")
        )

        attributes = Report.attribute_summary_frame(frames)
        attribute_summary = Ui.details(
            "Сводка по атрибутам",
            Ui.table(
                attributes,
                renderers={
                    "статус схемы": lambda value: Chip.of(str(value), Chip.schema),
                    "нарушений правил": lambda value: Fmt.num(value),
                    "доля нарушений": lambda value: Bar.defect(value),
                    "пустых значений": lambda value: Fmt.num(value),
                    "доля пустых значений": lambda value: Bar.defect(value),
                    "критерии": lambda value: value,
                },
            )
            + Ui.note(
                "«Пустые значения» — значения null после приведения данных к спецификации. "
                "Если для атрибута предусмотрено значение вместо null, оно подставляется до расчёта "
                "и затем проверяется по правилу атрибута."
            ),
        )

        return Ui.section(
            "Качество значений",
            "K2–K4 и K6: правила атрибутов, инварианты трейса, уникальность ключей и сводная целостность.",
            Report.probes(("K2", "K3", "K4", "K6"), outcomes)
            + "<h3>Правила атрибутов · K2</h3>"
            + Ui.cards(rule_cards)
            + blamed
            + "<h3>Целостность · K3, K4, K6</h3>"
            + Ui.cards(integrity_cards)
            + integrity_note
            + inv_block
            + attribute_summary,
            "values",
        )

    @staticmethod
    def solo(frame: pl.DataFrame, agents: int) -> pl.DataFrame:
        return (
            frame.drop("agent_id")
            if (agents == 1 and "agent_id" in frame.columns)
            else frame
        )

    @staticmethod
    def readiness_section(
        level: str, policy: Policy, frames: dict, grain: str, outcomes: dict
    ) -> str:
        agents = int(frames["overview"].to_dicts()[0]["agents"])
        rejection_frame = Report.solo(frames["rejection"], agents).with_columns(
            (1 - pl.col("dropped_rate")).alias("passed_rate")
        )
        rejection_frame = rejection_frame.select(
            tuple(
                filter(
                    lambda column: column in rejection_frame.columns,
                    ("agent_id", "n", "kept", "passed_rate", "dropped"),
                )
            )
        )
        rejection = Ui.table(
            rejection_frame,
            renderers={
                "passed_rate": lambda value: Bar.volume(value, Fmt.pct(value)),
                "n": lambda value: Fmt.num(value),
                "dropped": lambda value: Fmt.num(value),
                "kept": lambda value: Fmt.num(value),
            },
            headers={
                "agent_id": "агент",
                "n": "всего трейсов",
                "kept": "прошли обязательные правила",
                "passed_rate": "доля прошедших",
                "dropped": "не прошли",
            },
        )

        assess_frame = Report.solo(frames["assess"], agents)
        assess_frame = assess_frame.select(
            tuple(
                filter(
                    lambda column: column in assess_frame.columns,
                    (
                        "agent_id",
                        "traces",
                        "eff_traces",
                        "loss_rate",
                        "worst_rule_attr",
                        "volume_score",
                        "quality_score",
                        "accel_score",
                        "readiness",
                        "verdict",
                        "reason",
                    ),
                )
            )
        )
        assess = Ui.table(
            assess_frame,
            renderers={
                "traces": lambda v: Fmt.num(v),
                "eff_traces": lambda v: Fmt.num(v),
                "loss_rate": lambda v: Bar.defect(v),
                "volume_score": lambda v: Fmt.score(v),
                "quality_score": lambda v: Fmt.score(v),
                "accel_score": lambda v: Fmt.score(v),
                "readiness": lambda v: Bar.volume(v, Fmt.score(v)),
                "verdict": lambda v: Chip.of(str(v), Chip.readiness),
            },
            headers={
                "agent_id": "агент",
                "traces": "трейсов",
                "eff_traces": "эффективных",
                "loss_rate": "потери",
                "worst_rule_attr": "макс. доля нарушений",
                "volume_score": "объём V",
                "quality_score": "качество Q",
                "accel_score": "динамика A",
                "readiness": "готовность R",
                "verdict": "вердикт",
                "reason": "причина",
            },
        )

        coeffs = {
            "степень значимости": level,
            "окно анализа": f"{policy.window_days} дн.",
            "эталонный объём N*": Fmt.num(policy.volume_target),
            "минимум эффективных трейсов N_min": Fmt.num(policy.min_eff_traces),
            "максимальная доля потерь L_max": Fmt.pct(policy.max_loss),
            "минимальное качество Q_min": Fmt.score(policy.min_quality),
            "порог ready": Fmt.score(policy.ready_at),
            "порог pilot": Fmt.score(policy.pilot_at),
            "крутизна объёма k": policy.volume_steepness,
            "усиление тренда g": policy.trend_gain,
            "эталонный темп r*": Fmt.num(policy.rate_target),
            "допуск разброса θ": policy.stability_tol,
            "веса качества w_v / w_p / w_r": f"{policy.w_validity} / {policy.w_presence} / {policy.w_retention}",
            "веса динамики w_τ / w_s / w_ρ": f"{policy.w_trend} / {policy.w_stability} / {policy.w_rate}",
            "веса готовности w_V / w_Q / w_A": f"{policy.w_volume} / {policy.w_quality} / {policy.w_accel}",
        }

        formulas = {
            "Q — балл качества": f"{policy.w_validity}·validity + {policy.w_presence}·presence + {policy.w_retention}·retention",
            "V — балл объёма": f"σ({policy.volume_steepness}·(log10(N_eff+1) − log10({policy.volume_target:.0f})))",
            "A — динамика": f"{policy.w_trend}·σ({policy.trend_gain}·τ) + {policy.w_stability}·stab + {policy.w_rate}·(1−e^(−r/{policy.rate_target:.0f}))",
            "R — готовность": f"{policy.w_volume}·V + {policy.w_quality}·Q + {policy.w_accel}·A",
            "обязательные условия допуска": f"[N_eff ≥ {policy.min_eff_traces:.0f}] ∧ [потери ≤ {policy.max_loss}] ∧ [Q ≥ {policy.min_quality}]",
            "вердикт": f"условия ∧ R ≥ {policy.ready_at} → ready · условия ∧ R ≥ {policy.pilot_at} → pilot · иначе not_ready",
        }

        how = Ui.details(
            "Параметры методики K7",
            Ui.table(Report.kv(coeffs))
            + Ui.table(Report.kv(formulas))
            + Ui.note(
                "σ(x)=1/(1+e^−x); stab=clip₍₀,₁₎(1−σ_Q/θ); τ=corr(индекс корзины, приток). "
                "Параметры определяются степенью значимости и настройками запуска."
            ),
            opened=True,
            anchor="k7-method",
        )

        overview = frames["overview"].to_dicts()[0]
        span_ms = ((overview["t1_ns"] or 0) - (overview["t0_ns"] or 0)) / Ns.per_ms
        panels = Fig.panels(frames["flow"], frames["traffic"], grain)
        figures = str().join(map(lambda pair: Ui.figure(pair[0], pair[1]), panels))
        charts = (
            "<h3>Иллюстрации динамики притока · K7</h3>"
            + (
                f'<div class="fig-grid panels-{len(panels)}">{figures}</div>'
                if figures
                else Ui.note(
                    f"Графики не сформированы. Временных корзин: {frames['flow'].height}; "
                    f"зерно: {grain}; длительность выборки: {Fmt.ms(span_ms)}."
                )
            )
            + Ui.note(
                f"Графики агрегированы с адаптивной корзиной {grain} (не более {TimeGrain.bucket_cap} точек). "
                f"Числа trend, stability и A в таблице K7 рассчитаны отдельно на корзине {READINESS_GRAIN}."
            )
        )

        return Ui.section(
            "Отбраковка и готовность",
            "K5 оценивает долю отбракованных трейсов; K7 отдельно сопоставляет объём, качество и динамику с порогами политики.",
            Report.probes(("K5", "K7"), outcomes)
            + "<h3>Соответствие трейсов обязательным правилам · K5</h3>"
            + rejection
            + "<h3>Готовность · K7</h3>"
            + assess
            + Ui.note(
                f"K7 использует окно последних {policy.window_days} дней. Таблица K5 рассчитана по всей выборке; "
                "поэтому доли могут различаться."
            )
            + charts
            + how,
            "readiness",
        )

    @staticmethod
    def criteria_section(
        level: str, policy: Policy, gates: dict, outcomes: dict
    ) -> str:
        registered = Criteria.registry(level, policy, gates)

        def summary_row(criterion: dict) -> dict:
            code = criterion["code"]
            outcome = outcomes[code]
            return {
                "код": code,
                "критерий": criterion["title"],
                "проверка": Criteria.briefs[code],
                "результат": Chip.render(*outcome),
                "детали": (
                    Ui.link(Criteria.details[code], "открыть")
                    if outcome[1] != "muted"
                    else "—"
                ),
            }

        summary = pl.DataFrame(tuple(map(summary_row, registered)))
        table = Ui.table(
            summary,
            renderers={
                "код": lambda value: (
                    Ui.link(Criteria.details[str(value)], str(value))
                    if outcomes[str(value)][1] != "muted"
                    else str(value)
                ),
                "результат": lambda value: value,
                "детали": lambda value: value,
            },
            row_anchor=lambda row: Criteria.anchor(row["код"]),
            wrap=("критерий", "проверка"),
        )
        blocks = Ui.details(
            "Критерии и пороги",
            str().join(
                map(
                    lambda criterion: Ui.criterion(
                        f"rule-{criterion['code'].lower()}",
                        criterion["code"],
                        criterion["title"],
                        criterion["check"],
                        criterion["breach"],
                        criterion["source"],
                        Chip.render(*outcomes[criterion["code"]]),
                        (
                            "Порог диагностики"
                            if criterion["code"] == Criteria.diagnostic_code
                            else "Условие несоответствия"
                        ),
                        Criteria.anchor(criterion["code"]),
                    ),
                    registered,
                )
            ),
            opened=True,
            anchor="criteria-rules",
        )
        ruled = pl.DataFrame(
            tuple(
                map(
                    lambda r: {
                        "атрибут": r[0],
                        "обязательный": "да" if r[1] else "нет",
                        "правило валидного значения": r[2],
                    },
                    filter(
                        lambda r: r[0] not in Report.hidden_attributes,
                        rules.rules(SpansData),
                    ),
                )
            )
        )
        swept = Ui.details(
            "Правила атрибутов по спецификации телеметрии",
            Ui.table(ruled, row_anchor="rule", wrap=("правило валидного значения",))
            + Ui.note(
                "Нарушение обязательного атрибута приводит к отбраковке трейса по K5."
            ),
        )

        return Ui.section(
            "Результаты по критериям",
            "K1–K7 формируют итог по методике v0.3.0. K8 в методику не входит.",
            table + blocks + swept,
            "criteria",
        )

    @staticmethod
    def kinds_frame(
        composition: pl.DataFrame, spans: int, dialog_traces: int
    ) -> pl.DataFrame:
        counts = dict(
            map(lambda r: (str(r["aef_kind"]), r["n"]), composition.to_dicts())
        )
        state = lambda kind, n: (
            "отсутствует"
            if n == 0
            else "недостаточно"
            if kind == "llm" and dialog_traces and n < dialog_traces
            else "представлен"
        )
        row = lambda kind: {
            "тип спана (aef_kind)": kind,
            "спанов": int(counts.get(kind, 0)),
            "доля": (counts.get(kind, 0) / spans) if spans else 0.0,
            "приоритет": Chip.of(Triage.kind_priority[kind], Chip.severity),
            "состояние": state(kind, int(counts.get(kind, 0))),
        }

        return pl.DataFrame(tuple(map(row, Triage.kind_order)))

    @staticmethod
    def kinds_section(frames: dict, outcomes: dict) -> str:
        overview = frames["overview"].to_dicts()[0]
        shape = frames["shape"].to_dicts()[0]
        degeneracy = frames["degeneracy"].to_dicts()[0]
        composition = frames["composition"]

        cards = (
            (
                "спанов на трейс, медиана",
                Fmt.num(shape["spans_median"]),
                f"максимум {Fmt.num(shape['spans_max'])}",
            ),
            (
                "односпановых трейсов",
                Fmt.pct(shape["single_share"]),
                f"{Fmt.num(shape['singles'])} трейсов",
            ),
            (
                "длительность трейса, медиана",
                Fmt.ms(shape["dur_ms_median"]),
                f"p95 {Fmt.ms(shape['dur_ms_p95'])}",
            ),
            (
                "спанов с ошибкой",
                Fmt.pct(overview["error_rate"]),
                "status_code = ERROR",
            ),
            (
                "макс. доля одной пары input/output",
                Fmt.pct(degeneracy["dominant_share"]),
                "порог K8: менее 50%",
            ),
        )

        table = Ui.table(
            Report.kinds_frame(
                composition, int(overview["spans"]), int(shape["dialog_traces"])
            ),
            renderers={
                "тип спана (aef_kind)": lambda v: Ui.code(str(v)),
                "спанов": lambda v: Fmt.num(v),
                "доля": lambda v: Bar.volume(v, Fmt.pct(v)),
                "приоритет": lambda v: v,
            },
        )

        repeats = Ui.details(
            "Повторяемость input/output по группам спанов",
            Ui.table(
                frames["repeats"],
                renderers={
                    "n": lambda v: Fmt.num(v),
                    "uniq_in": lambda v: Fmt.num(v),
                    "uniq_out": lambda v: Fmt.num(v),
                },
                headers={
                    "aef_kind": "тип спана (aef_kind)",
                    "span_name": "имя спана",
                    "n": "спанов",
                    "uniq_in": "уникальных input",
                    "uniq_out": "уникальных output",
                },
            )
            + Ui.note(
                "Количество уникальных input/output сопоставляется с числом спанов в группе."
            ),
        )

        quantiles = Ui.details(
            "Квантили числовых атрибутов",
            Report.quantiles_table(frames["distribution"])
            + Ui.note(
                "q25…q95 — квантили числовых атрибутов и длительности спана; на результат K8 не влияют."
            ),
        )

        return Ui.section(
            "Распределение спанов",
            "Количество, доля, приоритет и состояние каждого aef_kind. Типы с нулевым количеством не скрываются.",
            Report.probes(("K8",), outcomes)
            + Ui.cards(cards)
            + table
            + repeats
            + quantiles,
            "kinds",
        )

    @staticmethod
    def quantiles_table(distribution: pl.DataFrame) -> str:
        timestamps = frozenset(
            map(lambda a: a.name, filter(lambda a: a.group == "time", SpansData.attrs))
        )
        chosen = lambda name: (
            Report.moment
            if name in timestamps
            else (lambda v: Fmt.ms(None if v is None else v / Ns.per_ms))
            if name == "duration_ns"
            else Fmt.num
        )
        shown = lambda row: dict(
            map(
                lambda kv: (
                    kv[0],
                    kv[1] if kv[0] == "сигнал" else chosen(row["сигнал"])(kv[1]),
                ),
                row.items(),
            )
        )
        grid = pl.DataFrame(tuple(map(shown, Grid.quantiles(distribution).to_dicts())))

        return Ui.table(grid) + Ui.note(
            "Время показано датами, длительность — в мс/с; исходные значения — в наносекундах."
        )

    @staticmethod
    def glossary_section() -> str:
        terms = {
            "спан": "запись об одном шаге выполнения: llm, HTTP-запрос, tool или шаг графа",
            "трейс": "дерево спанов одного обращения; связывается через trace_id и parent_span_id",
            "aef_kind": "тип спана: " + ", ".join(Categories.aef_kind),
            "диалоговый трейс": f"трейс, содержащий спан aef_kind={Landscape.dialog_kind}",
            "обязательный атрибут": (
                "атрибут, нарушение которого отбраковывает весь трейс "
                f"({len(SpansData.mandatory_names)} из {len(SpansData.attrs)})"
            ),
            "нарушение правила": "значение не соответствует правилу атрибута из спецификации телеметрии",
            "отбраковка": "исключение всего трейса из-за нарушения обязательного атрибута",
            "готовность R": "итоговый показатель, рассчитанный из объёма V, качества Q и динамики A и сравниваемый с порогами политики",
            "степень значимости": "характеристика агента по методике; определяет параметры расчёта K5 и K7",
            "session_id_derived": "признак: идентификатор сессии восстановлен из контекста при сборе телеметрии",
            "session_id_generated": "признак: идентификатор сессии сгенерирован при сборе, в источнике отсутствовал",
        }

        return Ui.section(
            "Термины отчёта",
            "Определения показателей и технических обозначений.",
            Ui.table(
                Report.kv(terms),
                headers={"показатель": "термин", "значение": "значение"},
            ),
            "glossary",
        )

    @staticmethod
    def stop_note(diff: dict) -> str:
        crit = diff["crit"]
        present = tuple(filter(lambda r: r[4] not in SchemaState.broken, crit))
        message = (
            "Не найдено ни одного обязательного атрибута — анализ значений остановлен."
            if not present
            else f"Обязательным требованиям схемы не соответствуют {len(crit) - len(present)} из {len(crit)} атрибутов — "
            "проверка значений остановлена. Выше приведён результат K1."
        )

        return Ui.callout(
            "bad", f"<b>{message}</b> Исправьте выгрузку и выполните проверку повторно."
        )

    @staticmethod
    def schema_meta_section(level: str, meta: dict) -> str:
        agents = int(meta.get("agents") or 0)
        agent = (
            str(meta.get("agent") or "—") if agents <= 1 else f"несколько ({agents})"
        )
        cards = (
            ("agent_id", agent, ""),
            ("строк во входе", Fmt.num(meta.get("rows")), ""),
            ("степень значимости", level, Report.significance_hint),
            ("режим проверки", "только схема", "анализ значений остановлен"),
        )
        return Ui.section(
            "Объект и охват проверки",
            "Идентификатор агента и объём входной выборки.",
            Ui.cards(cards),
            "meta",
        )

    @staticmethod
    def schema_only(
        level: str, policy: Policy, gates: dict, diff: dict, meta: dict
    ) -> str:
        # анализ значений остановлен: рассчитан только K1, остальные критерии не вычислялись
        outcomes = {
            **dict(map(lambda c: (c, ("не рассчитан", "muted")), Criteria.titles)),
            "K1": ("нарушен", "bad"),
        }
        hero = Ui.hero(
            "red",
            "Схема данных не соответствует спецификации",
            "Обязательные атрибуты отсутствуют или имеют несовместимый тип. Проверка значений не выполнялась "
            "(критерий K1).",
            (
                (
                    "атрибутов по спецификации",
                    Fmt.num(diff["counts"]["колонок по спецификации"]),
                    f"передано {diff['counts']['из них присутствует']}",
                ),
                (
                    "обязательных нарушено",
                    Fmt.num(diff["counts"]["критичных нарушено"]),
                    f"из {diff['counts']['критичных (красных)']} обязательных",
                ),
            ),
        )

        agent = str(meta.get("agent") or "—")
        verdict = Ui.section(
            "Итог проверки",
            "Выполнена только проверка K1; остальные критерии не рассчитывались.",
            hero,
            "verdict",
        )
        return Ui.doc(
            Report.title,
            f"agent_id={agent} · степень значимости {level} · {Criteria.method}",
            Report.schema_meta_section(level, meta)
            + verdict
            + Report.criteria_section(level, policy, gates, outcomes)
            + Report.schema_section(diff, outcomes)
            + Report.stop_note(diff)
            + Report.glossary_section(),
        )

    @staticmethod
    def doc(
        level: str,
        policy: Policy,
        gates: dict,
        diff: dict,
        frames: dict,
        problems: tuple[dict, ...],
        light: str,
        grain: str,
        scope: str = "full",
    ) -> str:
        frames = {**frames, "_diff": diff}
        overview = frames["overview"].to_dicts()[0]
        agent = (
            str(overview["agent"] or "—")
            if overview["agents"] == 1
            else f"несколько агентов ({overview['agents']})"
        )
        subtitle = f"agent_id={agent} · степень значимости {level} · {Criteria.method}"
        outcomes = Criteria.outcomes(frames, diff, problems, policy, gates)
        body = (
            Report.meta_section(level, policy, frames, grain)
            + Report.verdict_section(frames, outcomes, scope)
            + Report.criteria_section(level, policy, gates, outcomes)
            + Report.problems_section(problems)
            + Report.diagnostics_section(problems)
            + Report.schema_section(diff, outcomes)
            + Report.kinds_section(frames, outcomes)
            + Report.values_section(frames, gates, outcomes)
            + Report.readiness_section(level, policy, frames, grain, outcomes)
            + Report.glossary_section()
        )

        return Ui.doc(Report.title, subtitle, body)

    @staticmethod
    def empty_input(level: str) -> dict:
        body = Ui.callout(
            "bad",
            "<b>Выгрузка пуста: 0 спанов.</b> Схема корректна, но записей нет — метрики качества "
            "по пустым данным не считаются. Проверьте выборку на источнике: период (t_min / t_max), "
            "agent_id и фильтр дистрибутива, затем запустите валидацию повторно.",
        )

        return {
            "report_html": Ui.doc(
                Report.title, f"степень значимости: {level} · пустая выгрузка", body
            ),
            "status": {"value": "red", "title": "пустая выгрузка"},
            "verdict": {"error": "входные данные пусты (0 спанов)"},
        }

    @staticmethod
    def failure(message: str) -> dict:
        body = Ui.callout("bad", f"<b>Валидация не выполнена.</b> {message}")

        return {
            "report_html": Ui.doc(Report.title, "ошибка валидации", body),
            "status": {"value": "red", "title": "валидация не выполнена"},
            "verdict": {"error": message},
        }


def finalize(
    level: str,
    policy: Policy,
    gates: dict,
    diff: dict,
    frames: dict,
    grain: str,
    scope: str = "full",
) -> dict:
    """Вердикт, статус и HTML-отчёт из готовых фреймов метрик.

    Вынесено из main без изменений: полноскановый путь (см. модуль
    laim-mart-report-batch) строит те же фреймы по свёрнутой таблице и
    получает идентичный результат этим же кодом.
    """
    overview_row = frames["overview"].to_dicts()[0]
    shape_row = frames["shape"].to_dicts()[0]
    gates_spec = dict(gates)
    gates = Args.resolve_gates(
        gates_spec, overview_row.get("spans", 0), shape_row.get("traces", 0)
    )
    # valid считаем по резолвленным (в том числе относительным) допускам,
    # а не по тем, что попали в ленивый план до знания объёма.
    counts = frames["verdict"].to_dicts()[0]
    valid = bool(
        counts["null_violations"] <= gates.get("max_nulls", 0)
        and counts["blocking_rule_violations"] <= gates.get("max_rules", 0)
        and counts["duplicate_keys"] <= gates.get("max_dups", 0)
        and counts["trace_breaks"] <= gates.get("max_breaks", 0)
    )
    frames = {
        **frames,
        "verdict": frames["verdict"].with_columns(pl.lit(valid).alias("valid")),
    }
    problems = Triage.problems(frames, diff, policy, SpansData, gates)
    verdict_problems = Report.action_items(problems)
    critical = any(map(lambda p: p["sev"] == "критично", verdict_problems))
    readiness = tuple(map(str, frames["assess"].get_column("verdict").to_list()))
    ready = bool(readiness) and all(map(lambda v: v == "ready", readiness))
    pilot = any(map(lambda v: v in ("ready", "pilot"), readiness))
    advisory = int(frames["verdict"].get_column("advisory_rule_violations").item()) > 0
    # блокирующие счётчики ненулевые, но в пределах допуска — не red, но и не green
    tolerated = valid and any(
        counts[c] > 0
        for c in (
            "null_violations",
            "blocking_rule_violations",
            "duplicate_keys",
            "trace_breaks",
        )
    )
    if scope == "quality":
        # K7 в светофор не входит: отбрасываем проблемы, привязанные к нему,
        # и не смотрим на readiness.
        quality_problems = tuple(
            filter(
                lambda p: Criteria.anchor("K7") not in p.get("crit", ""),
                verdict_problems,
            )
        )
        quality_critical = any(map(lambda p: p["sev"] == "критично", quality_problems))
        light = (
            "red"
            if (not valid or quality_critical)
            else "amber"
            if (diff["degraded"] or quality_problems or advisory or tolerated)
            else "green"
        )
    else:
        light = (
            "red"
            if (not valid or critical or not pilot)
            else "amber"
            if (
                not ready
                or diff["degraded"]
                or verdict_problems
                or advisory
                or tolerated
            )
            else "green"
        )

    outcomes = Criteria.outcomes(frames, diff, problems, policy, gates)
    readiness_rows = tuple(
        map(
            lambda row: {
                **row,
                "verdict": str(row.get("verdict", "not_ready")),
                "reason": str(row.get("reason", "")),
            },
            frames["assess"].to_dicts(),
        )
    )

    degeneracy_row = frames["degeneracy"].to_dicts()[0]
    coverage_row = frames["coverage"].to_dicts()[0]
    rejection_rows = frames["rejection"].to_dicts()
    rejection_row = rejection_rows[0] if rejection_rows else {}
    composition = dict(
        map(
            lambda row: (str(row["aef_kind"]), int(row["n"] or 0)),
            frames["composition"].to_dicts(),
        )
    )
    span_total = int(overview_row.get("spans") or 0)
    span_distribution = tuple(
        map(
            lambda kind: {
                "aef_kind": kind,
                "priority": Triage.kind_priority[kind],
                "span_count": composition.get(kind, 0),
                "span_share": (
                    composition.get(kind, 0) / span_total if span_total else 0.0
                ),
                "present": composition.get(kind, 0) > 0,
            },
            Triage.kind_order,
        )
    )

    correctness = frames["correctness"].to_dicts()[0]
    correctness_n = frames["correctness_n"].to_dicts()[0]
    nulls = frames["nulls"].to_dicts()[0]
    nulls_n = frames["nulls_n"].to_dicts()[0]
    schema_states = dict(map(lambda row: (row[0], row[4]), diff["rows"]))
    visible_attrs = tuple(
        filter(lambda attr: attr.name not in Report.hidden_attributes, SpansData.attrs)
    )
    attribute_quality = tuple(
        map(
            lambda attr: {
                "attribute": attr.name,
                "group": attr.group,
                "mandatory": attr.mandatory,
                "blocks_k2": attr.mandatory,
                "schema_status": schema_states[attr.name],
                "violation_count": int(correctness_n.get(attr.name) or 0),
                "violation_rate": float(correctness.get(attr.name) or 0.0),
                "null_count": int(nulls_n.get(attr.name) or 0),
                "null_rate": float(nulls.get(attr.name) or 0.0),
            },
            visible_attrs,
        )
    )

    policy_metrics = {
        "window_days": policy.window_days,
        "volume_target": policy.volume_target,
        "volume_steepness": policy.volume_steepness,
        "min_eff_traces": policy.min_eff_traces,
        "max_loss": policy.max_loss,
        "min_quality": policy.min_quality,
        "ready_at": policy.ready_at,
        "pilot_at": policy.pilot_at,
        "trend_gain": policy.trend_gain,
        "rate_target": policy.rate_target,
        "stability_tol": policy.stability_tol,
        "w_validity": policy.w_validity,
        "w_presence": policy.w_presence,
        "w_retention": policy.w_retention,
        "w_trend": policy.w_trend,
        "w_stability": policy.w_stability,
        "w_rate": policy.w_rate,
        "w_volume": policy.w_volume,
        "w_quality": policy.w_quality,
        "w_accel": policy.w_accel,
    }
    diagnostic_thresholds = {
        "single_cap": Triage.single_cap,
        "dialog_floor": Triage.dialog_floor,
        "dominant_cap": Triage.dominant_cap,
        "systemic_cap": Triage.systemic_cap,
        "trend_floor": Triage.trend_floor,
        "trend_buckets": Triage.trend_buckets,
    }

    return {
        "report_html": Report.doc(
            level,
            policy,
            gates,
            diff,
            {**frames, "_gates_spec": gates_spec},
            problems,
            light,
            grain,
            scope,
        ),
        "status": {
            "value": light,
            "title": Report.status_title(light, valid, scope),
        },
        "verdict": {
            "schema": diff["counts"],
            "quality": frames["verdict"].to_dicts(),
            "criteria": Criteria.payload(outcomes),
            "readiness": readiness_rows,
            "metrics": {
                "overview": overview_row,
                "trace_structure": shape_row,
                "content_repetition": degeneracy_row,
                "coverage": coverage_row,
                "rejection": rejection_row,
                "policy": policy_metrics,
                "gates": gates,
                "gates_spec": gates_spec,
                "diagnostic_thresholds": diagnostic_thresholds,
            },
            "span_distribution": span_distribution,
            "attribute_quality": attribute_quality,
            "problems": tuple(
                map(
                    lambda p: {
                        "sev": p["sev"],
                        "title": p["title"],
                        "scale": p["scale"],
                        "affects_verdict": Criteria.affects_verdict(p),
                    },
                    problems,
                )
            ),
        },
    }


def main(**params) -> dict:
    try:
        dont_colorize_log = str(
            params.get("dont_colorize_log", "true")
        ).strip().lower() in ("true", "1", "yes", "on")
        cs = (
            ColorSchemeDataScience()
            if dont_colorize_log
            else ColorSchemeDataScienceCold()
        )
        cs.print_section("Валидация данных")
        spans = params.get("spans_df")
        source = str(params.get("source_path") or "")
        if spans is None and not source:
            return Report.failure(
                "источник не задан: подключите порт spans_df или заполните source_path"
            )

        level = str(params.get("significance", "C")).strip().upper()
        if level not in Significance.levels:
            return Report.failure("significance должен быть одним из A, B, C, D, E")
        scope = str(params.get("verdict_scope") or "full").strip().lower()
        if scope not in VERDICT_SCOPES:
            return Report.failure(
                "verdict_scope должен быть одним из "
                + ", ".join(VERDICT_SCOPES)
                + f", получено {scope!r}"
            )
        policy = Policy.of(level, **Args.overrides(params))
        gates = Args.thresholds(params)
        device = Device.of(params.get("device", ""))
        recast = str(params.get("recast", "true")).strip().lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

        lazy = (
            pl.from_pandas(spans).lazy()
            if spans is not None
            else pl.scan_parquet(PurePath(source).as_posix(), rechunk=False)
        )
        schema = lazy.collect_schema()
        diff = SchemaCheck.diff(schema, SpansData, Significance.of(level))

        if not diff["critical_ok"]:
            source_names = frozenset(schema.names())
            agent_expr = (
                pl.col("agent_id").cast(pl.String, strict=False)
                if "agent_id" in source_names
                else pl.lit(None, dtype=pl.String)
            )
            meta = (
                lazy.select(
                    pl.len().alias("rows"),
                    agent_expr.first().alias("agent"),
                    agent_expr.n_unique().alias("agents"),
                )
                .collect()
                .to_dicts()[0]
            )
            outcomes = {
                **dict(
                    map(lambda code: (code, ("не рассчитан", "muted")), Criteria.titles)
                ),
                "K1": ("нарушен", "bad"),
            }
            return {
                "report_html": Report.schema_only(level, policy, gates, diff, meta),
                "status": {"value": "red", "title": "проверка схемы не пройдена"},
                "verdict": {
                    "schema": diff["counts"],
                    "quality": None,
                    "criteria": Criteria.payload(outcomes),
                    "readiness": (),
                },
            }

        # Пустая выгрузка валидируема только по схеме: агрегаты по нулю строк
        # дают null, и триаж падал бы на сравнениях None с порогами.
        if int(lazy.select(pl.len()).collect().item()) == 0:
            return Report.empty_input(level)

        conformed = (
            Recast.frame(lazy, schema, SpansData)
            if recast
            else lazy.select(
                map(
                    lambda a: (
                        pl.col(a.name).cast(a.type_polars, strict=False)
                        if a.name in schema.names()
                        else pl.lit(a.empty, dtype=a.type_polars).alias(a.name)
                    ),
                    SpansData.attrs,
                )
            )
        )
        quality = Quality(conformed)
        ready = Readiness(quality, policy)
        land = Landscape(conformed)

        bounds = conformed.select(
            pl.col("start_time_ns").min().alias("t0"),
            pl.col("end_time_ns").max().alias("t1"),
        ).collect()
        span_days = max(
            0.0,
            ((bounds.item(0, "t1") or 0) - (bounds.item(0, "t0") or 0)) / Ns.per_day,
        )
        grain = TimeGrain.pick(span_days)

        lazies = {
            # относительные допуски резолвятся в finalize по фактическому объёму;
            # в ленивый план идут только абсолютные, valid там пересчитывается
            "verdict": quality.verdict(
                **{k: v for k, v in gates.items() if isinstance(v, int)}
            ),
            "consistency": quality.trace_consistency(),
            "nulls": quality.nulls(agg=pl.Expr.mean),
            "nulls_n": quality.nulls(agg=pl.Expr.sum),
            "correctness": quality.correctness(),
            "correctness_n": quality.correctness(agg=pl.Expr.sum),
            "rejection": quality.rejection(By.agent),
            "assess": ready.assess(By.agent, grain=READINESS_GRAIN),
            "blame": ready.blame(),
            "distribution": quality.distribution(),
            "flow": quality.flow(grain),
            "overview": land.overview(),
            "shape": land.trace_shape(),
            "composition": land.composition(),
            "degeneracy": land.degeneracy(),
            "repeats": land.repeats(),
            "coverage": land.coverage(),
            "traffic": land.traffic(grain),
        }
        cs.print_subsection("расчёт метрик качества")
        collect_batch_size = int(params.get("collect_batch_size") or 0)
        if collect_batch_size < 0:
            return Report.failure("collect_batch_size должен быть неотрицательным")

        def collect_metrics():
            pending = tuple(lazies.values())
            if collect_batch_size == 0:
                return pl.collect_all(pending, engine=device.engine)
            collected = []
            for offset in range(0, len(pending), collect_batch_size):
                collected.extend(
                    pl.collect_all(
                        pending[offset : offset + collect_batch_size],
                        engine=device.engine,
                    )
                )
            return tuple(collected)

        collected, m_elapsed, m_cpu, m_gpu = perf.measure_during_call(
            collect_metrics, (), {}
        )
        cs.print_performance_metrics(
            f"валидация ({len(lazies)} метрик)", m_elapsed, m_cpu, m_gpu
        )
        frames = dict(zip(lazies.keys(), collected))

        return finalize(level, policy, gates, diff, frames, grain, scope)

    except Exception as error:
        return Report.failure(f"{type(error).__name__}: {error}")
