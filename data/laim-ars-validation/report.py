from typing import ClassVar, Callable
from dataclasses import dataclass
from itertools import starmap
from math import isfinite

from html import escape
from pathlib import PurePath
from tempfile import mkdtemp

import polars as pl
import altair as al

import viz


type Frame = pl.DataFrame


@dataclass(frozen=True)
class Tone:
    good: ClassVar[str] = "#16A34A"
    warn: ClassVar[str] = "#D97706"
    bad: ClassVar[str] = "#DC2626"
    info: ClassVar[str] = "#2563EB"
    ink: ClassVar[str] = "#1F2A37"
    muted: ClassVar[str] = "#6B7280"

    good_bg: ClassVar[str] = "#F0FDF4"
    warn_bg: ClassVar[str] = "#FFFBEB"
    bad_bg: ClassVar[str] = "#FEF2F2"
    info_bg: ClassVar[str] = "#EFF6FF"

    of_light: ClassVar[dict] = {"green": good, "amber": warn, "red": bad}
    bg_of_light: ClassVar[dict] = {"green": good_bg, "amber": warn_bg, "red": bad_bg}


@dataclass(frozen=True)
class Css:
    style: ClassVar[str] = (
        "*{box-sizing:border-box}"
        "html{scroll-behavior:smooth}"
        "body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;padding:30px 34px;background:#F7F8FA;color:#1F2937;line-height:1.48}"
        ".page{width:min(1160px,100%);margin:0 auto}"
        "h1{font-size:26px;font-weight:720;margin:0 0 3px;letter-spacing:-.025em;color:#111827;overflow-wrap:anywhere}"
        "h2{font-size:18px;font-weight:680;margin:28px 0 5px;color:#111827;letter-spacing:-.01em;scroll-margin-top:16px}"
        "h3{font-size:14px;font-weight:650;margin:16px 0 6px;color:#374151}"
        ".sub{color:#64748B;font-size:13.5px;margin:0 0 12px}"
        ".lead{color:#4B5563;font-size:13.5px;margin:2px 0 10px;max-width:920px}"
        ".hero{border-radius:13px;padding:17px 20px;margin:11px 0 7px;border:1px solid #E2E8F0;background:#FFFFFF;box-shadow:0 1px 2px rgba(15,23,42,.04)}"
        ".hero .verdict{font-size:19px;font-weight:720;display:flex;align-items:center;gap:10px;margin-bottom:4px;color:#111827}"
        ".hero .why{color:#475569;font-size:13.5px;margin:0 0 5px;max-width:980px}"
        ".dot{width:14px;height:14px;border-radius:50%;display:inline-block;flex:none}"
        ".signal-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:11px 0 8px}"
        ".signal{min-width:0;background:#FFFFFF;border:1px solid #E2E8F0;border-left-width:5px;border-radius:12px;padding:15px 17px;box-shadow:0 1px 2px rgba(15,23,42,.04)}"
        ".signal .label{color:#64748B;font-size:12px;font-weight:650;margin-bottom:3px}"
        ".signal .state{color:#111827;font-size:19px;font-weight:720;line-height:1.25;margin:0 0 6px}"
        ".signal .score{color:#111827;font-size:28px;font-weight:760;line-height:1.1;margin:4px 0 7px;font-variant-numeric:tabular-nums}"
        ".signal .reason{color:#374151;font-size:13px;margin:3px 0}"
        ".signal .facts{color:#64748B;font-size:12px;margin:7px 0 0;line-height:1.5}"
        ".signal .links{font-size:12.5px;font-weight:620;margin-top:8px;display:flex;gap:12px;flex-wrap:wrap}"
        ".cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:10px 0 5px}"
        ".card{min-width:0;background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:11px 13px;box-shadow:0 1px 2px rgba(15,23,42,.025)}"
        ".card .k{color:#64748B;font-size:11.5px;font-weight:600;letter-spacing:.01em}"
        ".card .v{color:#111827;font-size:20px;font-weight:720;margin-top:2px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}"
        ".card .h{color:#64748B;font-size:11.5px;margin-top:2px}"
        ".chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:650;white-space:nowrap}"
        ".problem{background:#FFFFFF;border:1px solid #E2E8F0;border-left-width:4px;border-radius:10px;padding:13px 16px;margin:9px 0;box-shadow:0 1px 2px rgba(15,23,42,.025)}"
        ".problem .t{font-size:15px;font-weight:650;margin:0 0 6px;display:flex;align-items:center;gap:9px;flex-wrap:wrap}"
        ".problem .row{font-size:13px;margin:3px 0;color:#374151}"
        ".problem .row b{color:#111827}"
        ".tw{overflow-x:auto;border-radius:11px;border:1px solid #E2E8F0;box-shadow:0 1px 2px rgba(15,23,42,.035);margin:7px 0 9px;background:#FFFFFF;-webkit-overflow-scrolling:touch}"
        "table{border-collapse:collapse;width:100%;min-width:max-content;background:#FFFFFF}"
        "th,td{padding:8px 11px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12.75px;vertical-align:top}"
        "th{background:#F8FAFC;color:#64748B;font-size:11.5px;font-weight:650;letter-spacing:.01em;border-bottom:1px solid #E2E8F0;position:sticky;top:0}"
        "td{border-bottom:1px solid #F1F5F9}"
        "td.wrap{white-space:normal;text-align:left;min-width:240px;max-width:520px;line-height:1.42}"
        "th:first-child,td:first-child{text-align:left}"
        "tr:last-child td{border-bottom:none}"
        ".bar{display:inline-flex;align-items:center;gap:8px;justify-content:flex-end}"
        ".bar .track{width:88px;height:6px;border-radius:4px;background:#EEF2F6;overflow:hidden;flex:none}"
        ".bar .fill{height:100%;border-radius:4px}"
        "details{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:11px;padding:9px 14px;margin:9px 0}"
        "details summary{cursor:pointer;color:#374151;font-size:13px;font-weight:650;outline:none}"
        "details[open] summary{margin-bottom:8px}"
        ".note{color:#64748B;font-size:12px;margin:4px 0 10px;max-width:920px}"
        ".callout{border-radius:11px;padding:11px 14px;margin:9px 0;font-size:13px;border:1px solid}"
        "img{display:block;max-width:100%;height:auto;border-radius:10px;border:1px solid #E2E8F0;background:#FFFFFF}"
        ".fig{margin:0;min-width:0}"
        ".fig img{width:100%}"
        ".fig-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start;margin:9px 0 12px}"
        ".fig-grid.panels-3 > .fig:last-child{grid-column:1/-1;width:calc((100% - 14px)/2);justify-self:center}"
        ".fig-grid.panels-2{grid-template-columns:repeat(2,minmax(0,1fr))}"
        ".fig-grid.panels-1{grid-template-columns:1fr;max-width:820px}"
        "section{margin:0 0 8px}"
        "a{color:#1D4ED8;text-decoration:none;text-underline-offset:2px}"
        "a:hover{text-decoration:underline}"
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.93em;background:#F1F5F9;color:#0F172A;padding:1px 4px;border-radius:4px}"
        ".crit{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:11px;padding:11px 15px;margin:8px 0;scroll-margin-top:16px}"
        ".crit .t{font-size:14.5px;font-weight:650;margin:0 0 4px}"
        ".crit p{font-size:13px;margin:3px 0;color:#374151}"
        ".crit .src{color:#64748B;font-size:12px}"
        ".criterion-links{font-size:12.5px;color:#475569;margin:7px 0 10px}"
        "@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.signal-grid{grid-template-columns:1fr}}"
        "@media(max-width:720px){body{padding:20px 14px}h1{font-size:23px}h2{font-size:18px;margin-top:25px}.hero{padding:15px 16px}.hero .verdict{font-size:18px}th,td{padding:7px 9px}.bar .track{width:64px}.fig-grid{grid-template-columns:1fr}.fig-grid.panels-3 > .fig:last-child{grid-column:auto;width:100%}}"
        "@media(max-width:520px){.cards{grid-template-columns:1fr}}"
        "@media print{body{padding:0;background:#FFF}.page{width:100%}.hero,.card,.problem,.tw,details{box-shadow:none}details{break-inside:avoid}.tw{overflow:visible}a{color:inherit;text-decoration:none}}"
    )


@dataclass(frozen=True)
class Fmt:
    thin: ClassVar[str] = " "

    @staticmethod
    def num(value: None | int | float) -> str:
        if value is None:
            return "—"
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}".replace(",", Fmt.thin)

        return f"{int(value):,}".replace(",", Fmt.thin)

    @staticmethod
    def pct(share: None | float, digits: int = 1) -> str:
        return "—" if share is None else f"{share * 100:.{digits}f}%"

    @staticmethod
    def ms(value: None | float) -> str:
        if value is None:
            return "—"
        if value >= 86_400_000:
            return f"{value / 86_400_000:.1f}{Fmt.thin}дн."
        if value >= 60_000:
            return f"{value / 60_000:.1f}{Fmt.thin}мин"
        if value >= 1_000:
            return f"{value / 1_000:.1f}{Fmt.thin}с"

        return f"{value:.0f}{Fmt.thin}мс"

    @staticmethod
    def score(value: None | float) -> str:
        return "—" if value is None else f"{value:.2f}"


@dataclass(frozen=True)
class Chip:
    palette: ClassVar[dict[str, tuple[str, str]]] = {
        "good": (Tone.good, Tone.good_bg),
        "warn": (Tone.warn, Tone.warn_bg),
        "bad": (Tone.bad, Tone.bad_bg),
        "info": (Tone.info, Tone.info_bg),
        "muted": (Tone.muted, "#F3F4F6"),
    }

    severity: ClassVar[dict[str, str]] = {
        "критично": "bad",
        "важно": "warn",
        "умеренно": "muted",
    }
    readiness: ClassVar[dict[str, str]] = {
        "ready": "good",
        "pilot": "warn",
        "not_ready": "bad",
    }
    schema: ClassVar[dict[str, str]] = {
        "ок": "good",
        "приведён": "warn",
        "отсутствует": "bad",
        "несовместим": "bad",
    }

    @staticmethod
    def render(text: str, tone: str) -> str:
        color, background = Chip.palette.get(tone, Chip.palette["muted"])

        return f'<span class="chip" style="color:{color};background:{background}">{escape(text)}</span>'

    @staticmethod
    def of(text: str, mapping: dict[str, str]) -> str:
        return Chip.render(text, mapping.get(text, "muted"))


@dataclass(frozen=True)
class Bar:
    width_max: ClassVar[float] = 100.0

    @staticmethod
    def render(share: None | float, tone: str, label: str) -> str:
        if share is None:
            return escape(label)
        color, _ = Chip.palette.get(tone, Chip.palette["muted"])
        width = max(0.0, min(1.0, share)) * Bar.width_max

        return (
            f'<span class="bar"><span class="track">'
            f'<span class="fill" style="width:{width:.1f}%;background:{color}"></span></span>'
            f"{escape(label)}</span>"
        )

    @staticmethod
    def defect(share: None | float) -> str:
        tone = "muted" if not share else "warn" if share < 0.5 else "bad"

        return Bar.render(share, tone, Fmt.pct(share, 2))

    @staticmethod
    def volume(share: None | float, label: str) -> str:
        return Bar.render(share, "info", label)


@dataclass(frozen=True)
class Ui:
    @staticmethod
    def doc(title: str, subtitle: str, body: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(title)}</title><style>{Css.style}</style></head><body>"
            f'<main class="page"><h1>{escape(title)}</h1><p class="sub">{escape(subtitle)}</p>'
            f"{body}</main></body></html>"
        )

    @staticmethod
    def hero(
        light: str, verdict: str, why: str, cards: tuple[tuple[str, str, str], ...]
    ) -> str:
        color = Tone.of_light.get(light, Tone.muted)

        return (
            f'<div class="hero" style="border-left:5px solid {color}">'
            f'<div class="verdict"><span class="dot" style="background:{color}"></span>{escape(verdict)}</div>'
            f'<p class="why">{escape(why)}</p>{Ui.cards(cards)}</div>'
        )

    @staticmethod
    def signal(
        label: str,
        state: str,
        tone: str,
        score: str,
        reason: str,
        facts: str,
        links: str,
    ) -> str:
        color, _ = Chip.palette.get(tone, Chip.palette["muted"])

        return (
            f'<article class="signal" style="border-left-color:{color}">'
            f'<div class="label">{escape(label)}</div><p class="state">{escape(state)}</p>'
            + (f'<div class="score">{escape(score)}</div>' if score else "")
            + (f'<p class="reason">{escape(reason)}</p>' if reason else "")
            + (f'<p class="facts">{escape(facts)}</p>' if facts else "")
            + (f'<div class="links">{links}</div>' if links else "")
            + "</article>"
        )

    @staticmethod
    def cards(items: tuple[tuple[str, str, str], ...]) -> str:
        card = lambda k, v, hint: (
            f'<div class="card"><div class="k">{escape(k)}</div><div class="v">{v}</div>'
            + (f'<div class="h">{escape(hint)}</div>' if hint else "")
            + "</div>"
        )

        return (
            f'<div class="cards">{str().join(starmap(card, items))}</div>'
            if items
            else ""
        )

    @staticmethod
    def section(title: str, lead: str, body: str, anchor: str = "") -> str:
        keyed = f' id="{anchor}"' if anchor else ""

        return f'<section><h2{keyed}>{escape(title)}</h2><p class="lead">{escape(lead)}</p>{body}</section>'

    @staticmethod
    def link(anchor: str, text: str) -> str:
        return f'<a href="#{anchor}">{escape(text)}</a>'

    @staticmethod
    def code(text: str) -> str:
        return f"<code>{escape(text)}</code>"

    @staticmethod
    def criterion(
        anchor: str,
        code: str,
        title: str,
        check: str,
        breach: str,
        source: str,
        result: str,
        breach_label: str = "Условие несоответствия",
        link_target: str = "",
    ) -> str:
        code_label = (
            f'<a href="#{escape(link_target, quote=True)}">{escape(code)}</a>'
            if link_target
            else escape(code)
        )
        return (
            f'<div class="crit" id="{escape(anchor, quote=True)}">'
            f'<p class="t">{code_label} · {escape(title)}</p>'
            f"<p><b>Проверка:</b> {escape(check)}</p>"
            f"<p><b>{escape(breach_label)}:</b> {escape(breach)}</p>"
            f"<p><b>Результат:</b> {result}</p>"
            f'<p class="src">Основание: {escape(source)}</p></div>'
        )

    @staticmethod
    def note(text: str) -> str:
        return f'<p class="note">{escape(text)}</p>'

    @staticmethod
    def details(
        summary: str, content: str, opened: bool = False, anchor: str = ""
    ) -> str:
        keyed = f' id="{escape(anchor, quote=True)}"' if anchor else ""
        state = " open" if opened else ""

        return (
            f"<details{keyed}{state}><summary>{escape(summary)}</summary>{content}</details>"
            if content
            else ""
        )

    @staticmethod
    def callout(tone: str, html_text: str) -> str:
        color, background = Chip.palette.get(tone, Chip.palette["muted"])

        return f'<div class="callout" style="border-color:{color}40;background:{background};color:{Tone.ink}">{html_text}</div>'

    @staticmethod
    def problem(index: int, item: dict) -> str:
        severity = str(item.get("severity_label") or item["sev"])
        link_label = str(item.get("criterion_label") or "Критерий")
        chip = Chip.of(
            severity,
            {**Chip.severity, severity: Chip.severity.get(item["sev"], "muted")},
        )
        criterion = (
            f'<p class="row"><b>{escape(link_label)}:</b> {item["crit"]}</p>'
            if item.get("crit")
            else ""
        )

        return (
            f'<div class="problem" style="border-left-color:{Chip.palette[Chip.severity.get(item["sev"], "muted")][0]}">'
            f'<p class="t">{index}. {escape(item["title"])} {chip}</p>'
            f'<p class="row"><b>Факт:</b> {escape(item["scale"])}</p>'
            f'<p class="row"><b>Влияние:</b> {escape(item["impact"])}</p>'
            f'<p class="row"><b>Действие:</b> {escape(item["action"])}</p>'
            + criterion
            + "</div>"
        )

    @staticmethod
    def cell(value, column: str, renderers: dict) -> str:
        renderer = renderers.get(column)
        if renderer is not None:
            return renderer(value)
        if isinstance(value, float):
            return escape(f"{value:.4f}")

        return escape(str(value))

    @staticmethod
    def table(
        frame: Frame,
        renderers: None | dict = None,
        headers: None | dict = None,
        row_anchor: None | str | Callable[[dict], str] = None,
        wrap: tuple[str, ...] = (),
    ) -> str:
        named = renderers or {}
        titles = headers or {}
        head = (
            "<tr>"
            + str().join(
                map(lambda c: f"<th>{escape(titles.get(c, c))}</th>", frame.columns)
            )
            + "</tr>"
        )
        keyed = lambda row: (
            row_anchor(row)
            if callable(row_anchor)
            else f"{row_anchor}-{row[frame.columns[0]]}"
        )
        opened = lambda row: (
            f'<tr id="{escape(str(keyed(row)), quote=True)}">' if row_anchor else "<tr>"
        )
        cell = lambda row, c: (
            (f'<td class="wrap">' if c in wrap else "<td>")
            + Ui.cell(row[c], c, named)
            + "</td>"
        )
        line = lambda row: (
            opened(row)
            + str().join(map(lambda c: cell(row, c), frame.columns))
            + "</tr>"
        )

        return f'<div class="tw"><table>{head}{str().join(map(line, frame.to_dicts()))}</table></div>'

    @staticmethod
    def figure(chart: "al.Chart", name: str) -> str:
        try:
            image = viz.save_chart(chart, PurePath(mkdtemp()) / name)

            return (
                f'<div class="fig">{viz.Html.image(name, viz.img_to_base64(image))}</div>'
                if image
                else ""
            )
        except Exception:
            return ""


@dataclass(frozen=True)
class Fig:
    # Две панели в ряд: размер сохраняет читаемые оси и подписи.
    width: ClassVar[int] = 600
    height: ClassVar[int] = 220
    min_buckets: ClassVar[int] = 2  # с одной корзиной линия времени вырождается в точку

    # Спаны в трейсе с input_request выделены синим; спаны в остальных
    # трейсах показаны нейтральным серым.
    accent: ClassVar[str] = "#2A78D6"
    context: ClassVar[str] = "#8A9099"
    surface: ClassVar[str] = "#FFFFFF"
    grid_line: ClassVar[str] = "#ECEFF3"
    axis_ink: ClassVar[str] = "#64748B"
    title_ink: ClassVar[str] = "#111827"
    reference: ClassVar[str] = "#94A3B8"

    # подпись оси времени по зерну: внутри часа важны минуты, внутри суток —
    # часы, на длинных периодах — дата
    fine_grains: ClassVar[tuple[str, ...]] = ("1m", "5m", "15m")
    mid_grains: ClassVar[tuple[str, ...]] = ("1h", "4h")

    dialog_name: ClassVar[str] = "трейс с input_request"
    tech_name: ClassVar[str] = "трейс без input_request"

    @staticmethod
    def stamp(grain: str) -> str:
        return (
            "%H:%M:%S"
            if grain == "1m"
            else "%H:%M"
            if grain in Fig.fine_grains
            else "%d.%m %H:%M"
            if grain == "1h"
            else "%d.%m"
        )

    @staticmethod
    def timeline(grain: str, buckets: int) -> al.X:
        return al.X(
            "ts:T",
            title=None,
            axis=al.Axis(
                format=Fig.stamp(grain),
                tickCount=max(2, min(6, buckets)),
                labelAngle=0,
                labelOverlap="greedy",
                labelFlush=True,
            ),
        )

    @staticmethod
    def bar_size(buckets: int) -> int:
        # на коротких рядах столбики не должны занимать целую временную корзину
        return max(4, min(36, Fig.width // max(buckets, 1) - 4))

    @staticmethod
    def themed(chart: "al.Chart", title: str, subtitle: str = "") -> al.Chart:
        heading = {"text": title, "subtitle": subtitle} if subtitle else title

        return (
            chart.properties(title=heading, width=Fig.width, height=Fig.height)
            .configure_view(stroke=None, fill=Fig.surface)
            .configure_axis(
                gridColor=Fig.grid_line,
                gridWidth=1,
                domainColor=Fig.grid_line,
                tickColor=Fig.grid_line,
                labelColor=Fig.axis_ink,
                titleColor=Fig.axis_ink,
                labelFontSize=11,
                titleFontSize=11,
            )
            .configure_title(
                color=Fig.title_ink,
                fontSize=13,
                fontWeight=600,
                anchor="start",
                subtitleColor=Fig.axis_ink,
                subtitleFontSize=11,
                dy=-4,
            )
            .configure_legend(
                orient="top",
                direction="horizontal",
                labelColor=Fig.axis_ink,
                labelFontSize=11,
                symbolSize=80,
            )
        )

    @staticmethod
    def inflow(flow: Frame, grain: str) -> None | al.Chart:
        if flow.height < Fig.min_buckets:
            return None
        raw = flow.with_row_index("i").select(pl.corr("i", "traces")).item()
        trend = float(raw) if raw is not None and isfinite(float(raw)) else 0.0
        arrow = (
            "растёт" if trend > 0.05 else "снижается" if trend < -0.05 else "стабилен"
        )
        # Altair не сериализует Polars datetime напрямую; frame уже агрегирован
        # до корзин, поэтому передаём его через pandas без риска по памяти.
        data = flow.to_pandas()
        base = al.Chart(data).encode(x=Fig.timeline(grain, flow.height))
        volume = al.Y(
            "traces:Q",
            title="трейсов в корзине",
            scale=al.Scale(zero=True),
            axis=al.Axis(tickMinStep=1, tickCount=4),
        )
        area = base.mark_area(
            opacity=0.10, color=Fig.accent, interpolate="linear"
        ).encode(y=volume)
        line = base.mark_line(
            strokeWidth=2, color=Fig.accent, interpolate="linear", point=True
        ).encode(y=volume)
        mean = (
            al.Chart(data)
            .mark_rule(strokeDash=(4, 4), strokeWidth=1, color=Fig.reference)
            .encode(y=al.Y("mean(traces):Q", title=None))
        )
        last = (
            al.Chart(data.tail(1))
            .mark_circle(size=70, color=Fig.accent, stroke=Fig.surface, strokeWidth=2)
            .encode(x=al.X("ts:T"), y=volume)
        )

        return Fig.themed(
            area + line + mean + last,
            "Приток трейсов",
            f"тренд {arrow} (corr {trend:+.2f}) · пунктир — среднее",
        )

    @staticmethod
    def mix(traffic: Frame, grain: str) -> None | al.Chart:
        if traffic.height < Fig.min_buckets:
            return None
        flows = traffic.rename(
            {"dialog_spans": Fig.dialog_name, "tech_spans": Fig.tech_name}
        ).unpivot(
            index="ts",
            on=(Fig.dialog_name, Fig.tech_name),
            variable_name="спаны",
            value_name="n",
        )
        bars = (
            al.Chart(flows.to_pandas())
            .mark_bar(
                size=Fig.bar_size(traffic.height), stroke=Fig.surface, strokeWidth=1
            )
            .encode(
                x=Fig.timeline(grain, traffic.height),
                y=al.Y("n:Q", title="спанов в корзине"),
                color=al.Color(
                    "спаны:N",
                    title=None,
                    scale=al.Scale(
                        domain=(Fig.dialog_name, Fig.tech_name),
                        range=(Fig.accent, Fig.context),
                    ),
                ),
                order=al.Order("спаны:N", sort="ascending"),
            )
        )

        return Fig.themed(
            bars,
            "Состав выборки по типу трейса",
            "спаны в трейсах с/без aef_kind=input_request",
        )

    @staticmethod
    def dialog_share(traffic: Frame, grain: str) -> None | al.Chart:
        if traffic.height < Fig.min_buckets:
            return None
        data = traffic.with_columns(
            (
                pl.col("dialog_spans")
                / (pl.col("dialog_spans") + pl.col("tech_spans")).clip(1)
            ).alias("share")
        ).to_pandas()
        share = al.Y(
            "share:Q",
            title="доля диалоговых",
            scale=al.Scale(domain=(0, 1)),
            axis=al.Axis(format=".0%"),
        )
        base = al.Chart(data).encode(x=Fig.timeline(grain, traffic.height))
        line = base.mark_line(
            strokeWidth=2, color=Fig.accent, interpolate="linear", point=True
        ).encode(y=share)
        area = base.mark_area(
            opacity=0.10, color=Fig.accent, interpolate="linear"
        ).encode(y=share)

        return Fig.themed(
            area + line,
            "Доля спанов в трейсах с input_request",
            "расчёт по каждой временной корзине",
        )

    @staticmethod
    def panels(
        flow: Frame, traffic: Frame, grain: str
    ) -> tuple[tuple["al.Chart", str], ...]:
        drawn = (
            (Fig.inflow(flow, grain), "inflow"),
            (Fig.mix(traffic, grain), "traffic_mix"),
            (Fig.dialog_share(traffic, grain), "dialog_share"),
        )

        return tuple(filter(lambda pair: pair[0] is not None, drawn))


@dataclass(frozen=True)
class Grid:
    quantile_sep: ClassVar[str] = "__q"

    @staticmethod
    def quantiles(distribution: Frame) -> Frame:
        row = distribution.to_dicts()[0]
        keyed = tuple(filter(lambda kv: Grid.quantile_sep in kv[0], row.items()))
        names = tuple(
            dict.fromkeys(map(lambda kv: kv[0].split(Grid.quantile_sep)[0], keyed))
        )
        levels = tuple(
            dict.fromkeys(map(lambda kv: kv[0].split(Grid.quantile_sep)[1], keyed))
        )
        record = lambda name: {
            "сигнал": name,
            **dict(
                map(
                    lambda q: (f"q{q}", row.get(f"{name}{Grid.quantile_sep}{q}")),
                    levels,
                )
            ),
        }

        return pl.DataFrame(tuple(map(record, names)))
