"""Человекочитаемая репрезентация правил валидации спецификации.

Описания строятся обходом дерева самого выражения валидации — правила не
дублируются текстом, поэтому при обновлении спецификации описания
актуализируются автоматически (принцип единого источника истины).
"""

from typing import Any, ClassVar, Iterator
from dataclasses import dataclass
from io import BytesIO

import json

import polars as pl

from common_core import Patterns
from core import SpanAttr, SpansDataSpec


class Unrenderable(Exception):
    """Выражение содержит узел, для которого нет человекочитаемой формы."""


@dataclass(frozen=True)
class Wording:
    patterns: ClassVar[dict[str, str]] = {
        Patterns.base64: "строка в формате base64 (латинские буквы, цифры, «+», «/», до двух «=» в конце)",
        Patterns.session_id: "непустая строка",
        Patterns.codebase_elem: "код агента: «CI» или «CE» и цифры (например, CI01234567)",
        Patterns.nexus_version: "версия вида «число.число.число» (неизвестная часть — «?»)",
        Patterns.json_str_array: 'JSON-массив строк (например, ["a", "b"])',
        Patterns.json_like: "JSON-объект или JSON-массив",
    }

    compare: ClassVar[dict[str, str]] = {
        "Gt": "больше",
        "GtEq": "не меньше",
        "Lt": "меньше",
        "LtEq": "не больше",
    }

    @staticmethod
    def value(raw: Any) -> str:
        match raw:
            case True:
                return "истина"
            case False:
                return "ложь"
            case "":
                return "пустая строка"
            case float():
                return f"{raw:g}"
            case _:
                return str(raw)


@dataclass(frozen=True, eq=False)
class Node:
    @staticmethod
    def literal(node: dict) -> Any:
        scalar = node.get("Scalar", node)
        match scalar:
            case {"String": text}:
                return text
            case {"Boolean": flag}:
                return flag
            case {"Int": num} | {"Dyn": {"Int": num}}:
                return num
            case {"Float": num} | {"Dyn": {"Float": num}}:
                return num
            case {"List": blob}:
                return tuple(
                    pl.read_ipc_stream(BytesIO(bytes(blob))).to_series().to_list()
                )
            case _:
                raise Unrenderable(f"литерал: {scalar}")

    @staticmethod
    def render(node: dict) -> str:
        match node:
            case {"Column": name}:
                return str(name)

            case {"Literal": lit}:
                value = Node.literal(lit)
                if value is True:
                    return "без ограничений (любое значение)"

                return Wording.value(value)

            case {"BinaryExpr": {"left": left, "op": op, "right": right}}:
                return Node.binary(left, op, right)

            case {"Ternary": {"predicate": when, "truthy": then, "falsy": other}}:
                return f"если {Node.render(when)}, то {Node.render(then)}; иначе {Node.render(other)}"

            case {"Function": {"input": args, "function": fn}}:
                return Node.function(args, fn)

            case _:
                raise Unrenderable(f"узел: {tuple(node.keys())}")

    @staticmethod
    def binary(left: dict, op: str, right: dict) -> str:
        match op:
            case "And":
                return f"{Node.render(left)} и {Node.render(right)}"
            case "Or":
                return f"{Node.render(left)} или {Node.render(right)}"

        if op == "Eq":
            if "Literal" not in right:
                return f"{Node.render(left)} = {Node.render(right)}"
            value = Node.literal(right["Literal"])
            if value == "":
                return f"{Node.render(left)} — пустая строка"
            if isinstance(value, str):
                return f"{Node.render(left)} = «{value}»"

            return f"{Node.render(left)} = {Wording.value(value)}"

        if op in Wording.compare:
            worded = Node.length_guard(left, op, right)

            return (
                worded
                if worded
                else f"{Node.render(left)} {Wording.compare[op]} {Node.render(right)}"
            )

        raise Unrenderable(f"операция: {op}")

    @staticmethod
    def length_guard(left: dict, op: str, right: dict) -> None | str:
        match left:
            case {
                "Function": {"input": [inner], "function": {"StringExpr": "LenChars"}}
            }:
                value = Node.literal(right["Literal"]) if "Literal" in right else None

                return (
                    f"{Node.render(inner)} — непустая строка"
                    if (op, value) == ("Gt", 0)
                    else f"длина {Node.render(inner)} {Wording.compare[op]} {Wording.value(value)}"
                )
            case _:
                return None

    @staticmethod
    def function(args: list, fn: Any) -> str:
        subject = Node.render(args[0])
        match fn:
            case {"Boolean": "Not"}:
                match args[0]:
                    case {
                        "BinaryExpr": {
                            "left": {"Column": a},
                            "op": "And",
                            "right": {"Column": b},
                        }
                    }:
                        return f"{a} и {b} не равны истине одновременно"
                    case _:
                        return f"не выполняется «{subject}»"

            case {"Boolean": {"IsIn": _}}:
                allowed = Node.literal(args[1]["Literal"])

                return f"{subject} — одно из: {', '.join(map(Wording.value, allowed))}"

            case {"Boolean": {"IsBetween": _}}:
                low, high = (
                    Node.literal(args[1]["Literal"]),
                    Node.literal(args[2]["Literal"]),
                )

                return f"{subject} в диапазоне от {Wording.value(low)} до {Wording.value(high)}"

            case {"StringExpr": {"Contains": _}}:
                pattern = Node.literal(args[1]["Literal"])
                known = Wording.patterns.get(pattern)

                return (
                    f"{subject} — {known}"
                    if known
                    else f"{subject} соответствует шаблону {pattern}"
                )

            case {"StringExpr": "LenChars"}:
                return f"длина {subject}"

            case _:
                raise Unrenderable(f"функция: {fn}")


def describe(attr: SpanAttr) -> str:
    try:
        return Node.render(json.loads(attr.validation.meta.serialize(format="json")))
    except Exception:
        return "см. правило атрибута в спецификации телеметрии"


def empty_marker(attr: SpanAttr) -> str:
    return "—" if attr.empty is None else f"«{Wording.value(attr.empty)}»"


def rules(spec: SpansDataSpec) -> Iterator[tuple[str, bool, str, str]]:
    return map(
        lambda a: (a.name, a.mandatory, describe(a), empty_marker(a)), spec.attrs
    )
