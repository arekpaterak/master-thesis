from __future__ import annotations

import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import marimo as mo
from pydantic import BaseModel
from pydantic.fields import FieldInfo

_UNION = (Union, types.UnionType)


def union_variants(union_type: Any) -> dict[str, type[BaseModel]]:
    """Map each discriminator tag to its variant model for a discriminated union.

    Works on an alias such as ``Annotated[A | B, Field(discriminator="type")]`` where
    each variant carries the discriminator as a single-value ``Literal`` field
    (e.g. ``type: Literal["a"]``).
    """
    discriminator = _discriminator_name(union_type)
    variants: dict[str, type[BaseModel]] = {}
    for member in get_args(_strip_annotated(union_type)):
        model = _strip_annotated(member)
        (tag,) = get_args(model.model_fields[discriminator].annotation)
        variants[str(tag)] = model
    return variants


def build_mo_ui_dict(
    model_cls: type[BaseModel],
    defaults: BaseModel | None = None,
    *,
    hints: dict[str, dict[str, Any]] | None = None,
    skip: set[str] | None = None,
) -> mo.ui.dictionary:
    """Build a ``mo.ui.dictionary`` of widgets for Pydantic ``model_cls``'s fields.

    ``defaults`` seeds the initial widget values; pass an existing config instance to
    pre-fill from a loaded preset/run. When omitted it falls back to ``model_cls()``
    (which requires every field to have a default).

    Single-value ``Literal`` fields (discriminator tags / constants) are skipped,
    nested models recurse, and discriminated-union fields raise (render them with
    :func:`union_variants` in a dependent cell).
    Pass ``skip`` to exclude fields you render manually (e.g. a union field).

    Collect ``.value`` and pass to ``model_cls.model_validate`` to rebuild the config.

    The returned dictionary's own rendering is a generic key/value block; for a normal
    stacked layout display its children instead, e.g. ``mo.vstack(list(form.values()))``.
    """
    hints = hints or {}
    skip = skip or set()
    if defaults is None:
        defaults = model_cls()
    elements: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        if name in skip:
            continue
        if _is_discriminated_union(field.annotation):
            raise NotImplementedError(
                f"{model_cls.__name__}.{name} is a discriminated union; render it with "
                "union_variants() in a dependent cell."
            )
        spec = {**_json_schema_extra(field), **hints.get(name, {})}
        if _is_constant(field, spec):
            continue
        elements[name] = _widget_for(name, field, getattr(defaults, name), spec)
    return mo.ui.dictionary(elements)


def _widget_for(name: str, field: FieldInfo, current: Any, spec: dict[str, Any]) -> Any:
    label = field.description or name
    widget = spec.get("widget")
    annotation = _strip_optional(field.annotation)
    options = spec.get("options") or _literal_options(field.annotation)

    if widget == "dropdown" or (widget is None and options is not None):
        return mo.ui.dropdown(options=list(options), value=current, label=label)
    if widget == "text" or annotation is str:
        return mo.ui.text(value=current or "", label=label)
    if annotation is bool:
        return mo.ui.checkbox(value=bool(current), label=label)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build_mo_ui_dict(annotation, current)
    if annotation in (int, float):
        low, high = _bounds(field, spec)
        step = spec.get("step")
        if step is None and annotation is int:
            step = 1
        if widget == "slider" or (widget is None and low is not None and high is not None):
            return mo.ui.slider(start=low, stop=high, value=current, step=step, label=label)
        return mo.ui.number(value=current, step=step, label=label)
    raise NotImplementedError(f"No widget mapping for {name!r} of type {field.annotation!r}")


def _bounds(field: FieldInfo, spec: dict[str, Any]) -> tuple[Any, Any]:
    """Lower/upper bound for a numeric field, from ``spec`` or ``Field(ge/gt/le/lt)``.

    ``Field(ge=..., le=...)`` lowers to constraint objects in ``field.metadata`` that
    expose the bound under the attribute names below. Reading them by name avoids a
    direct import of pydantic's (transitively-installed) constraint library.
    """
    low = spec.get("start")
    high = spec.get("stop")
    for meta in field.metadata:
        low = low if low is not None else getattr(meta, "ge", getattr(meta, "gt", None))
        high = high if high is not None else getattr(meta, "le", getattr(meta, "lt", None))
    return low, high


def _is_constant(field: FieldInfo, spec: dict[str, Any]) -> bool:
    """A field with a single allowed value (e.g. a discriminator tag) has no widget."""
    if spec.get("widget"):
        return False
    options = _literal_options(field.annotation)
    return options is not None and len(options) == 1


def _is_discriminated_union(annotation: Any) -> bool:
    base = annotation
    if get_origin(annotation) is Annotated:
        base = get_args(annotation)[0]
    if get_origin(base) not in _UNION:
        return False
    members = [_strip_annotated(m) for m in get_args(base) if m is not type(None)]
    return len(members) > 1 and all(
        isinstance(m, type) and issubclass(m, BaseModel) for m in members
    )


def _literal_options(annotation: Any) -> list[Any] | None:
    base = _strip_optional(annotation)
    if get_origin(base) is Literal:
        return list(get_args(base))
    return None


def _strip_optional(annotation: Any) -> Any:
    if get_origin(annotation) in _UNION:
        members = [m for m in get_args(annotation) if m is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _strip_annotated(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _discriminator_name(union_type: Any) -> str:
    if get_origin(union_type) is Annotated:
        for meta in get_args(union_type)[1:]:
            name = getattr(meta, "discriminator", None)
            if name:
                return name
    return "type"


def _json_schema_extra(field: FieldInfo) -> dict[str, Any]:
    extra = field.json_schema_extra
    return dict(extra) if isinstance(extra, dict) else {}
