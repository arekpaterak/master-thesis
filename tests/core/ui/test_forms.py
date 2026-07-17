from typing import Literal

import pytest
from pydantic import BaseModel, Field

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    GreedyAcceptanceConfig,
    SimulatedAnnealingAcceptanceConfig,
    UnconditionalAcceptanceConfig,
)
from core.lns.config import SearchParams
from core.ui.forms import (
    build_mo_ui_dict,
    build_union_selector,
    collect_union_value,
    union_variants,
)


class Demo(BaseModel):
    mode: Literal["a", "b"] = "a"
    flag: bool = True
    steps: int = Field(default=10, ge=1, le=100)
    rate: float = 0.5
    hidden: int = Field(
        default=128, json_schema_extra={"widget": "dropdown", "options": [64, 128, 256]}
    )
    name: str = "x"


class Nested(BaseModel):
    inner: Demo = Demo()
    tag: Literal["only"] = "only"


class TestUnionVariants:
    def test_maps_tag_to_variant(self) -> None:
        variants = union_variants(AcceptanceCriterionConfig)

        assert variants == {
            "greedy": GreedyAcceptanceConfig,
            "sa": SimulatedAnnealingAcceptanceConfig,
            "unconditional": UnconditionalAcceptanceConfig,
        }


class TestConfigForm:
    def test_widget_types_per_field(self) -> None:
        form = build_mo_ui_dict(Demo, Demo())

        assert set(form.value) == {"mode", "flag", "steps", "rate", "hidden", "name"}
        assert list(form["mode"].options) == ["a", "b"]
        assert list(form["hidden"].options.values()) == [64, 128, 256]  # numeric dropdown via hint
        assert form["steps"].start == 1 and form["steps"].stop == 100  # slider from ge/le

    def test_roundtrips_through_model_validate(self) -> None:
        defaults = Demo()
        form = build_mo_ui_dict(Demo, defaults)

        assert Demo.model_validate(form.value) == defaults

    def test_skips_single_value_literal(self) -> None:
        form = build_mo_ui_dict(
            SimulatedAnnealingAcceptanceConfig, SimulatedAnnealingAcceptanceConfig()
        )

        # no "type"
        assert set(form.value) == {"temperature_start", "temperature_min", "augmented_exchange"}
        assert (
            SimulatedAnnealingAcceptanceConfig(**form.value) == SimulatedAnnealingAcceptanceConfig()
        )

    def test_empty_form_for_constant_only_model(self) -> None:
        form = build_mo_ui_dict(GreedyAcceptanceConfig, GreedyAcceptanceConfig())

        assert form.value == {}
        assert GreedyAcceptanceConfig(**form.value) == GreedyAcceptanceConfig()

    def test_nested_model_recurses(self) -> None:
        form = build_mo_ui_dict(Nested, Nested())

        assert set(form.value) == {"inner"}  # "tag" is a single-value literal -> skipped
        assert set(form.value["inner"]) == {"mode", "flag", "steps", "rate", "hidden", "name"}
        assert Nested.model_validate(form.value) == Nested()

    def test_hint_overrides_widget(self) -> None:
        form = build_mo_ui_dict(
            Demo, Demo(), hints={"rate": {"widget": "slider", "start": 0.0, "stop": 1.0}}
        )

        assert form["rate"].start == 0.0 and form["rate"].stop == 1.0

    def test_raises_on_discriminated_union_field(self) -> None:
        with pytest.raises(NotImplementedError):
            build_mo_ui_dict(SearchParams, SearchParams())

    def test_skip_excludes_fields(self) -> None:
        form = build_mo_ui_dict(SearchParams, SearchParams(), skip={"acceptance"})

        assert "acceptance" not in form.value
        assert "num_rollouts" in form.value

    def test_defaults_fall_back_to_model_cls(self) -> None:
        assert build_mo_ui_dict(Demo).value == build_mo_ui_dict(Demo, Demo()).value

    def test_hint_label_overrides_title(self) -> None:
        class Titled(BaseModel):
            mode: Literal["a", "b"] = Field(default="a", title="Mode")

        form = build_mo_ui_dict(Titled, Titled(), hints={"mode": {"label": "Override label"}})
        assert form["mode"]._args.label == "Override label"

    def test_title_used_as_label(self) -> None:
        class Titled(BaseModel):
            mode: Literal["a", "b"] = Field(default="a", title="My Mode")

        form = build_mo_ui_dict(Titled, Titled())
        assert form["mode"]._args.label == "My Mode"


class TestUnionSelector:
    def test_returns_type_dropdown_and_variant_forms(self) -> None:
        type_dropdown, variant_forms = build_union_selector(AcceptanceCriterionConfig)

        assert set(type_dropdown.options) == {"greedy", "sa", "unconditional"}
        assert set(variant_forms) == {"greedy", "sa", "unconditional"}

    def test_seeds_default_from_matching_instance(self) -> None:
        default = SimulatedAnnealingAcceptanceConfig(temperature_start=0.5, temperature_min=0.01)
        type_dropdown, variant_forms = build_union_selector(AcceptanceCriterionConfig, default)

        assert type_dropdown.value == "sa"
        assert variant_forms["sa"].value["temperature_start"] == 0.5

    def test_collect_reconstructs_config(self) -> None:
        default = SimulatedAnnealingAcceptanceConfig()
        type_dropdown, variant_forms = build_union_selector(AcceptanceCriterionConfig, default)

        result = collect_union_value(AcceptanceCriterionConfig, type_dropdown, variant_forms)
        assert result == default

    def test_collect_greedy_variant(self) -> None:
        default = GreedyAcceptanceConfig()
        type_dropdown, variant_forms = build_union_selector(AcceptanceCriterionConfig, default)

        result = collect_union_value(AcceptanceCriterionConfig, type_dropdown, variant_forms)
        assert result == default
