import importlib
import pytest

PAGE_MODULES = [
    "ui.pages.overview",
    "ui.pages.scanner",
    "ui.pages.manual",
    "ui.pages.performance",
    "ui.pages.explainability",
    "ui.pages.dataset",
    "ui.pages.about",
]


@pytest.mark.parametrize("module_name", PAGE_MODULES)
def test_page_module_imports(module_name: str):
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "render")), f"{module_name} must define render(bundle)"


def test_app_wiring():
    app = importlib.import_module("app")
    assert set(app.PAGES) == {
        "Dashboard",
        "Threat Scanner",
        "Manual Prediction",
        "Model Performance",
        "Explainability",
        "Dataset Insights",
        "About",
    }
    for page in app.PAGES.values():
        assert callable(getattr(page, "render"))


def test_stylesheet_loads():
    from ui import css

    assert css.CSS.lstrip().startswith("<style>")
    assert css.CSS.rstrip().endswith("</style>")
    assert "--sev-critical" in css.CSS  


def test_manual_presets_reference_valid_form_keys():
    from ui.pages.manual import PRESETS

    expected = {
        "destination_port",
        "protocol",
        "connection_duration",
        "total_packets",
        "average_packet_size",
        "max_packet_size",
        "flow_bytes_per_second",
        "flow_packets_per_second",
    }
    for name, values in PRESETS.items():
        assert set(values) == expected, f"preset '{name}' has mismatched keys"
