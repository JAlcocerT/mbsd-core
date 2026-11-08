"""The 1.0 public contract freeze: the classified surface matches the code.

Any new, removed or renamed public symbol fails here until ``mbsd.stability``
classifies it, so every surface change during the candidate period is an
explicit, reviewed decision.
"""

from __future__ import annotations

import importlib
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import mbsd
from mbsd import Mechanism, stability
from mbsd.planar import PlanarMechanism
from mbsd.spatial import SpatialMechanism

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
INSTANCE_ATTRIBUTES = {"model", "loads", "springs"}


def _names(namespace: str, *statuses: str) -> set[str]:
    return {name for name, entry in stability.SURFACE[namespace].items() if entry["status"] in statuses}


@pytest.mark.parametrize(
    "namespace, statuses",
    [
        ("mbsd", ("stable",)),
        ("mbsd.planar", ("stable",)),
        ("mbsd.planar.analysis", ("stable",)),
        ("mbsd.spatial", ("stable",)),
        ("mbsd.experimental", ("experimental",)),
    ],
)
def test_all_lists_match_the_classified_surface(namespace, statuses):
    module = importlib.import_module(namespace)
    assert set(module.__all__) == _names(namespace, *statuses)
    for name in module.__all__:
        getattr(module, name)


@pytest.mark.parametrize("namespace", ["mbsd.errors", "mbsd.reports"])
def test_modules_without_all_expose_their_classified_names(namespace):
    module = importlib.import_module(namespace)
    for name in stability.SURFACE[namespace]:
        assert inspect.isclass(getattr(module, name))


@pytest.mark.parametrize(
    "cls, namespace",
    [(PlanarMechanism, "mbsd.planar.PlanarMechanism"), (SpatialMechanism, "mbsd.spatial.SpatialMechanism")],
)
def test_every_public_builder_method_is_classified(cls, namespace):
    public = {name for name in dir(cls) if not name.startswith("_")}
    classified = set(stability.SURFACE[namespace])
    assert public - classified == set(), "unclassified public methods"
    assert classified - public <= INSTANCE_ATTRIBUTES, "classified methods that no longer exist"


def test_classifications_are_valid_and_deprecations_name_a_replacement():
    for entries in stability.SURFACE.values():
        for entry in entries.values():
            assert entry["status"] in stability.STATUSES
            if entry["status"] == "deprecated":
                assert entry["removal"] == "1.0.0" and entry["replacement"]
    assert stability.status("mbsd.planar", "solver") == "internal"
    assert stability.status("mbsd.spatial", "SpatialForce") == "internal"
    assert stability.status("mbsd.experimental", "SpatialForce") == "experimental"


def test_experimental_names_moved_with_a_warning_alias():
    import mbsd.experimental as experimental
    import mbsd.spatial as spatial

    for name in experimental.__all__:
        assert name not in spatial.__all__
        with pytest.warns(DeprecationWarning, match="moved to mbsd.experimental"):
            assert getattr(spatial, name) is getattr(experimental, name)


def test_deprecated_names_warn_and_still_work():
    import mbsd.spatial as spatial

    for name in stability.DEPRECATED_SPATIAL:
        assert name not in spatial.__all__
        with pytest.warns(DeprecationWarning, match="removed in 1.0"):
            getattr(spatial, name)
    with pytest.raises(AttributeError):
        spatial.not_a_symbol  # noqa: B018


def test_deprecated_builder_methods_warn():
    m = Mechanism.spatial()
    m.body("rotor", mass=1.0, inertia=(0.1, 0.2, 0.3))
    with pytest.warns(DeprecationWarning):
        result = m.simulate_free_body(np.linspace(0.0, 0.1, 3), angular_velocity0_body=(0.0, 0.0, 1.0))
    for method in ("rigid_body_diagnostics", "rigid_body_result_to_dict"):
        with pytest.warns(DeprecationWarning, match=method):
            getattr(m, method)(result)


def test_stable_surface_imports_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for namespace in ("mbsd", "mbsd.planar", "mbsd.planar.analysis", "mbsd.spatial", "mbsd.experimental"):
            module = importlib.import_module(namespace)
            for name in module.__all__:
                getattr(module, name)


def test_api_surface_file_matches_runtime():
    on_disk = json.loads((SCHEMAS / "mbsd.api-surface-v1.json").read_text())
    assert on_disk == json.loads(json.dumps(mbsd.api_surface()))
    assert on_disk["schema"] == "mbsd.api_surface"
