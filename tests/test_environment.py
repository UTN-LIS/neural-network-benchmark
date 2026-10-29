"""Tests for the environment capture module."""

from __future__ import annotations

from prebenchmark import environment


def test_capture_environment_has_both_sections():
    env = environment.capture_environment()
    assert set(env) == {"hardware", "library_versions"}


def test_hardware_capture_reports_platform_and_cuda_probe():
    hardware = environment.capture_hardware()
    assert hardware["python_version"]
    assert isinstance(hardware["cuda_available"], bool)
    assert hardware["device_count"] >= 0


def test_library_versions_include_core_stack():
    versions = environment.capture_library_versions()
    assert versions["python"]
    for name in ("torch", "numpy", "pandas", "sklearn"):
        assert versions[name]  # installed in this project
