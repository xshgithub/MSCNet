import pytest

import util.slconfig as slconfig
from util.slconfig import ConfigDict, SLConfig



def test_dump_does_not_depend_on_addict_to_dict(monkeypatch):
    def missing_to_dict(self):
        raise AttributeError("to_dict")

    monkeypatch.setattr(ConfigDict, "to_dict", missing_to_dict, raising=False)

    cfg = SLConfig({"model": {"name": "demo"}, "epochs": 1})

    dumped = cfg.dump()

    assert "epochs = 1" in dumped
    assert "model = dict(" in dumped


def test_dump_falls_back_when_yapf_raises_attribute_error(monkeypatch):
    def broken_format_code(*args, **kwargs):
        raise AttributeError("formatter failed")

    monkeypatch.setattr(slconfig, "FormatCode", broken_format_code)

    cfg = SLConfig({"epochs": 1})

    assert "epochs" in cfg.dump()


def test_dwt_lite_backbone_api_is_registered():
    dwt_source = (ROOT / "models" / "dqdetr" / "dwt_new.py").read_text(encoding="utf-8")
    backbone_source = (ROOT / "models" / "dqdetr" / "backbone.py").read_text(encoding="utf-8")

    assert "class DWTLite" in dwt_source
    assert "class DSFusionLite" in dwt_source
    assert "class MultiBackboneLite" in dwt_source
    assert "def build_dwt_backbone_lite" in dwt_source
    assert "dwt_resnet50_lite" in backbone_source
    assert "build_dwt_backbone_lite" in backbone_source


def test_dwt_lite_uses_bottleneck_and_depthwise_fusion():
    dwt_source = (ROOT / "models" / "dqdetr" / "dwt_new.py").read_text(encoding="utf-8")

    assert "bottleneck_channels" in dwt_source
    assert "groups=hidden_channels" in dwt_source
    assert "groups=self.channels" in dwt_source