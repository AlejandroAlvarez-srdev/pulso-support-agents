from __future__ import annotations

from pulso_support import demo


def test_demo_runs_offline_end_to_end(capsys, monkeypatch):
    monkeypatch.delenv("PULSO_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("PULSO_CHECKPOINT_DB", raising=False)
    demo.main()
    out = capsys.readouterr().out
    assert "needs approval" in out
    assert "has been issued" in out
