"""CPU regression checks for the recipe and causal objective, no downloads."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train


def tiny_config():
    from transformers import GPTNeoXConfig
    config = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=128,
                          max_position_embeddings=32, rotary_pct=0.25,
                          tie_word_embeddings=False, attention_dropout=0.0,
                          hidden_dropout=0.0, use_cache=False)
    config._attn_implementation = "sdpa"
    return config


def test_scratch_constructor_and_paper_initialization(monkeypatch):
    from transformers import GPTNeoXForCausalLM
    monkeypatch.setattr(GPTNeoXForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("pretrained weights forbidden"))
    model = train.build_model(1234, tiny_config())
    same = train.build_model(1234, tiny_config())
    assert model.embed_out.weight.data_ptr() != model.gpt_neox.embed_in.weight.data_ptr()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, dict(same.named_parameters())[name]), name
        assert parameter.dtype == torch.float32
        if name.endswith("bias") or name.endswith("lrcp.up"):
            assert torch.count_nonzero(parameter) == 0
        if "layernorm.weight" in name or "layer_norm.weight" in name:
            assert torch.equal(parameter, torch.ones_like(parameter))
    for layer in model.gpt_neox.layers:
        assert layer.attention.lrcp.source == "value"
        assert not layer.attention.lrcp.use_fused_training
        assert not layer.attention.lrcp.use_fused_inference
    stats = train.paper_initialize(model)
    assert stats["small_std"] == pytest.approx(np.sqrt(2/(5*32)))
    assert stats["wang_std"] == pytest.approx(2/2/np.sqrt(32))


def test_optimizer_groups_partition_and_branch_decay():
    model = train.build_model(1234, tiny_config())
    groups = train.optimizer_groups(model)
    by_id = {id(p): group for group in groups for p in group["params"]}
    assert len(by_id) == sum(len(group["params"]) for group in groups)
    assert set(by_id) == {id(p) for p in model.parameters()}
    for name, parameter in model.named_parameters():
        group = by_id[id(parameter)]
        if ".lrcp." in name:
            assert group["weight_decay"] == 0
            assert group["lr"] == 1e-4
            assert group["lr_scale"] == .1
        elif name.endswith("bias") or "layernorm" in name or "layer_norm" in name:
            assert group["weight_decay"] == 0
        else:
            assert group["weight_decay"] == .1
            assert group["lr"] == .001


def test_schedule_uses_original_300b_horizon():
    assert train.scheduled_lr(0) == 0
    assert train.scheduled_lr(1) == pytest.approx(.001/1430)
    assert train.scheduled_lr(1430) == .001
    assert train.scheduled_lr(143000) == pytest.approx(.0001)
    assert train.scheduled_lr(200000) == pytest.approx(.0001)
    for step in (1431, 20000, 47684, 100000, 143000):
        effective = min(step, 143000 - 1430) - 1430
        reference = max(.001/2 * (np.cos(np.pi * effective/143000) + 1), .0001)
        assert train.scheduled_lr(step) == pytest.approx(reference, rel=1e-15)
    assert train.scheduled_lr(train.DEFAULT_STEPS) > .0007
    assert train.DEFAULT_STEPS * train.GLOBAL_BATCH * train.SEQUENCE == 100_000_595_968


def test_loss_predicts_every_next_token_including_last():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    tokens = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
    class Model:
        def __call__(self, *, input_ids, use_cache):
            assert not use_cache
            assert torch.equal(input_ids, tokens[:, :-1])
            return SimpleNamespace(logits=logits)
    loss = train.loss_for_batch(Model(), tokens)
    ref = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(loss, ref, rtol=0, atol=0)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, -1])


def test_stop_file_does_not_touch_cuda(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    stop.touch()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("STOP must not synchronize"))
    with pytest.raises(train.StopRequested):
        train.Stopper(stop).check()


def test_synthetic_never_implicit_train():
    args = train.parser().parse_args([])
    with pytest.raises(ValueError, match="--data-manifest"):
        train.run(args)
    args = train.parser().parse_args(["--data-manifest", "actual.json"])
    with pytest.raises(ValueError, match="--validation-manifest"):
        train.run(args)


def test_microbatch_must_preserve_global_batch():
    args = train.parser().parse_args(["--microbatch", "3", "--smoke"])
    with pytest.raises(ValueError, match="divide global batch"):
        train.run(args)
