# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import importlib
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch._dynamo
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig, TorchtitanEngineConfig
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    TPUSplashAttention,
    TPUVarlenAttention,
    apply_activation_in_fp32_tpu,
    apply_splash_attention_tpu,
    bucket_length,
    configure_torch_compile_for_tpu,
    extend_torchtitan_engine_config,
    force_sum_reduction_on_all_fsdp_modules,
    get_tpu_seq_bucket_size,
    pad_packed_inputs_for_tpu,
    prepare_tpu_loss_inputs,
    replace_varlen_attention_with_tpu_attention,
    resolve_tpu_activation_in_fp32,
    resolve_tpu_torchtitan_options,
    splash_block_size_for,
    tpu_rollout_activation_env_vars,
    unwrap_metadata,
)
from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    PlatformTPU,
    patch_ray_worker,
    resolve_tpu_topology_bounds,
)


def test_bucket_length_and_env_override():
    assert get_tpu_seq_bucket_size() == 256
    assert bucket_length(1) == 256
    assert bucket_length(256) == 256
    assert bucket_length(257) == 512
    assert bucket_length(70, bucket_size=64) == 128

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "64"}):
        assert get_tpu_seq_bucket_size() == 64
        assert bucket_length(65) == 128


def test_unwrap_metadata():
    assert unwrap_metadata([torch.tensor(3.5)]) == 3.5
    assert unwrap_metadata((True, False)) is True
    assert unwrap_metadata("flex") == "flex"


def test_pad_packed_inputs_for_tpu_builds_4d_document_causal_mask():
    # Two packed documents of lengths 3 and 2 -> orig_seq_len = 5
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        _, _, _, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
        )

    assert orig_seq_len == 5
    assert attention_masks is not None
    assert attention_masks.shape == (1, 1, 8, 8)
    assert attention_masks.dtype == torch.bool
    # Document 0 (tokens 0..2) attends causally within [0..2] and not to document 1 (tokens 3..4)
    assert attention_masks[0, 0, 2, 0].item() is True
    assert attention_masks[0, 0, 0, 2].item() is False
    assert attention_masks[0, 0, 3, 2].item() is False
    assert attention_masks[0, 0, 4, 3].item() is True
    # Padded tail (tokens 5..7) has self-attention only (no cross-token attention)
    assert attention_masks[0, 0, 6, 6].item() is True
    assert attention_masks[0, 0, 6, 5].item() is False


def test_pad_packed_inputs_for_tpu_skips_mask_and_honors_aligned_length():
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_seq_len", 16)

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        padded_ids, padded_pos, padded_labels, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
            build_attention_mask=False,
        )

    assert orig_seq_len == 5
    assert attention_masks is None
    assert padded_ids.shape == (1, 16)
    assert padded_pos.shape == (1, 16)
    assert padded_labels.shape == (1, 16)


def test_splash_block_size_for():
    assert splash_block_size_for(512) == 512
    assert splash_block_size_for(1024) == 512
    assert splash_block_size_for(768) == 256
    assert splash_block_size_for(256) == 256
    assert splash_block_size_for(384) == 128
    assert splash_block_size_for(64) == 0


def test_apply_splash_attention_tpu_and_cpu_fallback():
    class _DummyInnerAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = (256, 0)

        def forward(self, q, k, v, **kwargs):
            return q + k + v

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = _DummyInnerAttn()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert apply_splash_attention_tpu([model]) == 2
    assert isinstance(model[0].inner_attention, TPUSplashAttention)
    assert model[0].inner_attention.local_window_size == 256
    # Idempotent on repeated calls (e.g. __init__ followed by initialize)
    assert apply_splash_attention_tpu([model]) == 0

    q = torch.ones(1, 256, 2, 4)
    out = model[0].inner_attention(q, q, q)
    assert torch.allclose(out, q * 3)


def test_configure_torch_compile_for_tpu():
    configure_torch_compile_for_tpu(recompile_limit=64)
    assert torch._dynamo.config.automatic_dynamic_shapes is False
    assert torch._dynamo.config.assume_static_by_default is True
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert torch._dynamo.config.recompile_limit >= 64


def test_resolve_tpu_torchtitan_options_and_config_extension():
    extend_torchtitan_engine_config()
    cfg_compiled = TorchtitanEngineConfig(use_torch_compile=True, attn_type="varlen")
    assert resolve_tpu_torchtitan_options(cfg_compiled) == (True, True, "DEFER_AND_FUSE")

    cfg_eager = TorchtitanEngineConfig(use_torch_compile=False, attn_type="varlen")
    assert resolve_tpu_torchtitan_options(cfg_eager) == (False, False, None)

    cfg_invalid = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=False,
        use_simple_fsdp=None,
        tpu_eager_mode=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_splash_attention=True"):
        resolve_tpu_torchtitan_options(cfg_invalid)

    cfg_tp = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=True,
        use_simple_fsdp=True,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_simple_fsdp supports pure FSDP/HSDP only"):
        resolve_tpu_torchtitan_options(cfg_tp)


def test_replace_varlen_attention_with_tpu_attention():
    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = torch.nn.Identity()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert replace_varlen_attention_with_tpu_attention([model]) == 2
    assert isinstance(model[0].inner_attention, TPUVarlenAttention)
    assert isinstance(model[1].inner_attention, TPUVarlenAttention)
    # Idempotent when called a second time
    assert replace_varlen_attention_with_tpu_attention([model]) == 0


def test_resolve_tpu_topology_bounds_multi_host():
    # v6e-8 across two hosts: the mesh spans hosts, so the bounds are the host bounds.
    topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
        total_chips=8, num_nodes=2
    )
    assert (topology, host_bounds, chips_per_host_bounds, chips_per_host) == ("2,4,1", "2,4,1", "1,1,1", "4")

    # v6e-16 used to fall through to "1,1,1", which silently trains on one chip.
    assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=4)[:2] == ("4,4,1", "4,4,1")


def test_resolve_tpu_topology_bounds_single_host():
    # A 4-chip slice on one host is addressed inside the host, not across hosts.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=1) == ("2,2,1", "1,1,1", "2,2,1", "4")
    # Same chip count spread over two hosts cannot use in-host bounds.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=2) == ("2,2,1", "1,1,1", "1,1,1", "2")


def test_resolve_tpu_topology_bounds_pod_type_and_env_override():
    # Pod type wins over the chip count: this job holds 8 of a 16-chip slice.
    assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2, pod_type="v6e-16")[0] == "4,4,1"

    # The env var wins over everything, including an unknown chip count.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,3,1"}):
        assert resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)[0] == "2,3,1"

    with mock.patch.dict(os.environ, {"VERL_TPU_CHIPS_PER_HOST": "2"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2)[3] == "2"


def test_resolve_tpu_topology_bounds_raises_on_unknown_slice():
    # Guessing "1,1,1" here would train on a subset of the slice without any error.
    with pytest.raises(ValueError, match="TORCH_TPU_TOPOLOGY"):
        resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)


def _two_slice_nodes():
    return [
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
        {"Alive": False, "Resources": {"TPU": 4.0, "tpu-group-9": 1.0}},
    ]


def test_auto_assign_accelerator_type_splits_trainer_and_rollout_slices():
    platform = PlatformTPU()
    with (
        mock.patch("ray.is_initialized", return_value=True),
        mock.patch("ray.nodes", return_value=_two_slice_nodes()),
    ):
        assert platform.auto_assign_accelerator_type("global_pool", None) == "tpu-group-0"
        assert platform.auto_assign_accelerator_type("rollout_pool_0", None) == "tpu-group-1"
        assert platform.auto_assign_accelerator_type("rollout_pool_reward_0", None) == "tpu-group-1"
        assert platform.auto_assign_accelerator_type("rollout_pool_0", "tpu-group-7") == "tpu-group-7"


def test_auto_assign_accelerator_type_single_slice_shares_slice():
    platform = PlatformTPU()
    nodes = [{"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}}]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        assert platform.auto_assign_accelerator_type("rollout_pool_0", None) == "tpu-group-0"


def test_get_ray_init_kwargs_names_the_setup_hook_by_module_path():
    hook = PlatformTPU().get_ray_init_kwargs()["runtime_env"]["worker_process_setup_hook"]
    assert hook == "verl_hardware_plugin.accelerators.tpu.platform_tpu.patch_ray_worker"
    module_name, _, func_name = hook.rpartition(".")
    assert getattr(importlib.import_module(module_name), func_name) is patch_ray_worker


def test_get_tpu_seq_bucket_size_respects_pad_to_length_config():
    cfg_enabled = SimpleNamespace(pad_to_length=True, pad_to_length_bucket=512)
    cfg_disabled = SimpleNamespace(pad_to_length=False, pad_to_length_bucket=512)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VERL_TPU_SEQ_BUCKET_SIZE", None)
        assert get_tpu_seq_bucket_size(cfg_enabled) == 512
        assert get_tpu_seq_bucket_size(cfg_disabled) == 256

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "128"}):
        assert get_tpu_seq_bucket_size(cfg_enabled) == 128


def test_force_sum_reduction_on_all_fsdp_modules():
    class _FakeFSDPLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.set_force_sum_reduction_for_comms = mock.MagicMock()

    class _FakeFSDPRoot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = _FakeFSDPLayer()
            self.layer1 = _FakeFSDPLayer()
            self.plain = torch.nn.Linear(2, 2)
            self.set_force_sum_reduction_for_comms = mock.MagicMock()

    assert force_sum_reduction_on_all_fsdp_modules(None) == 0
    assert force_sum_reduction_on_all_fsdp_modules([torch.nn.Linear(2, 2)]) == 0

    root = _FakeFSDPRoot()
    assert force_sum_reduction_on_all_fsdp_modules([root]) == 3
    root.set_force_sum_reduction_for_comms.assert_called_once_with(True)
    root.layer0.set_force_sum_reduction_for_comms.assert_called_once_with(True)
    root.layer1.set_force_sum_reduction_for_comms.assert_called_once_with(True)


def test_prepare_tpu_loss_inputs_matches_unmodified_sft_loss():
    # 2 packed samples of lengths 3 and 2 (orig_seq_len = 5), padded to bucket 8
    padded_log_probs = torch.tensor(
        [-1.0, -2.0, -3.0, -4.0, -5.0, -100.0, -200.0, -300.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    unpadded_log_probs = padded_log_probs.detach().clone()[:5]
    nested_log_probs = torch.nested.nested_tensor_from_jagged(unpadded_log_probs, cu_seqlens)
    setattr(nested_log_probs, TPU_PADDED_VALUES_ATTR, padded_log_probs)

    loss_mask = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 1]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({"loss_mask": loss_mask}, batch_size=[2])
    tu.assign_non_tensor(
        micro_batch,
        pad_mode=DatasetPadMode.NO_PADDING,
        dp_size=2,
        batch_num_tokens=6,
    )

    model_output = {"log_probs": nested_log_probs}
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = sft_loss(config=None, model_output=loss_model_output, data=loss_data)
    assert metrics == {}
    loss.backward()

    # Rolled unpadded mask is [1, 1, 0, 1, 0], padded with 3 zeros -> [1, 1, 0, 1, 0, 0, 0, 0].
    # Selected log_probs are -1.0, -2.0, -4.0 -> sum = -7.0 -> loss = -(-7.0) / 6 * 2 = 7/3.
    assert loss.item() == pytest.approx(7.0 / 3.0)
    assert padded_log_probs.grad is not None
    expected_grad = torch.tensor([-1.0 / 3.0, -1.0 / 3.0, 0.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(padded_log_probs.grad, expected_grad)


def _ppo_micro_batch(lengths, max_response_len=None, seed=0):
    """Packed PPO micro-batch as `TorchTitanTPUEngineWithLMHead` builds it: nested prompts/responses on CPU."""
    gen = torch.Generator().manual_seed(seed)

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    prompt_lens = [p for p, _ in lengths]
    response_lens = [r for _, r in lengths]
    micro_batch = TensorDict(
        {
            "prompts": nested([torch.randint(0, 100, (p,), generator=gen) for p in prompt_lens]),
            "responses": nested([torch.randint(0, 100, (r,), generator=gen) for r in response_lens]),
            "input_ids": nested([torch.randint(0, 100, (p + r,), generator=gen) for p, r in lengths]),
            "response_mask": nested([torch.ones(r, dtype=torch.int64) for r in response_lens]),
            "old_log_probs": nested([-torch.rand(r, generator=gen) for r in response_lens]),
            "advantages": nested([torch.randn(r, generator=gen) for r in response_lens]),
            "ref_log_prob": nested([-torch.rand(r, generator=gen) for r in response_lens]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(micro_batch, dp_size=2, batch_num_tokens=sum(response_lens) * 2, global_batch_size=8)
    if max_response_len is not None:
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_response_len)
    return micro_batch


def _ppo_model_output(lengths, padded_len, with_entropy, seed=0):
    """Bucket-padded device outputs + their trimmed CPU views, as `prepare_model_outputs` returns them."""
    gen = torch.Generator().manual_seed(seed + 1)
    total = sum(p + r for p, r in lengths)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    leaves = {"log_probs": -torch.rand(padded_len, generator=gen)}
    if with_entropy:
        leaves["entropy"] = torch.rand(padded_len, generator=gen)
    model_output = {}
    for key, leaf in leaves.items():
        leaf.requires_grad_(True)
        nested = torch.nested.nested_tensor_from_jagged(leaf.detach()[:total], cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, leaf)
        model_output[key] = nested
    return model_output, leaves, cu_seqlens


def _ppo_config():
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
    )


@pytest.mark.parametrize("lengths", [[(5, 7)], [(3, 2), (6, 9), (4, 1)]])
@pytest.mark.parametrize("with_entropy", [False, True])
def test_prepare_tpu_loss_inputs_ppo_matches_unmodified_ppo_loss(lengths, with_entropy):
    padded_len = bucket_length(sum(p + r for p, r in lengths), 16)
    response_bucket = bucket_length(max(r for _, r in lengths), 16)

    # Reference: core ppo_loss on exact (unbucketed) packed outputs, as on GPU.
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len, with_entropy)
    total = int(cu_seqlens[-1])
    reference_output = {
        key: torch.nested.nested_tensor_from_jagged(leaf[:total], cu_seqlens) for key, leaf in leaves.items()
    }
    ref_loss, ref_metrics = ppo_loss(_ppo_config(), reference_output, _ppo_micro_batch(lengths))
    ref_loss.backward()
    ref_grads = {key: leaf.grad.clone() for key, leaf in leaves.items()}
    for leaf in leaves.values():
        leaf.grad = None

    # TPU adapter: bucket-padded outputs, static layout, same unmodified core loss.
    micro_batch = _ppo_micro_batch(lengths, max_response_len=response_bucket)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = ppo_loss(_ppo_config(), loss_model_output, loss_data)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-6)
    for key, leaf in leaves.items():
        torch.testing.assert_close(leaf.grad, ref_grads[key], rtol=0, atol=1e-7)
    assert set(metrics) == set(ref_metrics)
    # The source micro-batch keeps its real sequence metadata.
    assert micro_batch["responses"].offsets().diff().tolist() == [r for _, r in lengths]


def test_prepare_tpu_loss_inputs_ppo_shapes_depend_only_on_buckets():
    """Different exact lengths in the same buckets must give the loss identical shapes (no XLA recompiles)."""

    def loss_inputs(lengths):
        model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
        micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
        return prepare_tpu_loss_inputs(model_output, micro_batch)

    out_a, data_a = loss_inputs([(5, 7), (9, 11)])
    out_b, data_b = loss_inputs([(12, 16), (3, 2)])
    for key in ("log_probs", "entropy"):
        assert out_a[key].values().shape == out_b[key].values().shape == torch.Size([2 * (16 + 1)])
    for key in ("prompts", "responses"):
        assert data_a[key].offsets().tolist() == data_b[key].offsets().tolist()
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert data_a[key].shape == data_b[key].shape == torch.Size([2, 16])
    assert tu.get_non_tensor_data(data_a, "max_response_len", None) == 16


def test_prepare_tpu_loss_inputs_ppo_falls_back_to_trimmed_outputs():
    """A packed output without its bucket-padded device tensor keeps the exact-length (trimmed) path."""
    lengths = [(3, 2), (4, 5)]
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len=32, with_entropy=False)
    model_output["values"] = torch.nested.nested_tensor_from_jagged(torch.zeros(int(cu_seqlens[-1])), cu_seqlens)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, _ppo_micro_batch(lengths, 16))
    assert loss_model_output["log_probs"].values().shape == torch.Size([int(cu_seqlens[-1])])
    assert loss_data["responses"].offsets().diff().tolist() == [2, 5]


def test_resolve_tpu_activation_in_fp32_precedence(monkeypatch):
    monkeypatch.delenv("VERL_TPU_ACTIVATION_IN_FP32", raising=False)

    # 3. Off unless asked for; no chip-based default.
    assert resolve_tpu_activation_in_fp32() is False
    assert resolve_tpu_activation_in_fp32(SimpleNamespace(activation_in_fp32=None)) is False
    assert tpu_rollout_activation_env_vars() == {}

    # 2. The engine config.
    assert resolve_tpu_activation_in_fp32(SimpleNamespace(activation_in_fp32=True)) is True
    assert resolve_tpu_activation_in_fp32(SimpleNamespace(activation_in_fp32=False)) is False
    extend_torchtitan_engine_config()
    assert resolve_tpu_activation_in_fp32(TorchtitanEngineConfig(attn_type="varlen")) is False
    assert resolve_tpu_activation_in_fp32(TorchtitanEngineConfig(attn_type="varlen", activation_in_fp32=True)) is True

    # 1. The environment beats the config, in both directions, and drives the rollout half.
    monkeypatch.setenv("VERL_TPU_ACTIVATION_IN_FP32", "0")
    assert resolve_tpu_activation_in_fp32(SimpleNamespace(activation_in_fp32=True)) is False
    assert tpu_rollout_activation_env_vars() == {}
    monkeypatch.setenv("VERL_TPU_ACTIVATION_IN_FP32", "1")
    assert resolve_tpu_activation_in_fp32(SimpleNamespace(activation_in_fp32=False)) is True
    assert tpu_rollout_activation_env_vars() == {"VLLM_TPU_ACTIVATION_FP32": "silu"}


def test_apply_activation_in_fp32_tpu():
    class _FFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.activation_in_fp32 = False

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward = _FFN()
            self.norm = torch.nn.LayerNorm(4)

    model = torch.nn.Sequential(_Block(), _Block())
    assert apply_activation_in_fp32_tpu([model]) == 2
    assert all(block.feed_forward.activation_in_fp32 for block in model)
    # Idempotent on repeated calls (e.g. __init__ followed by initialize)
    assert apply_activation_in_fp32_tpu([model]) == 0
    assert apply_activation_in_fp32_tpu(None) == 0
    assert apply_activation_in_fp32_tpu(["not a module"]) == 0

    # A torchtitan without the knob must not train silently in bf16 against an fp32 rollout.
    with pytest.raises(ValueError, match="activation_in_fp32"):
        apply_activation_in_fp32_tpu(torch.nn.Sequential(torch.nn.LayerNorm(4)))
