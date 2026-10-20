# Google TPU vLLM Rollout

The plugin runs verl's vLLM rollout on TPU. With `VERL_PLATFORM=tpu`, the usual
`actor_rollout_ref.rollout.name=vllm` setting selects the TPU rollout; other platforms keep verl's
stock vLLM rollout.

Training and rollout run on separate TPU slices (a slice is a group of TPU hosts joined by a fast
chip interconnect). One vLLM engine spans all chips of the rollout slice and receives the trainer's
updated weights through the `ray-plasma-store` or `tpu-sync` checkpoint engine (see [Weight Sync](./weight_sync.md)).

## Requirements

- The prerequisites from the [Installation Guide](./install_guidance.md), plus vLLM `v0.31.0` and
  [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) `33ee547b`, the versions the rollout is tested with (see the `Dockerfile` in verl's
  [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo)).
- A verl checkout with the TPU GRPO examples in `examples/tpu/grpo`.
- At least two TPU slices, one for training and one for rollout. A TPU chip belongs to a single
  process, so training and rollout cannot share chips.
- On KubeRay, a TPU worker group named `tpu-group` with one replica per slice. Ray then labels the
  slices `tpu-group-0`, `tpu-group-1`, ..., which the plugin uses to give training and rollout a
  slice each.

## Run

The reference recipes are in verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo):
GRPO on GSM8K with Qwen3-0.6B (a 5-step smoke test) and Qwen3-4B, on two TPU v6e-4 slices. Its README walks
through the image, the KubeRay cluster and the `ray job submit` command, which loads the plugin in every Ray
worker (`py_modules` and `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin`).

To keep the plugin's `INFO` and `WARNING` logs from the driver process, such as the weight-sync timings (see
[Weight Sync](./weight_sync.md#verify)), add `"VERL_LOGGING_LEVEL": "INFO"` and
`"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"` to the job's `env_vars`.

The rollout settings that matter on TPU (the recipes already set them):

| Setting | Value on TPU |
|---------|--------------|
| `actor_rollout_ref.rollout.name` | `vllm` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | All chips of the rollout slice, e.g. `4` on v6e-4, for one replica; the chips of one host for one replica per host (see [Weight Sync](./weight_sync.md#several-rollout-replicas)) |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` (the default); larger values are rejected |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `ray-plasma-store`, or `tpu-sync` (see [Weight Sync](./weight_sync.md)) |
| `actor_rollout_ref.hybrid_engine` | `False` |

## Verify

On any host, without vLLM or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_vllm_rollout.py -v
```

On TPU, the job log contains

```text
Registered rollout replica loader: vllm (TPU-aware)
Reusing 2 verl placement group(s) for TPU rollout
```

where the number is the number of rollout hosts, and the job ends with status `SUCCEEDED`. The rollout is
tested with Qwen3-0.6B and Qwen3-4B on two v6e-4 slices and on two v6e-8 slices.

## Related Documentation

- [User Guide](./README.md)
- [Installation Guide](./install_guidance.md)
- For developers: the code in `verl_hardware_plugin/accelerators/tpu/rollout/` documents how each part works and which
  upstream fix would make each vLLM patch unnecessary.
