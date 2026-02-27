import dataclasses
import logging

from miles.utils import megatron_bridge_utils
from miles.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .common import all_gather_param
from .hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import miles_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights):
        from miles.utils.lora_utils import is_lora_enabled

        if is_lora_enabled(self.args):
            yield from self._get_hf_weight_chunks_lora(megatron_local_weights)
        else:
            yield from self._get_hf_weight_chunks_vanilla(megatron_local_weights)

    def _get_hf_weight_chunks_lora(self, megatron_local_weights):
        """LoRA merge-on-sync: merge LoRA into base weights, convert to HF format directly.

        Bypasses bridge export_hf_weights entirely because LoRA wrapping changes
        the model structure in ways bridge doesn't understand. Instead, uses
        convert_to_hf (same as distributed mode).
        """
        import re

        from miles.utils.lora_utils import is_lora_param
        from ..megatron_to_hf import convert_to_hf

        scaling = self.args.lora_alpha / self.args.lora_rank

        # Separate base and LoRA weights, strip module. and vp_stages.X. prefixes
        def _strip_all_prefixes(name):
            name = strip_param_name_prefix(name)  # strip module.
            name = re.sub(r"^vp_stages\.\d+\.", "", name)  # strip vp_stages.X.
            return name

        base_weights = {}
        lora_weights = {}
        for k, v in megatron_local_weights.items():
            clean_k = _strip_all_prefixes(k)
            if is_lora_param(clean_k):
                lora_weights[clean_k] = v
            else:
                base_weights[clean_k] = v

        # Debug: log sample keys to understand naming patterns
        base_sample = list(base_weights.keys())[:5]
        lora_sample = list(lora_weights.keys())[:10]
        logger.info(f"[LoRA merge] base_weights count={len(base_weights)}, sample keys: {base_sample}")
        logger.info(f"[LoRA merge] lora_weights count={len(lora_weights)}, sample keys: {lora_sample}")

        # Merge LoRA into base weights and convert to HF format
        merge_count = 0
        skip_count = 0
        shape_mismatch_count = 0
        tp_attr_missing_count = 0

        tp_attr_defaults = {
            "tensor_model_parallel": False,
            "partition_dim": 0,
            "partition_stride": 1,
            "parallel_mode": None,
        }

        def _copy_tp_attrs(src, dst):
            for k, default_v in tp_attr_defaults.items():
                setattr(dst, k, getattr(src, k, default_v))

        def _to_cuda_with_tp_attrs(x, *, name):
            y = x.cuda()
            _copy_tp_attrs(x, y)
            if not hasattr(x, "tensor_model_parallel"):
                nonlocal tp_attr_missing_count
                tp_attr_missing_count += 1
                if tp_attr_missing_count <= 5:
                    logger.warning(f"[LoRA merge] missing TP attrs on backup tensor, fallback defaults: {name}")
            return y

        def _get_tp_rank_and_size():
            try:
                from megatron.core import parallel_state

                return parallel_state.get_tensor_model_parallel_rank(), parallel_state.get_tensor_model_parallel_world_size()
            except Exception:
                return 0, max(1, int(getattr(self.args, "tensor_model_parallel_size", 1) or 1))

        def _align_delta_to_param(delta, param):
            # Common case: already aligned.
            if delta.shape == param.shape:
                return delta
            # Some modules may have opposite orientation.
            if delta.T.shape == param.shape:
                return delta.T
            # TP-sharded case: take rank-local shard on a compatible dimension.
            if delta.ndim == 2 and param.ndim == 2:
                tp_rank, tp_size = _get_tp_rank_and_size()
                if delta.shape[0] % param.shape[0] == 0 and delta.shape[1] == param.shape[1]:
                    n = delta.shape[0] // param.shape[0]
                    idx = tp_rank % n if n > 0 else 0
                    start = idx * param.shape[0]
                    return delta.narrow(0, start, param.shape[0])
                if delta.shape[1] % param.shape[1] == 0 and delta.shape[0] == param.shape[0]:
                    n = delta.shape[1] // param.shape[1]
                    idx = tp_rank % n if n > 0 else 0
                    start = idx * param.shape[1]
                    return delta.narrow(1, start, param.shape[1])
                if tp_size > 1:
                    # Best effort fallback: chunk dim0 for TP if divisible.
                    if delta.shape[0] % tp_size == 0 and delta.shape[1] == param.shape[1]:
                        chunk = delta.shape[0] // tp_size
                        start = (tp_rank % tp_size) * chunk
                        if chunk == param.shape[0]:
                            return delta.narrow(0, start, chunk)
            return None

        def _merged_hf_weights():
            nonlocal merge_count, skip_count, shape_mismatch_count
            lora_full_cache = {}

            def _get_full_lora_weight(lora_name, lora_weight):
                if lora_name not in lora_full_cache:
                    local_lora = _to_cuda_with_tp_attrs(lora_weight, name=lora_name)
                    lora_full_cache[lora_name] = all_gather_param(lora_name, local_lora)
                return lora_full_cache[lora_name]

            for name, param in base_weights.items():
                # Strip .to_wrap. from LoRA-wrapped base param names
                clean_name = name.replace(".to_wrap.", ".")
                local_param = _to_cuda_with_tp_attrs(param, name=clean_name)
                # Reconstruct full Megatron tensor from TP shards before HF conversion.
                full_param = all_gather_param(clean_name, local_param)

                # Find matching LoRA A/B
                base_module = clean_name.rsplit(".", 1)[0]
                param_suffix = clean_name.rsplit(".", 1)[1]

                if param_suffix == "weight":
                    lora_a_key = f"{base_module}.adapter.linear_in.weight"
                    lora_b_key = f"{base_module}.adapter.linear_out.weight"
                    lora_a_key_alt = name.replace(".to_wrap.weight", ".adapter.linear_in.weight")
                    lora_b_key_alt = name.replace(".to_wrap.weight", ".adapter.linear_out.weight")

                    lora_a = lora_weights.get(lora_a_key)
                    lora_a_name = lora_a_key
                    if lora_a is None:
                        lora_a = lora_weights.get(lora_a_key_alt)
                        lora_a_name = lora_a_key_alt

                    lora_b = lora_weights.get(lora_b_key)
                    lora_b_name = lora_b_key
                    if lora_b is None:
                        lora_b = lora_weights.get(lora_b_key_alt)
                        lora_b_name = lora_b_key_alt

                    if lora_a is not None and lora_b is not None:
                        lora_a_full = _get_full_lora_weight(lora_a_name, lora_a)
                        lora_b_full = _get_full_lora_weight(lora_b_name, lora_b)
                        if (
                            lora_a_full.ndim != 2
                            or lora_b_full.ndim != 2
                            or lora_b_full.shape[1] != lora_a_full.shape[0]
                        ):
                            shape_mismatch_count += 1
                            if shape_mismatch_count <= 5:
                                logger.warning(
                                    f"[LoRA merge] matmul mismatch, skip merge for {name}: "
                                    f"lora_b={tuple(lora_b_full.shape)} lora_a={tuple(lora_a_full.shape)} "
                                    f"(expected lora_b.shape[1] == lora_a.shape[0])"
                                )
                            delta = None
                        else:
                            try:
                                delta = (lora_b_full @ lora_a_full) * scaling
                            except RuntimeError as e:
                                shape_mismatch_count += 1
                                if shape_mismatch_count <= 5:
                                    logger.warning(f"[LoRA merge] matmul runtime mismatch, skip {name}: {e}")
                                delta = None

                        if delta is None:
                            aligned_delta = None
                        else:
                            aligned_delta = _align_delta_to_param(delta, full_param)
                        if aligned_delta is None:
                            shape_mismatch_count += 1
                            if shape_mismatch_count <= 5:
                                logger.warning(
                                    f"[LoRA merge] shape mismatch, skip merge for {name}: "
                                    f"param={tuple(full_param.shape)} "
                                    f"delta={None if delta is None else tuple(delta.shape)}"
                                )
                        else:
                            merge_count += 1
                            full_param = full_param + aligned_delta
                    elif ".to_wrap." in name:
                        skip_count += 1
                        if skip_count <= 3:
                            logger.warning(
                                f"[LoRA merge] FAILED to find LoRA for base={name}, "
                                f"tried keys: {lora_a_key}, {lora_b_key}, {lora_a_key_alt}, {lora_b_key_alt}"
                            )

                # Use convert_to_hf to convert Megatron param to HF format
                # convert_to_hf expects names with module.module. prefix
                megatron_name = f"module.module.{clean_name}"
                hf_tensors = convert_to_hf(self.args, self.model_name, megatron_name, full_param)
                for hf_name, hf_param in hf_tensors:
                    yield (hf_name, hf_param.cuda().contiguous().clone())

            logger.info(
                f"[LoRA merge] Done: merged={merge_count}, skipped={skip_count}, "
                f"shape_mismatch_skipped={shape_mismatch_count}, "
                f"tp_attr_missing={tp_attr_missing_count}, "
                f"total_base={len(base_weights)}, total_lora={len(lora_weights)}"
            )

        yield from chunk_named_params_by_size(_merged_hf_weights(), chunk_size=self.args.update_weight_buffer_size)

    def _get_hf_weight_chunks_vanilla(self, megatron_local_weights):
        """Original non-LoRA path using bridge."""
        from miles.utils.lora_utils import is_lora_param

        renamed_megatron_local_weights = {
            strip_param_name_prefix(k): v for k, v in megatron_local_weights.items() if not is_lora_param(k)
        }
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            named_weights = (
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=hf_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight in named_weights
            )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
