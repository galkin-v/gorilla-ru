from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
RU_BFCL_BENCHMARK = "ru_bfcl_v3"
RU_BFCL_SOURCE_ROOT = ROOT.parent / "ru_bfcl"
RU_BFCL_SUPPORTED_V4_CATEGORIES = {
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
}
RU_BFCL_CATEGORY_ALIASES = {
    "simple": "simple_python",
    "java": "simple_java",
    "javascript": "simple_javascript",
}
RU_BFCL_PROMPT_FILE_MAP = {
    "BFCL_v3_simple.json": "BFCL_v4_simple_python.json",
    "BFCL_v3_java.json": "BFCL_v4_simple_java.json",
    "BFCL_v3_javascript.json": "BFCL_v4_simple_javascript.json",
    "BFCL_v3_multiple.json": "BFCL_v4_multiple.json",
    "BFCL_v3_parallel.json": "BFCL_v4_parallel.json",
    "BFCL_v3_parallel_multiple.json": "BFCL_v4_parallel_multiple.json",
    "BFCL_v3_irrelevance.json": "BFCL_v4_irrelevance.json",
    "BFCL_v3_live_simple.json": "BFCL_v4_live_simple.json",
    "BFCL_v3_live_multiple.json": "BFCL_v4_live_multiple.json",
    "BFCL_v3_live_parallel.json": "BFCL_v4_live_parallel.json",
    "BFCL_v3_live_parallel_multiple.json": "BFCL_v4_live_parallel_multiple.json",
    "BFCL_v3_live_irrelevance.json": "BFCL_v4_live_irrelevance.json",
    "BFCL_v3_live_relevance.json": "BFCL_v4_live_relevance.json",
    "BFCL_v3_multi_turn_base.json": "BFCL_v4_multi_turn_base.json",
    "BFCL_v3_multi_turn_miss_func.json": "BFCL_v4_multi_turn_miss_func.json",
    "BFCL_v3_multi_turn_miss_param.json": "BFCL_v4_multi_turn_miss_param.json",
    "BFCL_v3_multi_turn_long_context.json": "BFCL_v4_multi_turn_long_context.json",
}
RU_BFCL_ID_PREFIX_REMAP = {
    "simple_": "simple_python_",
    "java_": "simple_java_",
    "javascript_": "simple_javascript_",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copilot provider contract runner for Gorilla BFCL."
    )
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--candidate-api-key", default="")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument("--request-params-json", default="{}")
    parser.add_argument("--resume", default="1")
    parser.add_argument("--show-live-stats", default="1")
    return parser.parse_args()


def _to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return _to_bool(value)
    return default


def _safe_json_obj(raw: str) -> dict[str, Any]:
    payload = json.loads(raw) if raw.strip() else {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _normalize_test_categories(raw: Any) -> list[str]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace("\n", ",").split(",")]
        normalized = [part for part in parts if part]
        return normalized or ["multi_turn"]
    if isinstance(raw, (list, tuple)):
        normalized = [str(item).strip() for item in raw if str(item).strip()]
        return normalized or ["multi_turn"]
    return ["multi_turn"]


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower() or "metric"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _serialize_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Multi-turn BFCL prompts are nested role/content objects; JSON keeps full context.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False)
    return str(value)


def _serialize_response(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _is_ru_bfcl_benchmark(benchmark_name: str) -> bool:
    return benchmark_name.strip() == RU_BFCL_BENCHMARK


def _metric_prefix_for_benchmark(benchmark_name: str) -> str:
    if _is_ru_bfcl_benchmark(benchmark_name):
        return RU_BFCL_BENCHMARK
    return "bfcl_v3"


def _remap_ru_bfcl_id(test_id: str) -> str:
    for source_prefix, target_prefix in RU_BFCL_ID_PREFIX_REMAP.items():
        if test_id.startswith(source_prefix):
            return f"{target_prefix}{test_id[len(source_prefix):]}"
    return test_id


def _rewrite_jsonl_with_ru_id_remap(source_path: Path, target_path: Path) -> None:
    rows = _load_jsonl(source_path)
    rewritten_rows: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        test_id = entry.get("id")
        if isinstance(test_id, str):
            entry["id"] = _remap_ru_bfcl_id(test_id)
        rewritten_rows.append(entry)
    _write_jsonl(target_path, rewritten_rows)


def _prepare_ru_bfcl_v4_data_layout(bfcl_project_root: Path) -> Path:
    if not RU_BFCL_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"RU BFCL dataset directory not found: {RU_BFCL_SOURCE_ROOT}")

    staged_root = bfcl_project_root / "ru_bfcl_data_v4"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)
    (staged_root / "possible_answer").mkdir(parents=True, exist_ok=True)
    (staged_root / "memory_prereq_conversation").mkdir(parents=True, exist_ok=True)

    source_func_doc_dir = RU_BFCL_SOURCE_ROOT / "multi_turn_func_doc"
    if source_func_doc_dir.exists():
        shutil.copytree(source_func_doc_dir, staged_root / "multi_turn_func_doc")

    source_answer_root = RU_BFCL_SOURCE_ROOT / "possible_answer"
    for source_name, target_name in RU_BFCL_PROMPT_FILE_MAP.items():
        source_prompt = RU_BFCL_SOURCE_ROOT / source_name
        if not source_prompt.exists():
            raise FileNotFoundError(f"Expected RU BFCL prompt file is missing: {source_prompt}")
        _rewrite_jsonl_with_ru_id_remap(source_prompt, staged_root / target_name)

        source_answer = source_answer_root / source_name
        if source_answer.exists():
            _rewrite_jsonl_with_ru_id_remap(
                source_answer,
                staged_root / "possible_answer" / target_name,
            )

    return staged_root


def _apply_data_path_override_to_bfcl_modules(data_root: Path) -> None:
    import bfcl_eval.constants.eval_config as eval_config
    import bfcl_eval.utils as bfcl_utils

    prompt_path = data_root
    possible_answer_path = data_root / "possible_answer"
    multi_turn_func_doc_path = data_root / "multi_turn_func_doc"
    memory_prereq_path = data_root / "memory_prereq_conversation"

    eval_config.PROMPT_PATH = prompt_path
    eval_config.POSSIBLE_ANSWER_PATH = possible_answer_path
    eval_config.MULTI_TURN_FUNC_DOC_PATH = multi_turn_func_doc_path
    eval_config.MEMORY_PREREQ_CONVERSATION_PATH = memory_prereq_path
    eval_config.FORMAT_SENSITIVITY_IDS_PATH = (
        prompt_path / f"{eval_config.VERSION_PREFIX}_format_sensitivity.json"
    )

    bfcl_utils.PROMPT_PATH = prompt_path
    bfcl_utils.POSSIBLE_ANSWER_PATH = possible_answer_path
    bfcl_utils.MULTI_TURN_FUNC_DOC_PATH = multi_turn_func_doc_path
    bfcl_utils.MEMORY_PREREQ_CONVERSATION_PATH = memory_prereq_path
    bfcl_utils.FORMAT_SENSITIVITY_IDS_PATH = (
        prompt_path / f"{bfcl_utils.VERSION_PREFIX}_format_sensitivity.json"
    )


def _normalize_requested_categories_for_ru_bfcl(
    requested_categories: list[str],
) -> list[str]:
    normalized: list[str] = []
    for category in requested_categories:
        category = RU_BFCL_CATEGORY_ALIASES.get(category, category)
        if category in {"all", "all_scoring"}:
            normalized.extend(["single_turn", "multi_turn"])
            continue
        normalized.append(category)
    return normalized or ["multi_turn"]


def _patch_eval_runner_for_missing_categories() -> None:
    import bfcl_eval.eval_checker.eval_runner as eval_runner_module
    import bfcl_eval.eval_checker.eval_runner_helper as eval_runner_helper_module

    original_get_category_score = eval_runner_helper_module.get_category_score

    def _safe_get_category_score(score_dict: dict, test_category: str) -> dict:
        try:
            return original_get_category_score(score_dict, test_category)
        except FileNotFoundError:
            # RU BFCL staged data can omit V4-only categories (e.g., web_search_*).
            return {"accuracy": 0, "total_count": 0, "display_accuracy": "N/A"}

    eval_runner_helper_module.get_category_score = _safe_get_category_score
    eval_runner_module.get_category_score = _safe_get_category_score


def _validate_ru_bfcl_categories(requested_categories: list[str]) -> None:
    from bfcl_eval.utils import parse_test_category_argument

    resolved_categories = parse_test_category_argument(requested_categories)
    unsupported = sorted(set(resolved_categories) - RU_BFCL_SUPPORTED_V4_CATEGORIES)
    if unsupported:
        supported = ", ".join(sorted(RU_BFCL_SUPPORTED_V4_CATEGORIES))
        unsupported_text = ", ".join(unsupported)
        raise ValueError(
            "ru_bfcl_v3 supports only BFCL v3-compatible single-turn/live/multi-turn categories. "
            f"Unsupported categories: {unsupported_text}. Supported categories: {supported}."
        )


def _normalize_run_ids_map_for_ru_bfcl(
    run_ids_map_raw: Mapping[str, Any],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for category, test_ids in run_ids_map_raw.items():
        normalized_category = RU_BFCL_CATEGORY_ALIASES.get(str(category), str(category))
        if isinstance(test_ids, (list, tuple)):
            normalized[normalized_category] = [str(test_id) for test_id in test_ids]
    return normalized


def _runtime_model_key(model_id: str, *, is_fc_model: bool) -> str:
    slug = re.sub(r"[^A-Za-z0-9._/-]+", "-", model_id).strip("-")
    if not slug:
        slug = "candidate_model"
    suffix = "-FC" if is_fc_model else ""
    return f"copilot-runtime/{slug}{suffix}"


def _configure_bfcl_env(
    *,
    bfcl_project_root: Path,
    candidate_base_url: str,
    candidate_api_key: str,
    request_params: Mapping[str, Any],
) -> None:
    base_url = candidate_base_url.rstrip("/")
    api_key = candidate_api_key or "EMPTY"
    os.environ["BFCL_PROJECT_ROOT"] = str(bfcl_project_root)
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["REMOTE_OPENAI_BASE_URL"] = base_url
    os.environ["REMOTE_OPENAI_API_KEY"] = api_key

    headers = request_params.get("bfcl_openai_default_headers")
    if isinstance(headers, Mapping):
        os.environ["OPENAI_DEFAULT_HEADERS"] = json.dumps(dict(headers), ensure_ascii=False)


def _register_runtime_model(
    *,
    candidate_model_id: str,
    requested_model_key: str | None,
    is_fc_model: bool,
    use_openai_responses: bool,
) -> str:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler

    if requested_model_key and requested_model_key in MODEL_CONFIG_MAPPING:
        return requested_model_key

    model_key = requested_model_key or _runtime_model_key(
        candidate_model_id,
        is_fc_model=is_fc_model,
    )
    handler_cls = OpenAIResponsesHandler if use_openai_responses else OpenAICompletionsHandler
    MODEL_CONFIG_MAPPING[model_key] = ModelConfig(
        model_name=candidate_model_id,
        display_name=f"{candidate_model_id} ({'FC' if is_fc_model else 'Prompt'})",
        url="",
        org="Copilot External",
        license="Proprietary",
        model_handler=handler_cls,
        input_price=None,
        output_price=None,
        is_fc_model=is_fc_model,
        underscore_to_dot=False,
    )
    return model_key


def _build_run_ids_map(
    *,
    requested_categories: list[str],
    limit_samples: int,
) -> dict[str, list[str]]:
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

    if limit_samples < 0:
        return {}

    resolved_categories = parse_test_category_argument(requested_categories)
    selected_pairs: list[tuple[str, str]] = []
    for category in resolved_categories:
        entries = load_dataset_entry(category)
        for entry in entries:
            selected_pairs.append((category, str(entry["id"])))
            if len(selected_pairs) >= limit_samples:
                break
        if len(selected_pairs) >= limit_samples:
            break

    grouped: dict[str, list[str]] = defaultdict(list)
    for category, test_id in selected_pairs:
        grouped[category].append(test_id)
    return dict(grouped)


def _load_prompts_by_id(
    *,
    categories: list[str],
) -> dict[str, str]:
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument

    prompts: dict[str, str] = {}
    for category in parse_test_category_argument(categories):
        for entry in load_dataset_entry(category):
            prompts[str(entry["id"])] = _serialize_prompt(entry.get("question", ""))
    return prompts


def _collect_scores(
    *,
    score_root: Path,
    categories: list[str],
) -> dict[str, dict[str, Any]]:
    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
        parse_test_category_argument,
    )

    per_category: dict[str, dict[str, Any]] = {}
    for category in parse_test_category_argument(categories):
        score_path = (
            score_root
            / get_directory_structure_by_category(category)
            / get_file_name_by_category(category, is_score_file=True)
        )
        rows = _load_jsonl(score_path)
        if not rows:
            continue
        header = rows[0]
        accuracy = _safe_float(header.get("accuracy"), default=0.0)
        total_count = _safe_int(header.get("total_count"), default=0)
        correct_count = _safe_int(header.get("correct_count"), default=round(accuracy * total_count))
        failed_ids = {
            str(row.get("id"))
            for row in rows[1:]
            if isinstance(row, Mapping) and row.get("id") is not None
        }
        per_category[category] = {
            "accuracy": accuracy,
            "total_count": total_count,
            "correct_count": correct_count,
            "failed_ids": failed_ids,
            "score_path": str(score_path),
        }
    return per_category


def _build_predictions(
    *,
    result_root: Path,
    categories: list[str],
    model_id: str,
    prompts_by_id: Mapping[str, str],
    category_scores: Mapping[str, Mapping[str, Any]],
    metric_prefix: str,
) -> list[dict[str, Any]]:
    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
        parse_test_category_argument,
    )

    predictions: list[dict[str, Any]] = []
    sample_id = 0
    for category in parse_test_category_argument(categories):
        result_path = (
            result_root
            / get_directory_structure_by_category(category)
            / get_file_name_by_category(category, is_result_file=True)
        )
        rows = _load_jsonl(result_path)
        failed_ids = set(category_scores.get(category, {}).get("failed_ids", set()))
        for row in rows:
            test_id = str(row.get("id", ""))
            response = _serialize_response(row.get("result"))
            error = None
            if isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"):
                error = row["result"]
            status = "error" if error else "scored"
            passed = None
            if test_id and category in category_scores:
                passed = 0 if test_id in failed_ids else 1
            scores = {}
            if passed is not None:
                scores[f"{metric_prefix}_pass"] = passed
                scores[f"{metric_prefix}_{_sanitize_metric_name(category)}_pass"] = passed

            predictions.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompts_by_id.get(test_id, ""),
                    "response": response,
                    "target": "",
                    "status": status,
                    "error": error,
                    "scores": scores,
                    "metadata": {
                        "model_id": model_id,
                        "bfcl_test_id": test_id,
                        "bfcl_category": category,
                        "bfcl_result_source": str(result_path),
                    },
                }
            )
            sample_id += 1
    return predictions


def _build_metric_values(
    category_scores: Mapping[str, Mapping[str, Any]],
    metric_prefix: str,
) -> dict[str, tuple[float, int]]:
    metric_values: dict[str, tuple[float, int]] = {}
    if not category_scores:
        return metric_values

    weighted_correct = 0
    weighted_total = 0
    unweighted_values: list[float] = []
    for category, payload in sorted(category_scores.items()):
        accuracy = _safe_float(payload.get("accuracy"), default=0.0)
        total_count = _safe_int(payload.get("total_count"), default=0)
        correct_count = _safe_int(payload.get("correct_count"), default=round(accuracy * total_count))
        metric_values[f"{metric_prefix}_{_sanitize_metric_name(category)}_accuracy"] = (
            accuracy,
            total_count,
        )
        unweighted_values.append(accuracy)
        weighted_correct += correct_count
        weighted_total += total_count

    weighted_accuracy = (weighted_correct / weighted_total) if weighted_total else 0.0
    unweighted_accuracy = (
        (sum(unweighted_values) / len(unweighted_values)) if unweighted_values else 0.0
    )
    metric_values[f"{metric_prefix}_weighted_accuracy"] = (weighted_accuracy, weighted_total)
    metric_values[f"{metric_prefix}_unweighted_accuracy"] = (
        unweighted_accuracy,
        len(unweighted_values),
    )
    return metric_values


def main() -> int:
    args = _parse_args()
    started_at = datetime.now(UTC)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    request_params = _safe_json_obj(args.request_params_json)
    show_live_stats = _to_bool(args.show_live_stats)
    resume = _to_bool(args.resume)
    parallelism = max(1, int(args.parallelism))
    limit_samples = int(args.limit_samples)

    bfcl_project_root = output_dir / "bfcl_workspace"
    bfcl_project_root.mkdir(parents=True, exist_ok=True)
    _configure_bfcl_env(
        bfcl_project_root=bfcl_project_root,
        candidate_base_url=args.candidate_base_url,
        candidate_api_key=args.candidate_api_key,
        request_params=request_params,
    )

    requested_categories = _normalize_test_categories(
        request_params.get("bfcl_test_categories", request_params.get("bfcl_test_category"))
    )
    metric_prefix = _metric_prefix_for_benchmark(args.benchmark_name)

    if _is_ru_bfcl_benchmark(args.benchmark_name):
        staged_data_root = _prepare_ru_bfcl_v4_data_layout(bfcl_project_root)
        _apply_data_path_override_to_bfcl_modules(staged_data_root)
        requested_categories = _normalize_requested_categories_for_ru_bfcl(requested_categories)
        _validate_ru_bfcl_categories(requested_categories)

    bfcl_temperature = _safe_float(
        request_params.get("bfcl_temperature", request_params.get("temperature", 0.001)),
        default=0.001,
    )
    requested_model_key = str(request_params.get("bfcl_model_registry_name", "")).strip() or None
    bfcl_is_fc_model = _coerce_bool(request_params.get("bfcl_is_fc_model"), default=True)
    use_openai_responses = _coerce_bool(
        request_params.get("bfcl_use_openai_responses"),
        default=False,
    )

    model_key = _register_runtime_model(
        candidate_model_id=args.candidate_model_id,
        requested_model_key=requested_model_key,
        is_fc_model=bfcl_is_fc_model,
        use_openai_responses=use_openai_responses,
    )

    run_ids_map_raw = request_params.get("bfcl_run_ids")
    run_ids_map: dict[str, list[str]]
    if isinstance(run_ids_map_raw, Mapping):
        if _is_ru_bfcl_benchmark(args.benchmark_name):
            run_ids_map = _normalize_run_ids_map_for_ru_bfcl(run_ids_map_raw)
        else:
            run_ids_map = {
                str(category): [str(test_id) for test_id in test_ids]
                for category, test_ids in run_ids_map_raw.items()
                if isinstance(test_ids, (list, tuple))
            }
        if run_ids_map:
            if _is_ru_bfcl_benchmark(args.benchmark_name):
                _validate_ru_bfcl_categories(sorted(run_ids_map.keys()))
            requested_categories = sorted(run_ids_map.keys())
    else:
        run_ids_map = _build_run_ids_map(
            requested_categories=requested_categories,
            limit_samples=limit_samples,
        )

    run_ids = bool(run_ids_map)
    if run_ids:
        run_ids_path = bfcl_project_root / "test_case_ids_to_generate.json"
        _write_json(run_ids_path, run_ids_map)

    result_dir_name = str(request_params.get("bfcl_result_dir", "result")).strip() or "result"
    score_dir_name = str(request_params.get("bfcl_score_dir", "score")).strip() or "score"
    allow_overwrite = _coerce_bool(
        request_params.get("bfcl_allow_overwrite"),
        default=not resume,
    )
    partial_eval = _coerce_bool(
        request_params.get("bfcl_partial_eval"),
        default=run_ids,
    )

    from bfcl_eval._llm_response_generation import main as generation_main
    import bfcl_eval.eval_checker.eval_runner as eval_runner_module

    if _is_ru_bfcl_benchmark(args.benchmark_name):
        _patch_eval_runner_for_missing_categories()

    generation_args = SimpleNamespace(
        model=[model_key],
        test_category=requested_categories,
        temperature=bfcl_temperature,
        include_input_log=_coerce_bool(request_params.get("bfcl_include_input_log"), default=False),
        exclude_state_log=_coerce_bool(request_params.get("bfcl_exclude_state_log"), default=False),
        num_gpus=_safe_int(request_params.get("bfcl_num_gpus"), default=1),
        num_threads=parallelism,
        gpu_memory_utilization=_safe_float(
            request_params.get("bfcl_gpu_memory_utilization"),
            default=0.9,
        ),
        backend=str(request_params.get("bfcl_backend", "vllm")),
        skip_server_setup=_coerce_bool(request_params.get("bfcl_skip_server_setup"), default=True),
        local_model_path=request_params.get("bfcl_local_model_path"),
        result_dir=result_dir_name,
        allow_overwrite=allow_overwrite,
        run_ids=run_ids,
        enable_lora=_coerce_bool(request_params.get("bfcl_enable_lora"), default=False),
        max_lora_rank=request_params.get("bfcl_max_lora_rank"),
        lora_modules=request_params.get("bfcl_lora_modules"),
    )

    if show_live_stats:
        print(
            f"[{args.benchmark_name}] running BFCL generation for model={model_key} categories={requested_categories}",
            flush=True,
        )
    generation_main(generation_args)
    eval_runner_module.main(
        [model_key],
        requested_categories,
        result_dir_name,
        score_dir_name,
        partial_eval=partial_eval,
    )

    model_dir_name = model_key.replace("/", "_")
    result_root = bfcl_project_root / result_dir_name / model_dir_name
    score_root = bfcl_project_root / score_dir_name / model_dir_name
    if not result_root.exists():
        raise FileNotFoundError(f"BFCL result directory not found: {result_root}")
    if not score_root.exists():
        raise FileNotFoundError(f"BFCL score directory not found: {score_root}")

    prompts_by_id = _load_prompts_by_id(categories=requested_categories)
    category_scores = _collect_scores(
        score_root=score_root,
        categories=requested_categories,
    )
    predictions = _build_predictions(
        result_root=result_root,
        categories=requested_categories,
        model_id=args.candidate_model_id,
        prompts_by_id=prompts_by_id,
        category_scores=category_scores,
        metric_prefix=metric_prefix,
    )
    metric_values = _build_metric_values(category_scores, metric_prefix=metric_prefix)

    _write_jsonl(output_dir / "byob_predictions.jsonl", predictions)
    sample_count = len(predictions)
    successful_count = sum(1 for row in predictions if row.get("error") in (None, ""))
    scores_payload = {
        metric_name: {
            "stats": {
                "count": count,
                "mean": round(value, 6),
                "stddev": 0.0,
                "stderr": 0.0,
            },
            "value": value,
        }
        for metric_name, (value, count) in metric_values.items()
    }
    _write_json(
        output_dir / "byob_results.json",
        {
            "tasks": {
                args.benchmark_name: {
                    "metrics": {
                        "pass@1": {
                            "scores": scores_payload,
                        }
                    }
                }
            }
        },
    )

    finished_at = datetime.now(UTC)
    inference_time = max(0.0, (finished_at - started_at).total_seconds())
    _write_json(
        output_dir / "eval_factory_metrics.json",
        {
            "response_stats": {
                "count": sample_count,
                "successful_count": successful_count,
                "avg_latency_ms": 0.0,
                "avg_total_tokens": 0.0,
                "avg_completion_tokens": 0.0,
            },
            "timing": {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "inference_time_seconds": inference_time,
            },
        },
    )
    _write_json(
        output_dir / "params.json",
        {
            "parallelism": parallelism,
            "request_timeout": max(1, int(args.request_timeout)),
            "limit_samples": limit_samples if limit_samples >= 0 else None,
            "resume": resume,
            "show_live_stats": show_live_stats,
            "request_params": request_params,
            "bfcl_model_key": model_key,
            "bfcl_categories": requested_categories,
            "bfcl_result_root": str(result_root),
            "bfcl_score_root": str(score_root),
            "bfcl_project_root": str(bfcl_project_root),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
