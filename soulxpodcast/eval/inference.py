#!/usr/bin/env python3
"""Run SoulXPodcast inference on generic JSONL samples.

Supported input records:
  {"utt": "utt_0018", "prompt_audio": "audio/prompt.wav",
   "prompt_text": "...", "target_text": "...", "context": "..."}

  {"utt": "case1", "prompt_audio": ["audio/S1.wav", "audio/S2.wav"],
   "prompt_text": ["...", "..."], "target_text": ["...", "..."],
   "speaker": [0, 1], "context": ["...", "..."]}
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


SPEAKER_TAG_RE = re.compile(r"\[S(\d+)\]")


def resolve_path(base_path: str, value: str) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(base_path)), value))


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "sample"


def copy_reference_audio(prompt_wavs: list[str], refer_dir: Path, utt: str) -> list[str]:
    refer_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    safe_utt = safe_stem(utt)
    for prompt_idx, prompt_wav in enumerate(prompt_wavs):
        suffix = Path(prompt_wav).suffix or ".wav"
        filename = f"{safe_utt}__prompt{prompt_idx}{suffix}"
        dst = refer_dir / filename
        if os.path.abspath(prompt_wav) != os.path.abspath(dst):
            shutil.copy2(prompt_wav, dst)
        copied_paths.append(str(dst.resolve()))
    return copied_paths


def parse_gpus(gpus: str) -> list[str]:
    if not gpus:
        return []
    parsed = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not parsed:
        raise ValueError("--gpus was provided but no GPU id was parsed")
    return parsed


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def preserve_path_shape(original_value: Any, path_values: list[str]) -> str | list[str]:
    if isinstance(original_value, list):
        return path_values
    if len(path_values) == 1:
        return path_values[0]
    return path_values


def normalize_required_list(value: Any, field_name: str, sample_desc: str) -> list[str]:
    if value is None:
        raise ValueError(f"{sample_desc}: missing {field_name}")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{sample_desc}: {field_name} must be a string or list")
    if not value:
        raise ValueError(f"{sample_desc}: {field_name} must not be empty")
    return [stringify(item) for item in value]


def normalize_optional_list(value: Any, expected_len: int, field_name: str, sample_desc: str) -> list[str]:
    if value is None:
        return [""] * expected_len
    if isinstance(value, str):
        if expected_len == 1:
            return [value]
        raise ValueError(f"{sample_desc}: {field_name} must be a list for {expected_len} turns")
    if not isinstance(value, list):
        raise ValueError(f"{sample_desc}: {field_name} must be a string or list")
    if len(value) != expected_len:
        raise ValueError(f"{sample_desc}: {field_name} length mismatch, expected {expected_len}, got {len(value)}")
    return [stringify(item) for item in value]


def normalize_speaker_list(value: Any, expected_len: int, num_speakers: int, sample_desc: str) -> list[int]:
    if value is None:
        if num_speakers == 1:
            return [0] * expected_len
        raise ValueError(f"{sample_desc}: multi-speaker sample must provide speaker or dialogue_text")
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{sample_desc}: speaker must be an integer or list of integers")
    if len(value) != expected_len:
        raise ValueError(f"{sample_desc}: speaker length mismatch, expected {expected_len}, got {len(value)}")

    speakers = []
    for turn_idx, speaker in enumerate(value):
        if not isinstance(speaker, int):
            raise ValueError(f"{sample_desc}: speaker[{turn_idx}] must be an integer")
        if speaker < 0 or speaker >= num_speakers:
            raise ValueError(f"{sample_desc}: speaker[{turn_idx}]={speaker} outside prompt range [0, {num_speakers - 1}]")
        speakers.append(int(speaker))
    return speakers


def parse_dialogue_text(dialogue_text: str, num_speakers: int, sample_desc: str) -> tuple[list[int], list[str]]:
    if not isinstance(dialogue_text, str) or not dialogue_text.strip():
        raise ValueError(f"{sample_desc}: dialogue_text must be a non-empty string")
    matches = list(SPEAKER_TAG_RE.finditer(dialogue_text))
    if not matches:
        if num_speakers == 1:
            return [0], [dialogue_text.strip()]
        raise ValueError(f"{sample_desc}: dialogue_text must use [S1]...[Sn] speaker tags")

    speakers = []
    texts = []
    for index, match in enumerate(matches):
        speaker = int(match.group(1)) - 1
        if speaker < 0 or speaker >= num_speakers:
            raise ValueError(f"{sample_desc}: dialogue_text speaker S{speaker + 1} outside prompt range [S1, S{num_speakers}]")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(dialogue_text)
        text = dialogue_text[start:end].strip()
        if not text:
            raise ValueError(f"{sample_desc}: empty target text after {match.group(0)}")
        speakers.append(speaker)
        texts.append(text)
    return speakers, texts


def build_target_text_list(texts: list[str], speakers: list[int], contexts: list[str], use_context: bool, context_template: str) -> list[str]:
    result = []
    for text, speaker, context in zip(texts, speakers, contexts):
        final_text = text
        if use_context and context:
            final_text = context_template.format(context=context, text=text)
        result.append(f"[S{speaker + 1}]{final_text}")
    return result


def build_dialogue_text(texts: list[str], speakers: list[int]) -> str:
    return "".join(f"[S{speaker + 1}]{text}" for text, speaker in zip(texts, speakers))


def load_cases(input_jsonl: str, args, refer_dir: Path | None = None) -> list[dict]:
    cases = []
    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line_idx, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            sample_desc = f"{input_jsonl}:{line_idx}"
            if not isinstance(record, dict):
                raise ValueError(f"{sample_desc}: each line must be a JSON object")

            utt = stringify(record.get("utt") or f"sample_{line_idx:05d}")
            prompt_audio_values = normalize_required_list(record.get("prompt_audio"), "prompt_audio", sample_desc)
            prompt_texts = normalize_required_list(record.get("prompt_text"), "prompt_text", sample_desc)
            if len(prompt_audio_values) != len(prompt_texts):
                raise ValueError(
                    f"{sample_desc}: prompt_audio and prompt_text length mismatch: "
                    f"{len(prompt_audio_values)} vs {len(prompt_texts)}"
                )

            prompt_wavs = [resolve_path(input_jsonl, value) for value in prompt_audio_values]
            for prompt_idx, prompt_wav in enumerate(prompt_wavs):
                if not prompt_wav or not os.path.exists(prompt_wav):
                    raise FileNotFoundError(f"{sample_desc}: prompt_audio[{prompt_idx}] not found: {prompt_audio_values[prompt_idx]}")
            prompt_wavs_for_infer = copy_reference_audio(prompt_wavs, refer_dir, utt) if refer_dir else prompt_wavs

            if record.get("dialogue_text") is not None:
                speakers, target_texts = parse_dialogue_text(record["dialogue_text"], len(prompt_wavs_for_infer), sample_desc)
            else:
                target_texts = normalize_required_list(record.get("target_text"), "target_text", sample_desc)
                speakers = normalize_speaker_list(record.get("speaker"), len(target_texts), len(prompt_wavs_for_infer), sample_desc)

            contexts = normalize_optional_list(record.get("context"), len(target_texts), "context", sample_desc)
            prompt_contexts = normalize_optional_list(record.get("prompt_context"), len(prompt_wavs), "prompt_context", sample_desc)
            target_text_list = build_target_text_list(
                target_texts,
                speakers,
                contexts,
                use_context=args.use_context,
                context_template=args.context_template,
            )

            cases.append(
                {
                    "input_record": record,
                    "utt": utt,
                    "text": build_dialogue_text(target_texts, speakers),
                    "target_texts": target_texts,
                    "target_contexts": contexts,
                    "target_text_list": target_text_list,
                    "speakers": speakers,
                    "prompt_wav_list": prompt_wavs_for_infer,
                    "prompt_audio": prompt_wavs_for_infer if len(prompt_wavs_for_infer) > 1 else prompt_wavs_for_infer[0],
                    "source_prompt_audio": prompt_wavs if len(prompt_wavs) > 1 else prompt_wavs[0],
                    "prompt_text_list": prompt_texts,
                    "prompt_text": prompt_texts if len(prompt_texts) > 1 else prompt_texts[0],
                    "prompt_context": prompt_contexts if len(prompt_contexts) > 1 else prompt_contexts[0],
                    "source_jsonl": os.path.abspath(input_jsonl),
                }
            )
    return cases


def apply_context_args(model, args, num_speakers: int, num_turns: int) -> None:
    if args.keep_full_context:
        model.config.max_turn_size = max(model.config.max_turn_size, num_speakers + num_turns + 10)
        model.config.turn_tokens_threshold = max(model.config.turn_tokens_threshold, args.full_context_token_threshold)
        model.config.prompt_context = max(model.config.prompt_context, num_speakers)
        model.config.history_context = max(model.config.history_context, num_turns)
        model.config.history_text_context = max(model.config.history_text_context, num_turns)

    for attr in (
        "max_turn_size",
        "turn_tokens_threshold",
        "prompt_context",
        "history_context",
        "history_text_context",
    ):
        value = getattr(args, attr)
        if value is not None:
            setattr(model.config, attr, value)


def synthesize_case(model, dataset, case: dict, args):
    import torch

    from soulxpodcast.utils.infer_utils import check_models, process_single_input

    apply_context_args(
        model=model,
        args=args,
        num_speakers=len(case["prompt_wav_list"]),
        num_turns=len(case["target_text_list"]),
    )

    inputs = process_single_input(
        dataset=dataset,
        target_text_list=case["target_text_list"],
        prompt_wav_list=case["prompt_wav_list"],
        prompt_text_list=case["prompt_text_list"],
        use_dialect_prompt=False,
        dialect_prompt_text_list=[],
    )
    if args.max_tokens is not None:
        inputs["sampling_params"].max_tokens = args.max_tokens
    check_models(args.model_path, inputs)

    outputs = model.forward_longform(**inputs)
    silence = torch.zeros(int(args.sample_rate * args.silence_ms / 1000.0))
    segments = []
    for wav in outputs["generated_wavs"]:
        segments.append(wav.squeeze().detach().cpu())
        if args.silence_ms > 0:
            segments.append(silence)
    if not segments:
        raise RuntimeError(f"{case['utt']}: model returned no generated wavs")
    return torch.cat(segments[:-1] if args.silence_ms > 0 else segments, dim=-1)


def save_case_audio(model, dataset, case: dict, output_audio: Path, args) -> None:
    import torchaudio

    wav = synthesize_case(model, dataset, case, args)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_audio), wav.unsqueeze(0), args.sample_rate)


def relpath_or_empty(path_value: Any, output_root: Path) -> str:
    if not path_value:
        return ""
    return os.path.relpath(os.path.abspath(stringify(path_value)), output_root)


def make_output_record(case: dict, output_audio: Path, output_root: Path) -> dict:
    record = dict(case["input_record"])
    record.setdefault("utt", case["utt"])
    record["prompt_audio"] = preserve_path_shape(
        record.get("prompt_audio"),
        [relpath_or_empty(prompt_wav, output_root) for prompt_wav in case["prompt_wav_list"]],
    )
    record["output_audio"] = relpath_or_empty(output_audio, output_root)
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def output_paths_for_input(input_jsonl: str, args) -> tuple[str, Path, Path, Path]:
    subset_name = Path(input_jsonl).stem
    output_root = Path(args.output_dir).resolve()
    if len(args.input_jsonl) > 1:
        output_root = output_root / subset_name
    samples_dir = output_root / "samples"
    refer_dir = output_root / "refer"
    output_jsonl_path = output_root / f"{subset_name}_inference.jsonl"
    return subset_name, samples_dir, refer_dir, output_jsonl_path


def write_outputs(output_jsonl_path: Path, records_with_cases: list[tuple[dict, dict]]) -> str:
    legacy_manifest_path = output_jsonl_path.parent / "manifest.csv"
    if legacy_manifest_path.exists():
        legacy_manifest_path.unlink()
        logging.info("Removed legacy %s", legacy_manifest_path)

    records = [record for record, _ in records_with_cases]
    write_jsonl(output_jsonl_path, records)
    logging.info("Wrote %s", output_jsonl_path)
    return str(output_jsonl_path)


def run_input(model, dataset, input_jsonl: str, args) -> str:
    subset_name, samples_dir, refer_dir, output_jsonl_path = output_paths_for_input(input_jsonl, args)
    samples_dir.mkdir(parents=True, exist_ok=True)
    refer_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(input_jsonl, args, refer_dir=refer_dir)
    logging.info("Loaded %d cases from %s", len(cases), input_jsonl)

    records = []
    for index, case in enumerate(cases, start=1):
        output_audio = samples_dir / f"{case['utt']}.wav"
        logging.info("[%s %d/%d] synthesizing %s", subset_name, index, len(cases), case["utt"])
        if args.skip_existing and output_audio.exists():
            logging.info("Skipping existing output: %s", output_audio)
        else:
            save_case_audio(model, dataset, case, output_audio, args)

        if output_audio.exists():
            records.append((make_output_record(case, output_audio, output_jsonl_path.parent), case))
    return write_outputs(output_jsonl_path, records)


def worker_loop(worker_id: int, gpu_id: str, task_queue, result_queue, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s worker{worker_id}/gpu{gpu_id} %(message)s")

    args = argparse.Namespace(**args_dict)
    try:
        logging.info("Initializing model from %s with %s engine", args.model_path, args.llm_engine)
        from soulxpodcast.utils.infer_utils import initiate_model

        model, dataset = initiate_model(
            seed=args.seed + worker_id,
            model_path=args.model_path,
            llm_engine=args.llm_engine,
            fp16_flow=args.fp16_flow,
        )
        logging.info("Model initialized")
        result_queue.put({"type": "ready", "worker_id": worker_id, "gpu_id": gpu_id})

        while True:
            task = task_queue.get()
            if task is None:
                result_queue.put({"type": "done", "worker_id": worker_id, "gpu_id": gpu_id})
                break

            case = task["case"]
            output_audio = Path(task["output_audio"])
            try:
                logging.info("[%s] synthesizing %s", task["subset_name"], case["utt"])
                save_case_audio(model, dataset, case, output_audio, args)
                result_queue.put(
                    {
                        "type": "result",
                        "ok": True,
                        "subset_name": task["subset_name"],
                        "order": task["order"],
                        "record": make_output_record(case, output_audio, Path(task["output_root"])),
                        "case": case,
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result_queue.put(
                    {
                        "type": "result",
                        "ok": False,
                        "subset_name": task["subset_name"],
                        "order": task["order"],
                        "utt": case.get("utt", ""),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "type": "fatal",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_parallel(input_jsonls: list[str], args, gpu_ids: list[str]) -> list[str]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    records_by_subset: dict[str, list[tuple[int, dict, dict]]] = {}
    paths_by_subset: dict[str, Path] = {}
    task_count = 0

    for input_jsonl in input_jsonls:
        subset_name, samples_dir, refer_dir, output_jsonl_path = output_paths_for_input(input_jsonl, args)
        samples_dir.mkdir(parents=True, exist_ok=True)
        refer_dir.mkdir(parents=True, exist_ok=True)
        paths_by_subset[subset_name] = output_jsonl_path
        records_by_subset[subset_name] = []

        cases = load_cases(input_jsonl, args, refer_dir=refer_dir)
        logging.info("Loaded %d cases from %s", len(cases), input_jsonl)
        for order, case in enumerate(cases):
            output_audio = samples_dir / f"{case['utt']}.wav"
            if args.skip_existing and output_audio.exists():
                logging.info("Skipping existing output: %s", output_audio)
                records_by_subset[subset_name].append((order, make_output_record(case, output_audio, output_jsonl_path.parent), case))
                continue
            task_queue.put(
                {
                    "subset_name": subset_name,
                    "order": order,
                    "case": case,
                    "output_audio": str(output_audio),
                    "output_root": str(output_jsonl_path.parent),
                }
            )
            task_count += 1

    if task_count == 0:
        logging.info("No queued tasks; writing JSONL from existing outputs.")
        output_jsonls = []
        for subset_name, records_with_order in records_by_subset.items():
            output_jsonl_path = paths_by_subset[subset_name]
            records = [(record, case) for _, record, case in sorted(records_with_order, key=lambda item: item[0])]
            output_jsonls.append(write_outputs(output_jsonl_path, records))
        return output_jsonls

    worker_gpu_ids = gpu_ids[: min(len(gpu_ids), task_count)]
    if len(worker_gpu_ids) < len(gpu_ids):
        logging.info("Using %d of %d requested GPUs because only %d task(s) require synthesis", len(worker_gpu_ids), len(gpu_ids), task_count)

    for _ in worker_gpu_ids:
        task_queue.put(None)

    args_dict = vars(args).copy()
    processes = []
    for worker_id, gpu_id in enumerate(worker_gpu_ids):
        logging.info("Starting worker %d on GPU %s (%d/%d)", worker_id, gpu_id, worker_id + 1, len(worker_gpu_ids))
        process = ctx.Process(target=worker_loop, args=(worker_id, gpu_id, task_queue, result_queue, args_dict))
        process.start()
        processes.append(process)
        if args.worker_start_stagger_sec > 0 and worker_id + 1 < len(worker_gpu_ids):
            time.sleep(args.worker_start_stagger_sec)

    ready_workers = 0
    done_workers = 0
    finished_tasks = 0
    errors = []
    closed_workers = set()
    last_status_log = time.monotonic()
    status_interval = max(1.0, args.worker_status_interval_sec)
    while done_workers < len(processes):
        try:
            message = result_queue.get(timeout=status_interval)
        except queue.Empty:
            for worker_id, process in enumerate(processes):
                if worker_id in closed_workers or process.exitcode is None:
                    continue
                closed_workers.add(worker_id)
                done_workers += 1
                if process.exitcode != 0:
                    errors.append(
                        {
                            "type": "fatal",
                            "worker_id": worker_id,
                            "gpu_id": gpu_ids[worker_id],
                            "error": f"process exited with code {process.exitcode}",
                            "traceback": "",
                        }
                    )
                    logging.error("Worker %s on GPU %s exited with code %s", worker_id, gpu_ids[worker_id], process.exitcode)
            now = time.monotonic()
            if now - last_status_log >= status_interval:
                alive_workers = sum(1 for process in processes if process.is_alive())
                logging.info(
                    "Waiting for workers: ready=%d/%d, done=%d/%d, tasks=%d/%d, alive=%d",
                    ready_workers,
                    len(processes),
                    done_workers,
                    len(processes),
                    finished_tasks,
                    task_count,
                    alive_workers,
                )
                last_status_log = now
            continue

        msg_type = message.get("type")
        if msg_type == "ready":
            ready_workers += 1
            logging.info("Worker %s ready on GPU %s (%d/%d)", message["worker_id"], message["gpu_id"], ready_workers, len(processes))
        elif msg_type == "done":
            closed_workers.add(message["worker_id"])
            done_workers += 1
            logging.info("Worker %s done on GPU %s (%d/%d)", message["worker_id"], message["gpu_id"], done_workers, len(processes))
        elif msg_type == "fatal":
            closed_workers.add(message["worker_id"])
            errors.append(message)
            done_workers += 1
            logging.error("Worker %s fatal on GPU %s: %s", message["worker_id"], message["gpu_id"], message["error"])
        elif msg_type == "result":
            finished_tasks += 1
            if message["ok"]:
                records_by_subset[message["subset_name"]].append((message["order"], message["record"], message["case"]))
                logging.info("Finished %d/%d: %s on GPU %s", finished_tasks, task_count, message["record"]["utt"], message["gpu_id"])
            else:
                errors.append(message)
                logging.error("Failed %d/%d: %s on GPU %s: %s", finished_tasks, task_count, message["utt"], message["gpu_id"], message["error"])

    for process in processes:
        process.join()

    output_jsonls = []
    for subset_name, records_with_order in records_by_subset.items():
        output_jsonl_path = paths_by_subset[subset_name]
        records = [(record, case) for _, record, case in sorted(records_with_order, key=lambda item: item[0])]
        output_jsonls.append(write_outputs(output_jsonl_path, records))

    if errors:
        first = errors[0]
        raise RuntimeError(
            f"{len(errors)} task(s)/worker(s) failed. First error: {first.get('utt', 'worker')} {first.get('error')}\n"
            f"{first.get('traceback', '')}"
        )
    return output_jsonls


def get_args():
    parser = argparse.ArgumentParser(description="Synthesize generic SoulXPodcast JSONL samples.")
    parser.add_argument("--model-path", required=True, help="SoulXPodcast model directory.")
    parser.add_argument("--input-jsonl", action="append", required=True, help="Input JSONL path. Can be provided multiple times.")
    parser.add_argument("--output-dir", required=True, help="Directory for samples/, refer/, and output JSONL.")
    parser.add_argument("--llm-engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. 0 or 0,1,2. Multi-GPU uses one worker per GPU.")
    parser.add_argument("--max-workers", type=int, default=None, help="Limit worker count to the first N GPUs from --gpus.")
    parser.add_argument("--worker-start-stagger-sec", type=float, default=0.0, help="Seconds to wait between starting multi-GPU workers.")
    parser.add_argument("--worker-status-interval-sec", type=float, default=30.0, help="Seconds between parent progress logs while waiting for workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16-flow", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--silence-ms", type=float, default=200.0)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Prepend each turn context to target text before synthesis. By default context is wrapped as <|start_prompt|>...<|end_prompt|> before the text; without this flag context is only copied to outputs.",
    )
    parser.add_argument(
        "--context-template",
        default="<|start_prompt|>{context}<|end_prompt|>{text}",
        help="Template used when --use-context is enabled. Available fields: {context}, {text}.",
    )
    parser.add_argument(
        "--keep-full-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise longform context thresholds per case so middle history is not compacted. Disabled by default.",
    )
    parser.add_argument("--full-context-token-threshold", type=int, default=10**9)
    parser.add_argument("--max-turn-size", type=int, default=None)
    parser.add_argument("--turn-tokens-threshold", type=int, default=None)
    parser.add_argument("--prompt-context", type=int, default=None)
    parser.add_argument("--history-context", type=int, default=None)
    parser.add_argument("--history-text-context", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override SamplingParams.max_tokens per turn.")
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.model_path = os.path.abspath(args.model_path)
    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path is not a directory: {args.model_path}")
    input_jsonls = [os.path.abspath(path) for path in args.input_jsonl]
    for input_jsonl in input_jsonls:
        if not os.path.exists(input_jsonl):
            raise FileNotFoundError(f"--input-jsonl not found: {input_jsonl}")
    gpu_ids = parse_gpus(args.gpus)
    if args.max_workers is not None:
        if args.max_workers < 1:
            raise ValueError("--max-workers must be at least 1")
        if len(gpu_ids) > args.max_workers:
            logging.info("Limiting workers to first %d GPU(s) from --gpus: %s", args.max_workers, ",".join(gpu_ids[: args.max_workers]))
            gpu_ids = gpu_ids[: args.max_workers]

    if len(gpu_ids) > 1:
        logging.info("Running queue-based multi-GPU inference on GPUs: %s", ",".join(gpu_ids))
        output_jsonls = run_parallel(input_jsonls, args, gpu_ids)
    else:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            logging.info("Running single-GPU inference on GPU: %s", gpu_ids[0])

        from soulxpodcast.utils.infer_utils import initiate_model

        model, dataset = initiate_model(
            seed=args.seed,
            model_path=args.model_path,
            llm_engine=args.llm_engine,
            fp16_flow=args.fp16_flow,
        )
        output_jsonls = [run_input(model, dataset, input_jsonl, args) for input_jsonl in input_jsonls]

    logging.info("Inference complete. Output JSONLs:")
    for output_jsonl in output_jsonls:
        logging.info("  %s", output_jsonl)


if __name__ == "__main__":
    main()


# python eval/inference.py \
# --input-jsonl eval/data/same-dia-diff-context/final_dialogue.jsonl \
# --output-dir eval/data/mixed-style-runs/soul5/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --use-context 


# python eval/inference.py \
# --input-jsonl eval/data/custom_18/emo_emotion_name.jsonl \
# --output-dir eval/data/custom-18/soul5/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --use-context 



# python eval/inference.py \
# --input-jsonl eval/data/punct/manifest.jsonl \
# --output-dir eval/data/punct_runs/soul5/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --use-context 


# python eval/inference.py \
# --input-jsonl eval/data/punct/manifest.jsonl \
# --output-dir eval/data/punct_runs/soul5/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --use-context 


# python eval/inference.py \
# --input-jsonl eval/data/seedtts_testset/zh/meta_emodia.jsonl \
# --output-dir eval/data/seed-runs/soul5/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 




# python eval/inference.py \
# --input-jsonl eval/data/ttsd_instruct/ttsd_eval_zh_context.jsonl \
# --output-dir eval/data/ttsd_instruct_runs/ttsd_train5_no_instruct/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B-trained5 \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --no-skip-existing 



# python eval/inference.py \
# --input-jsonl eval/data/ttsd_instruct/ttsd_eval_zh_context.jsonl \
# --output-dir eval/data/ttsd_instruct_runs/ttsd_soul_pre_no_instruct/ \
# --model-path ../pretrained_models/SoulX-Podcast-1.7B \
# --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# --no-skip-existing 
