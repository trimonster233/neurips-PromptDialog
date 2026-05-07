#!/usr/bin/env python3
"""Run SoulXPodcast inference directly on TTSD-eval JSONL files."""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import traceback
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


SPEAKER_TAG_RE = re.compile(r"\[S([1-4])\]")


def resolve_path(base_path: str, value: str) -> str:
    if not value:
        return ""
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(base_path)), value))


def parse_gpus(gpus: str) -> list[str]:
    if not gpus:
        return []
    parsed = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not parsed:
        raise ValueError("--gpus was provided but no GPU id was parsed")
    return parsed


def split_dialogue(text: str) -> list[str]:
    """Split a TTSD dialogue string into [Sx]utterance turns."""
    matches = list(SPEAKER_TAG_RE.finditer(text))
    if not matches:
        raise ValueError("dialogue text must contain [S1]/[S2] speaker tags")

    turns = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        utterance = text[start:end].strip()
        if utterance:
            turns.append(f"[S{match.group(1)}]{utterance}")
    if not turns:
        raise ValueError("dialogue text contains speaker tags but no utterances")
    return turns


def infer_case_id(index: int, record: dict) -> str:
    prompt_audio = record.get("prompt_audio_speaker1", "")
    basename = os.path.basename(prompt_audio)
    match = re.match(r"(.+?)_S1(?:\.[^.]+)?$", basename)
    return match.group(1) if match else f"case{index:04d}"


def load_cases(input_jsonl: str) -> list[dict]:
    cases = []
    with open(input_jsonl, "r", encoding="utf-8") as fin:
        for line_idx, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            sample_desc = f"{input_jsonl}:{line_idx}"
            text = str(record.get("text", ""))
            target_text_list = split_dialogue(text)

            prompt_audio_speaker1 = resolve_path(input_jsonl, record.get("prompt_audio_speaker1", ""))
            prompt_audio_speaker2 = resolve_path(input_jsonl, record.get("prompt_audio_speaker2", ""))
            for field_name, prompt_audio in (
                ("prompt_audio_speaker1", prompt_audio_speaker1),
                ("prompt_audio_speaker2", prompt_audio_speaker2),
            ):
                if not prompt_audio or not os.path.exists(prompt_audio):
                    raise FileNotFoundError(f"{sample_desc}: {field_name} not found: {prompt_audio}")

            prompt_text_speaker1 = str(record.get("prompt_text_speaker1", ""))
            prompt_text_speaker2 = str(record.get("prompt_text_speaker2", ""))
            if not prompt_text_speaker1 or not prompt_text_speaker2:
                raise ValueError(f"{sample_desc}: prompt_text_speaker1/prompt_text_speaker2 must not be empty")

            cases.append(
                {
                    "utt": infer_case_id(line_idx, record),
                    "text": text,
                    "target_text_list": target_text_list,
                    "prompt_wav_list": [prompt_audio_speaker1, prompt_audio_speaker2],
                    "prompt_text_list": [prompt_text_speaker1, prompt_text_speaker2],
                    "prompt_audio_speaker1": prompt_audio_speaker1,
                    "prompt_text_speaker1": prompt_text_speaker1,
                    "prompt_audio_speaker2": prompt_audio_speaker2,
                    "prompt_text_speaker2": prompt_text_speaker2,
                    "source_jsonl": os.path.abspath(input_jsonl),
                }
            )
    return cases


def resolve_subset_inputs(args) -> list[str]:
    if args.input_jsonl:
        return [os.path.abspath(path) for path in args.input_jsonl]

    data_dir = os.path.abspath(args.data_dir)
    subset_files = {
        "zh": os.path.join(data_dir, "ttsd_eval_zh.jsonl"),
        "en": os.path.join(data_dir, "ttsd_eval_en.jsonl"),
    }
    input_jsonls = [subset_files[args.subset]] if args.subset != "all" else list(subset_files.values())
    for path in input_jsonls:
        if not os.path.exists(path):
            raise FileNotFoundError(f"TTSD subset JSONL not found: {path}")
    return input_jsonls


def resolve_model_path(model_path: str) -> str:
    resolved = os.path.abspath(model_path)
    if os.path.isdir(resolved):
        return resolved

    duplicate = f"{os.sep}pretrained_models{os.sep}pretrained_models{os.sep}"
    if duplicate in resolved:
        candidate = resolved.replace(duplicate, f"{os.sep}pretrained_models{os.sep}", 1)
        if os.path.isdir(candidate):
            logging.warning(
                "Resolved duplicated pretrained_models path: %s -> %s",
                resolved,
                candidate,
            )
            return candidate

    raise FileNotFoundError(f"--model-path is not a directory: {resolved}")


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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=[
                "utt",
                "text",
                "output_audio",
                "prompt_audio_speaker1",
                "prompt_text_speaker1",
                "prompt_audio_speaker2",
                "prompt_text_speaker2",
                "source_jsonl",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def make_eval_record(case: dict, output_audio: Path) -> dict:
    return {
        "utt": case["utt"],
        "text": case["text"],
        "output_audio": str(output_audio.resolve()),
        "prompt_audio_speaker1": case["prompt_audio_speaker1"],
        "prompt_text_speaker1": case["prompt_text_speaker1"],
        "prompt_audio_speaker2": case["prompt_audio_speaker2"],
        "prompt_text_speaker2": case["prompt_text_speaker2"],
        "source_jsonl": case["source_jsonl"],
    }


def save_case_audio(model, dataset, case: dict, output_audio: Path, args) -> None:
    import torchaudio

    wav = synthesize_case(model, dataset, case, args)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_audio), wav.unsqueeze(0), args.sample_rate)


def get_subset_paths(input_jsonl: str, args) -> tuple[str, Path, Path, Path]:
    subset_name = Path(input_jsonl).stem
    subset_dir = Path(args.output_dir).resolve() / subset_name
    samples_dir = subset_dir / "inference" / "samples"
    eval_jsonl_path = subset_dir / f"{subset_name}_for_eval.jsonl"
    manifest_path = subset_dir / "inference" / "manifest.csv"
    return subset_name, samples_dir, eval_jsonl_path, manifest_path


def write_subset_outputs(eval_jsonl_path: Path, manifest_path: Path, records: list[dict]) -> str:
    write_jsonl(eval_jsonl_path, records)
    write_manifest(manifest_path, records)
    logging.info("Wrote %s", eval_jsonl_path)
    logging.info("Wrote %s", manifest_path)
    return str(eval_jsonl_path)


def run_subset(model, dataset, input_jsonl: str, args) -> str:
    subset_name, samples_dir, eval_jsonl_path, manifest_path = get_subset_paths(input_jsonl, args)
    samples_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(input_jsonl)
    logging.info("Loaded %d cases from %s", len(cases), input_jsonl)

    eval_records = []
    for index, case in enumerate(cases, start=1):
        output_audio = samples_dir / f"{case['utt']}.wav"
        logging.info("[%s %d/%d] synthesizing %s", subset_name, index, len(cases), case["utt"])

        if args.skip_existing and output_audio.exists():
            logging.info("Skipping existing output: %s", output_audio)
        else:
            save_case_audio(model, dataset, case, output_audio, args)

        if output_audio.exists():
            eval_records.append(make_eval_record(case, output_audio))

    return write_subset_outputs(eval_jsonl_path, manifest_path, eval_records)


def worker_loop(worker_id: int, gpu_id: str, task_queue, result_queue, args_dict: dict) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s worker{worker_id}/gpu{gpu_id} %(message)s")

    args = argparse.Namespace(**args_dict)
    try:
        from soulxpodcast.utils.infer_utils import initiate_model

        model, dataset = initiate_model(
            seed=args.seed + worker_id,
            model_path=args.model_path,
            llm_engine=args.llm_engine,
            fp16_flow=args.fp16_flow,
        )
        result_queue.put({"type": "ready", "worker_id": worker_id, "gpu_id": gpu_id})

        while True:
            task = task_queue.get()
            if task is None:
                result_queue.put({"type": "done", "worker_id": worker_id, "gpu_id": gpu_id})
                break

            case = task["case"]
            output_audio = Path(task["output_audio"])
            subset_name = task["subset_name"]
            order = task["order"]
            try:
                logging.info("[%s] synthesizing %s", subset_name, case["utt"])
                save_case_audio(model, dataset, case, output_audio, args)
                result_queue.put(
                    {
                        "type": "result",
                        "ok": True,
                        "subset_name": subset_name,
                        "order": order,
                        "record": make_eval_record(case, output_audio),
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result_queue.put(
                    {
                        "type": "result",
                        "ok": False,
                        "subset_name": subset_name,
                        "order": order,
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

    records_by_subset: dict[str, list[tuple[int, dict]]] = {}
    output_paths_by_subset: dict[str, tuple[Path, Path]] = {}
    task_count = 0

    for input_jsonl in input_jsonls:
        subset_name, samples_dir, eval_jsonl_path, manifest_path = get_subset_paths(input_jsonl, args)
        samples_dir.mkdir(parents=True, exist_ok=True)
        output_paths_by_subset[subset_name] = (eval_jsonl_path, manifest_path)
        records_by_subset[subset_name] = []

        cases = load_cases(input_jsonl)
        logging.info("Loaded %d cases from %s", len(cases), input_jsonl)
        for order, case in enumerate(cases):
            output_audio = samples_dir / f"{case['utt']}.wav"
            if args.skip_existing and output_audio.exists():
                logging.info("Skipping existing output: %s", output_audio)
                records_by_subset[subset_name].append((order, make_eval_record(case, output_audio)))
                continue

            task_queue.put(
                {
                    "subset_name": subset_name,
                    "order": order,
                    "case": case,
                    "output_audio": str(output_audio),
                }
            )
            task_count += 1

    if task_count == 0:
        logging.info("No queued tasks; writing manifests from existing outputs.")
        output_jsonls = []
        for subset_name, records_with_order in records_by_subset.items():
            eval_jsonl_path, manifest_path = output_paths_by_subset[subset_name]
            records = [record for _, record in sorted(records_with_order, key=lambda item: item[0])]
            output_jsonls.append(write_subset_outputs(eval_jsonl_path, manifest_path, records))
        return output_jsonls

    for _ in gpu_ids:
        task_queue.put(None)

    args_dict = vars(args).copy()
    processes = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        process = ctx.Process(
            target=worker_loop,
            args=(worker_id, gpu_id, task_queue, result_queue, args_dict),
        )
        process.start()
        processes.append(process)

    ready_workers = 0
    done_workers = 0
    finished_tasks = 0
    errors = []
    closed_workers = set()
    while done_workers < len(processes):
        try:
            message = result_queue.get(timeout=30)
        except queue.Empty:
            for worker_id, process in enumerate(processes):
                if worker_id in closed_workers:
                    continue
                if process.exitcode is not None:
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
                        logging.error(
                            "Worker %s on GPU %s exited with code %s",
                            worker_id,
                            gpu_ids[worker_id],
                            process.exitcode,
                        )
                    else:
                        logging.warning(
                            "Worker %s on GPU %s exited without sending a done message",
                            worker_id,
                            gpu_ids[worker_id],
                        )
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
                records_by_subset[message["subset_name"]].append((message["order"], message["record"]))
                logging.info(
                    "Finished %d/%d: %s on GPU %s",
                    finished_tasks,
                    task_count,
                    message["record"]["utt"],
                    message["gpu_id"],
                )
            else:
                errors.append(message)
                logging.error(
                    "Failed %d/%d: %s on GPU %s: %s",
                    finished_tasks,
                    task_count,
                    message["utt"],
                    message["gpu_id"],
                    message["error"],
                )

    for process in processes:
        process.join()

    output_jsonls = []
    for subset_name, records_with_order in records_by_subset.items():
        eval_jsonl_path, manifest_path = output_paths_by_subset[subset_name]
        records = [record for _, record in sorted(records_with_order, key=lambda item: item[0])]
        output_jsonls.append(write_subset_outputs(eval_jsonl_path, manifest_path, records))

    if errors:
        first = errors[0]
        raise RuntimeError(
            f"{len(errors)} task(s)/worker(s) failed. First error: {first.get('utt', 'worker')} {first.get('error')}\n"
            f"{first.get('traceback', '')}"
        )
    return output_jsonls


def get_args():
    parser = argparse.ArgumentParser(
        description="Directly synthesize eval/ttsd JSONL files and export TTSD-eval compatible JSONL outputs."
    )
    parser.add_argument("--model-path", required=True, help="SoulXPodcast model directory.")
    parser.add_argument("--data-dir", default=str(PACKAGE_DIR / "eval" / "ttsd"))
    parser.add_argument("--subset", choices=["zh", "en", "all"], default="zh")
    parser.add_argument("--input-jsonl", action="append", default=[], help="Explicit TTSD JSONL path.")
    parser.add_argument("--output-dir", default=str(PACKAGE_DIR / "eval" / "ttsd_runs" / "soulxpodcast"))
    parser.add_argument("--llm-engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. 0 or 0,1,2. Multi-GPU uses one worker per GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16-flow", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--silence-ms", type=float, default=200.0)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument(
        "--keep-full-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise longform context thresholds per case so middle text/speech history is not compacted by SoulXPodcast. Disabled by default.",
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
    args.model_path = resolve_model_path(args.model_path)
    input_jsonls = resolve_subset_inputs(args)
    gpu_ids = parse_gpus(args.gpus)

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

        output_jsonls = []
        for input_jsonl in input_jsonls:
            output_jsonls.append(run_subset(model, dataset, input_jsonl, args))

    logging.info("TTSD inference complete. Evaluation JSONLs:")
    for output_jsonl in output_jsonls:
        logging.info("  %s", output_jsonl)


if __name__ == "__main__":
    main()
#   python eval/ttsd_infer.py \
#     --model-path ../pretrained_models/SoulX-Podcast-1.7B \
#     --data-dir eval/ttsd \
#     --gpus 1,2,3 \
#     --output-dir eval/ttsd_runs/soulxpodcast
