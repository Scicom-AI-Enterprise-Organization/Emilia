# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Emilia processing pipeline (the heavy, GPU-bound half).
#
# Everything CUDA-touching lives here so `main.py` (the orchestrator) can stay
# import-clean and fork worker processes safely: a worker sets
# CUDA_VISIBLE_DEVICES, THEN imports this module and calls `init_models()`.
# Importing torch / onnxruntime before the fork would poison the CUDA context in
# the children, so nothing in main.py may import this module at top level.

import json
import os
import re
import gc

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm
from pydub import AudioSegment
from pyannote.audio import Pipeline

from utils.tool import (
    export_to_mp3,
    calculate_audio_stats,
)
from utils.logger import Logger, time_logger
from models import separate_fast, dnsmos, whisper_asr, silero_vad

# ---------------------------------------------------------------------------
# Module-global model handles / config, populated by init_models() inside each
# worker process. They are intentionally module-level so the pipeline functions
# below read like the upstream Amphion code.
# ---------------------------------------------------------------------------
cfg = None
logger = None
device = None
device_name = None
batch_size = 16

dia_pipeline = None
asr_model = None
vad = None
separate_predictor1 = None
dnsmos_compute_score = None

supported_languages = None
multilingual_flag = None
force_language = None  # if set (e.g. "yue"), skip detection & transcribe every segment in it

audio_count = 0


def init_models(config, whisper_arch, compute_type, threads, bs):
    """Load every model once per worker process. Must be called AFTER
    CUDA_VISIBLE_DEVICES has been pinned for this worker."""
    global cfg, logger, device, device_name, batch_size
    global dia_pipeline, asr_model, vad, separate_predictor1, dnsmos_compute_score
    global supported_languages, multilingual_flag, force_language

    cfg = config
    batch_size = bs
    logger = Logger.get_logger()

    from utils.tool import detect_gpu, check_env

    if detect_gpu():
        logger.info("Using GPU")
        device_name = "cuda"
    else:
        logger.info("Using CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    check_env(logger)

    hf_token = cfg["huggingface_token"]
    if not hf_token.startswith("hf"):
        raise ValueError(
            "huggingface_token must start with 'hf'. Set HF_TOKEN in the env or "
            "huggingface_token in the config. Grant access to "
            "pyannote/speaker-diarization-3.1 first."
        )

    logger.debug(" * Loading Speaker Diarization Model")
    dia_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    dia_pipeline.to(device)

    logger.debug(" * Loading ASR Model")
    asr_model = whisper_asr.load_asr_model(
        whisper_arch,
        device_name,
        compute_type=compute_type,
        threads=threads,
        asr_options={
            "initial_prompt": "Um, Uh, Ah. Like, you know. I mean, right. Actually. Basically, and right? okay. Alright. Emm. So. Oh. 生于忧患,死于安乐。岂不快哉?当然,嗯,呃,就,这样,那个,哪个,啊,呀,哎呀,哎哟,唉哇,啧,唷,哟,噫!微斯人,吾谁与归?ええと、あの、ま、そう、ええ。äh, hm, so, tja, halt, eigentlich. euh, quoi, bah, ben, tu vois, tu sais, t'sais, eh bien, du coup. genre, comme, style. 응,어,그,음."
        },
    )

    logger.debug(" * Loading VAD Model")
    vad = silero_vad.SileroVAD(device=device)

    logger.debug(" * Loading Background Noise Model")
    separate_predictor1 = separate_fast.Predictor(
        args=cfg["separate"]["step1"], device=device_name
    )

    logger.debug(" * Loading DNSMOS Model")
    dnsmos_compute_score = dnsmos.ComputeScore(
        cfg["mos_model"]["primary_model_path"], device_name
    )
    logger.debug("All models loaded")

    supported_languages = cfg["language"]["supported"]
    multilingual_flag = cfg["language"]["multilingual"]
    force_language = cfg["language"].get("force") or None
    logger.debug(f"supported languages {supported_languages}")
    logger.debug(f"using multilingual asr {multilingual_flag}")
    if force_language:
        logger.debug(f"FORCING ASR language = {force_language} (detection bypassed)")


def get_length(file):
    """Robust duration in seconds.

    The upstream code used ``soundfile.read`` here, which silently fails on MP3
    when libsndfile lacks MPEG support (< 1.1.0) — that returns 0 and the whole
    clip gets skipped. Fall back through librosa and pydub/ffmpeg so MP3 input
    (this dataset) works regardless of the libsndfile build.
    """
    try:
        info = sf.info(file)
        return info.frames / info.samplerate
    except Exception:
        pass
    try:
        return librosa.get_duration(path=file)
    except Exception:
        pass
    try:
        seg = AudioSegment.from_file(file)
        return len(seg) / 1000.0
    except Exception as e:
        print("get_length failed:", e)
        return 0


@time_logger
def standardization(audio):
    """Preprocess: fixed sample rate, 16-bit, mono, loudness-normalized."""
    global audio_count
    name = "audio"

    duration = 0
    try:
        duration = get_length(audio)
    except Exception as e:
        print(e)

    if duration == 0 or (duration / 60 / 60) >= 5:
        return None

    if isinstance(audio, str):
        name = os.path.basename(audio)
        audio = AudioSegment.from_file(audio)
    elif isinstance(audio, AudioSegment):
        name = f"audio_{audio_count}"
        audio_count += 1
    else:
        raise ValueError("Invalid audio type")

    logger.debug("Entering the preprocessing of audio")

    audio = audio.set_frame_rate(cfg["entrypoint"]["SAMPLE_RATE"])
    audio = audio.set_sample_width(2)  # 16-bit
    audio = audio.set_channels(1)  # mono

    logger.debug("Audio file converted to WAV format")

    target_dBFS = -20
    gain = target_dBFS - audio.dBFS
    logger.info(f"Calculating the gain needed for the audio: {gain} dB")

    normalized_audio = audio.apply_gain(min(max(gain, -3), 3))

    waveform = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
    max_amplitude = np.max(np.abs(waveform))
    waveform /= max_amplitude

    logger.debug(f"waveform shape: {waveform.shape}")

    return {
        "waveform": waveform,
        "name": name,
        "sample_rate": cfg["entrypoint"]["SAMPLE_RATE"],
    }


@time_logger
def source_separation(predictor, audio):
    """Separate vocals from the rest using the UVR predictor."""
    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        rate = audio["sample_rate"]
        mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)

    vocals, no_vocals = predictor.predict(mix)

    vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    audio["waveform"] = vocals[:, 0]  # vocals is stereo, keep one channel

    return audio


@time_logger
def speaker_diarization(audio):
    """Speaker diarization -> dataframe of (segment, label, speaker, start, end)."""
    logger.debug(f"Start speaker diarization")
    logger.debug(f"audio waveform shape: {audio['waveform'].shape}")

    waveform = torch.tensor(audio["waveform"]).to(device)
    waveform = torch.unsqueeze(waveform, 0)

    segments = dia_pipeline(
        {
            "waveform": waveform,
            "sample_rate": audio["sample_rate"],
            "channel": 0,
        }
    )

    diarize_df = pd.DataFrame(
        segments.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
    diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)

    return diarize_df


@time_logger
def cut_by_speaker_label(vad_list):
    """Merge/trim VAD segments by speaker label, enforcing length/gap constraints."""
    MERGE_GAP = 2  # seconds
    MIN_SEGMENT_LENGTH = 3  # seconds
    MAX_SEGMENT_LENGTH = 30  # seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None
        last_speaker = updated_list[-1]["speaker"] if updated_list else None

        if vad["end"] - vad["start"] >= MAX_SEGMENT_LENGTH:
            current_start = vad["start"]
            segment_end = vad["end"]
            logger.warning(
                f"cut_by_speaker_label > segment longer than 30s, force trimming to 30s smaller segments"
            )
            while segment_end - current_start >= MAX_SEGMENT_LENGTH:
                vad["end"] = current_start + MAX_SEGMENT_LENGTH
                updated_list.append(vad)
                vad = vad.copy()
                current_start += MAX_SEGMENT_LENGTH
                vad["start"] = current_start
                vad["end"] = segment_end
            updated_list.append(vad)
            continue

        if (
            last_speaker is None
            or last_speaker != vad["speaker"]
            or vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
            continue

        if (
            vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]

    logger.debug(
        f"cut_by_speaker_label > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    logger.debug(
        f"cut_by_speaker_label > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list


@time_logger
def asr(vad_segments, audio):
    """Batched ASR over VAD segments."""
    if len(vad_segments) == 0:
        return []

    temp_audio = audio["waveform"]
    start_time = vad_segments[0]["start"]
    end_time = vad_segments[-1]["end"]
    start_frame = int(start_time * audio["sample_rate"])
    end_frame = int(end_time * audio["sample_rate"])
    temp_audio = temp_audio[start_frame:end_frame]  # trim silent head/tail

    for idx, segment in enumerate(vad_segments):
        vad_segments[idx]["start"] -= start_time
        vad_segments[idx]["end"] -= start_time

    temp_audio = librosa.resample(
        temp_audio, orig_sr=audio["sample_rate"], target_sr=16000
    )

    # Known single-language dataset: skip per-segment detection (which mislabels
    # Cantonese as "zh" and yields Mandarin-normalized text) and transcribe every
    # segment in the forced language. Nothing is dropped.
    if force_language:
        transcribe_result = asr_model.transcribe(
            temp_audio,
            vad_segments,
            batch_size=batch_size,
            language=force_language,
            print_progress=True,
        )
        result = transcribe_result["segments"]
        for idx, segment in enumerate(result):
            result[idx]["start"] += start_time
            result[idx]["end"] += start_time
            result[idx]["language"] = force_language
        return result

    if multilingual_flag:
        logger.debug("Multilingual flag is on")
        valid_vad_segments, valid_vad_segments_language = [], []
        for idx, segment in enumerate(vad_segments):
            start_frame = int(segment["start"] * 16000)
            end_frame = int(segment["end"] * 16000)
            segment_audio = temp_audio[start_frame:end_frame]
            language, prob = asr_model.detect_language(segment_audio)
            if language in supported_languages and prob > 0.5:
                valid_vad_segments.append(vad_segments[idx])
                valid_vad_segments_language.append(language)

        if len(valid_vad_segments) == 0:
            return []
        all_transcribe_result = []
        unique_languages = list(set(valid_vad_segments_language))
        for language_token in unique_languages:
            language = language_token
            vad_segments = [
                valid_vad_segments[i]
                for i, x in enumerate(valid_vad_segments_language)
                if x == language
            ]
            transcribe_result_temp = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result_temp["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result_temp["language"]
            all_transcribe_result.extend(result)
        all_transcribe_result = sorted(all_transcribe_result, key=lambda x: x["start"])
        return all_transcribe_result
    else:
        logger.debug("Multilingual flag is off")
        language, prob = asr_model.detect_language(temp_audio)
        if language in supported_languages and prob > 0.8:
            transcribe_result = asr_model.transcribe(
                temp_audio,
                vad_segments,
                batch_size=batch_size,
                language=language,
                print_progress=True,
            )
            result = transcribe_result["segments"]
            for idx, segment in enumerate(result):
                result[idx]["start"] += start_time
                result[idx]["end"] += start_time
                result[idx]["language"] = transcribe_result["language"]
            return result
        else:
            return []


@time_logger
def mos_prediction(audio, vad_list):
    """DNSMOS scoring per segment; returns (avg_mos, vad_list-with-dnsmos)."""
    audio = audio["waveform"]
    sample_rate = 16000

    audio = librosa.resample(
        audio, orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], target_sr=sample_rate
    )

    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio[start:end]

        dnsmos_score = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos_score

    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    logger.debug(f"avg predict_dnsmos for whole audio: {predict_dnsmos}")

    return predict_dnsmos, vad_list


def filter_by_mos(mos_list):
    """Keep segments passing the MOS / char-duration / duration filters."""
    filtered_audio_stats, all_audio_stats = calculate_audio_stats(mos_list)
    filtered_segment = len(filtered_audio_stats)
    all_segment = len(all_audio_stats)
    logger.debug(
        f"> {all_segment - filtered_segment}/{all_segment} "
        f"{(all_segment - filtered_segment) / all_segment:.2%} segments filtered."
    )
    filtered_list = [mos_list[idx] for idx, _ in filtered_audio_stats]
    return filtered_list


def main_process(audio_path, save_path=None, audio_name=None):
    """Full Emilia pipeline for one input audio file.

    Writes ``<save_path>/<audio_name>.json`` (the manifest) plus one MP3 per kept
    segment. The presence of a *valid* JSON at that path is the resume marker.
    """
    if not audio_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
        logger.warning(f"Unsupported file type: {audio_path}")

    audio_name = audio_name or os.path.splitext(os.path.basename(audio_path))[0]
    save_path = save_path or os.path.join(
        os.path.dirname(audio_path) + "_processed", audio_name
    )
    final_path = os.path.join(save_path, audio_name + ".json")
    if os.path.exists(final_path):
        return final_path
    os.makedirs(save_path, exist_ok=True)
    logger.debug(f"Processing audio: {audio_name}, from {audio_path}, save to: {save_path}")

    logger.info("Step 0: Preprocess (resample + mono + loudnorm + 16-bit)")
    audio = standardization(audio_path)
    if audio is None:
        logger.warning(f"skip {audio_path}, unreadable or too long")
        with open(final_path, "w") as fopen:
            json.dump([], fopen)
        return final_path

    logger.info("Step 1: Source Separation")
    audio = source_separation(separate_predictor1, audio)

    logger.info("Step 2: Speaker Diarization")
    speakerdia = speaker_diarization(audio)

    logger.info("Step 3: Fine-grained Segmentation by VAD")
    vad_list = vad.vad(speakerdia, audio)
    segment_list = cut_by_speaker_label(vad_list)

    logger.info("Step 4: ASR")
    asr_result = asr(segment_list, audio)

    logger.info("Step 5.1: MOS prediction")
    if len(asr_result) == 0:
        with open(final_path, "w") as fopen:
            json.dump([], fopen)
        return final_path
    avg_mos, mos_list = mos_prediction(audio, asr_result)
    logger.info(f"Step 5.1: done, average MOS: {avg_mos}")

    logger.info("Step 5.2: Filter by MOS / duration / char-rate")
    try:
        filtered_list = filter_by_mos(mos_list)
    except Exception as e:
        logger.warning(f"filter failed for {audio_name}: {e}")
        with open(final_path, "w") as fopen:
            json.dump([], fopen)
        return final_path

    logger.info("Step 6: write segments to MP3 + JSON")
    export_to_mp3(audio, filtered_list, save_path, audio_name)

    with open(final_path, "w") as f:
        json.dump(filtered_list, f, ensure_ascii=False)

    logger.info(f"All done, saved to: {final_path}")
    return final_path
