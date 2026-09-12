import os
import glob
import json
import pandas as pd
import numpy as np

from scipy.io import wavfile
from scipy.signal import resample_poly
from faster_whisper import WhisperModel


def process_single_audio(file_path, model, turns_dir):

    try:
        sample_rate, audio_data = wavfile.read(file_path)

        # Canal 0 = caller
        caller_audio = (
            audio_data[:, 0]
            if audio_data.ndim > 1
            else audio_data
        )

        caller_audio = (
            caller_audio.astype(np.float32)
            / 32768.0
        )

        file_name = os.path.basename(file_path)

        anon_id = os.path.splitext(file_name)[0]

        turns_path = os.path.join(
            turns_dir,
            f"{anon_id}.json"
        )

        with open(turns_path, "r") as f:
            turns_data = json.load(f)

        caller_turns = [
            turn
            for turn in turns_data["turns"]
            if turn["channel"] == 0
        ]

        transcripts = []

        for turn in caller_turns:

            start_sample = int(
                turn["start"] * sample_rate
            )

            end_sample = int(
                turn["end"] * sample_rate
            )

            segment_audio = caller_audio[
                start_sample:end_sample
            ]

            # 8 kHz -> 16 kHz
            if sample_rate != 16000:

                segment_audio = resample_poly(
                    segment_audio,
                    16000,
                    sample_rate
                ).astype(np.float32)

            segments, _ = model.transcribe(
                segment_audio,
                language="es",
                beam_size=1
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            if text:
                transcripts.append(text)

        final_text = " ".join(transcripts)

        return {
            "file_name": file_name,
            "anon_id": anon_id,
            "caller_transcript": final_text
        }

    except Exception as e:

        print(
            f"ERROR en {file_path}: {e}"
        )

        return {
            "file_name": os.path.basename(file_path),
            "anon_id": os.path.splitext(
                os.path.basename(file_path)
            )[0],
            "caller_transcript": ""
        }


if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    audio_pattern = os.path.join(
        project_root,
        "audio",
        "*.wav"
    )

    turns_dir = os.path.join(
        project_root,
        "turns"
    )

    audio_files = glob.glob(audio_pattern)


    total = len(audio_files)

    print(
        f"Iniciando transcripción "
        f"para {total} llamadas..."
    )

    # Mejor calidad que tiny
    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"
    )

    results = []

    for index, file_path in enumerate(
        audio_files,
        start=1
    ):

        result = process_single_audio(
            file_path,
            model,
            turns_dir
        )

        results.append(result)

        status = (
            "OK"
            if result["caller_transcript"].strip()
            else "VACÍO"
        )

        print(
            f"[{index}/{total}] "
            f"{status}: "
            f"{result['file_name']}"
        )

    output_csv = os.path.join(
        project_root,
        "src",
        "language",
        "caller_transcriptions.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        "\nTranscripciones guardadas en:",
        output_csv
    )