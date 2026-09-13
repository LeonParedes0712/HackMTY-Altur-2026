import json
import numpy as np
import csv

def get_call_duration(anon_id):
    with open("manifest.csv", "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["anon_id"] == anon_id:
                return float(row["duration_s"])

    return None




def load_turns(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data["turns"]


def get_caller_turns(turns):
    caller_turns = []

    for turn in turns:
        if turn["channel"] == 0:
            caller_turns.append(turn)

    return caller_turns


def get_turn_durations(caller_turns):
    durations = []

    for turn in caller_turns:
        duration = turn["end"] - turn["start"]
        durations.append(duration)

    return durations

def get_caller_speaking_ratio(durations, call_duration):
    caller_speaking_time = sum(durations)

    ratio = caller_speaking_time / call_duration

    return ratio


def get_response_times(turns):
    response_times = []

    for i in range(1, len(turns)):
        previous_turn = turns[i - 1]
        current_turn = turns[i]

        if previous_turn["channel"] == 1 and current_turn["channel"] == 0:
            response_time = current_turn["start"] - previous_turn["end"]
            response_times.append(response_time)

    return response_times

def extract_features(anon_id):
    path = f"turns/{anon_id}.json"

    turns = load_turns(path)
    caller_turns = get_caller_turns(turns)

    durations = get_turn_durations(caller_turns)
    response_times = get_response_times(turns)

    call_duration = get_call_duration(anon_id)

    features = {
        "anon_id": anon_id,
        "caller_turn_count": len(caller_turns),
        "caller_turn_duration_mean": np.mean(durations),
        "caller_turn_duration_std": np.std(durations),
        "response_time_mean": np.mean(response_times),
        "response_time_std": np.std(response_times),
        "caller_speaking_ratio": get_caller_speaking_ratio(
            durations,
            call_duration
        )
    }

    return features

def main():

    results = []

    with open("manifest.csv", "r") as f:
        reader = csv.DictReader(f)

        for row in reader:

            anon_id = row["anon_id"]

            features = extract_features(anon_id)

            # También guardamos lo que Altur ya nos dio
            features["label"] = row["label"]
            features["split"] = row["split"]

            results.append(features)

    output_path = "outputs/behavior_features.csv"

    fieldnames = [
        "anon_id",
        "caller_turn_count",
        "caller_turn_duration_mean",
        "caller_turn_duration_std",
        "response_time_mean",
        "response_time_std",
        "caller_speaking_ratio",
        "label",
        "split"
    ]

    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"Features generadas para {len(results)} llamadas."
    )

    print(
        f"Archivo guardado en: {output_path}"
    )


if __name__ == "__main__":
    main()