from pathlib import Path

from src.processor.process_video import process_video
import src.analyser.evaluate as evaluate

import src.tracker.tracker_constants as tracker_constants


def process(input_file: Path, tracker_type: str):
    result = process_video(
        input_file=input_file, tracker_type=tracker_type, should_show_live_output=True
    )
    print(result)


def evaluate_video(input_file: Path, tracker_type: str):
    result = evaluate.evaluate_tracker(input_file=input_file, tracker_type=tracker_type)
    print(result)


def evaluate_all(input_file: Path):
    results = evaluate.evaluate_all_trackers(
        input_file=input_file, trackers_list=tracker_constants.TRACKERS_ALL
    )
    print(results)


if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent
    input_file = ROOT / "input" / "input1.mp4"
    evaluate_all(input_file=input_file)
