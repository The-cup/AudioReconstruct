from audio_reconstruct.tasks.collect_and_preprocess import run_collect_and_preprocess
from audio_reconstruct.tasks.define_and_test_model import run_define_and_test_model
from audio_reconstruct.tasks.train_and_evaluate import run_train_and_evaluate


def run_all_tasks() -> None:
    """Run the full pipeline once each task has been implemented."""
    run_collect_and_preprocess()
    run_define_and_test_model()
    run_train_and_evaluate()

