from data.load_data import load_raw_data
from data.preprocess import run_preprocessing_pipeline
from config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


def get_dataset(
        dataset_name="LibriSpeech",
        dataset_sub_name="train-clean-100",
        base_dir=RAW_DATA_DIR
):
    return load_raw_data(
        dataset_name=dataset_name,
        dataset_sub_name=dataset_sub_name,
        base_dir=base_dir,
    )

def preprocess_dataset(
        dataset,
        save_dir=PROCESSED_DATA_DIR,
        load_data=False
):
    return run_preprocessing_pipeline(
        dataset=dataset,
        save_dir=save_dir,
        load_data=load_data
    )


if __name__ == "__main__":
    dataset = get_dataset(dataset_name="LibriSpeech", dataset_sub_name="train-clean-360", base_dir=RAW_DATA_DIR)
    preprocess_dataset(dataset, save_dir=PROCESSED_DATA_DIR, load_data=False)
