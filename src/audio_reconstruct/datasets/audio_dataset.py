from torch.utils.data import Dataset


class AudioReconstructionDataset(Dataset):
    """Dataset skeleton for audio reconstruction tasks."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise NotImplementedError("Dataset loading is not implemented yet.")

