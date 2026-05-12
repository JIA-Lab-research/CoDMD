from torch.utils.data import Dataset
import glob
import io
import os
import torch


class TextDataset(Dataset):
    """Simple dataset that reads one text prompt per line."""

    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def create_rcm_tar_dataloader(
    tar_dir: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
    world_size: int = 1,
    rank: int = 0,
):
    """Create a webdataset DataLoader for RCM pre-generated tar data.

    Each tar shard contains samples with:
        {idx:09d}.latent.pt  - VAE-encoded video latent [C, T, H, W]
        {idx:09d}.embed.pt   - T5 text embedding [seq_len, d_model]
        {idx:09d}.prompt.txt - raw text prompt

    Args:
        tar_dir: directory containing shard-*.tar files.
        batch_size: per-GPU batch size.
        num_workers: DataLoader workers.
        shuffle_buffer: number of samples to buffer for shuffling.
        world_size: total number of GPUs.
        rank: current GPU rank.

    Returns:
        An iterable DataLoader yielding dicts with keys:
            "prompt_embeds": [B, seq_len, d_model] bfloat16
            "prompts": list of str (length B)
    """
    import webdataset as wds

    tar_pattern = sorted(glob.glob(os.path.join(tar_dir, "shard-*.tar")))
    if not tar_pattern:
        # Try alternative naming
        tar_pattern = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    if not tar_pattern:
        raise FileNotFoundError(
            f"No tar files found in {tar_dir}")

    print(f"[RCM DataLoader] Found {len(tar_pattern)} tar shards in {tar_dir}")

    def decode_pt(data):
        return torch.load(io.BytesIO(data), map_location="cpu",
                          weights_only=False)
 
    def process_sample(sample):
        result = {
            "prompt_embeds": decode_pt(sample["embed.pt"]),
            "prompts": sample["prompt.txt"].decode("utf-8").strip(),
        }
        if "latent.pt" in sample:
            result["latent"] = decode_pt(sample["latent.pt"])
        return result

    dataset = (
        wds.WebDataset(tar_pattern, nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer)
        .map(process_sample)
        .batched(batch_size, partial=False, collation_fn=_collate_rcm)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return dataloader


def _collate_rcm(samples):
    """Collate function for RCM webdataset samples."""
    result = {
        "prompt_embeds": torch.stack([s["prompt_embeds"] for s in samples]),
        "prompts": [s["prompts"] for s in samples],
    }
    if "latent" in samples[0]:
        result["latent"] = torch.stack([s["latent"] for s in samples])
    return result
