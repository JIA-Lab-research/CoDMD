"""
CoDMD Inference Script — DDP parallel multi-seed video generation.

Generates multiple videos per prompt with different seeds.
Output naming: {prompt}-{seed_index}.mp4

Usage (single GPU):
    python inference.py \
        --config_path configs/wan_dmd_tar.yaml \
        --checkpoint_folder <CHECKPOINT_DIR> \
        --output_folder <OUTPUT_DIR> \
        --prompt_file_path prompts.txt \
        --num_seeds 5

Usage (multi GPU with torchrun):
    torchrun --nproc_per_node=8 --master_port=29600 \
        inference.py \
        --config_path configs/wan_dmd_tar.yaml \
        --checkpoint_folder <CHECKPOINT_DIR> \
        --output_folder <OUTPUT_DIR> \
        --prompt_file_path prompts.txt \
        --num_seeds 5
""" 
import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from copula_dmd.models.wan.bidirectional_inference import BidirectionalInferencePipeline


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def load_prompts(prompt_file_path):
    """Load prompts from txt file, one per line."""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="DDP parallel inference for DMD-distilled WAN video model (multi-seed)"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--prompt_file_path", type=str, required=True,
                        help="Path to txt file with prompts for T2V generation (one per line)")
    parser.add_argument("--naming_prompt_file_path", type=str, default=None,
                        help="Path to txt file with original prompts for output naming (one per line). "
                             "Must have same number of lines as prompt_file_path. "
                             "If not provided, uses prompt_file_path for naming.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of different seeds per prompt (generates index 0 to num_seeds-1)")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (None for all).",
    )
    args = parser.parse_args()

    _init_logging()

    # Initialize DDP
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Load prompts for generation (before loading pipeline to check missing videos first)
    prompts = load_prompts(args.prompt_file_path)

    # Load naming prompts (original prompts for output filenames)
    if args.naming_prompt_file_path is not None:
        naming_prompts = load_prompts(args.naming_prompt_file_path)
        if args.max_prompts is not None:
            naming_prompts = naming_prompts[: args.max_prompts]
        assert len(naming_prompts) == len(prompts), (
            f"naming_prompt_file has {len(naming_prompts)} lines but "
            f"prompt_file has {len(prompts)} lines. They must match 1:1."
        )
    else:
        naming_prompts = prompts

    logging.info(f"Loaded {len(prompts)} prompts from {args.prompt_file_path}")
    if args.naming_prompt_file_path is not None:
        logging.info(f"Using naming prompts from {args.naming_prompt_file_path}")

    os.makedirs(args.output_folder, exist_ok=True)

    # Build all (prompt, naming_prompt, seed_index, save_path) tasks
    all_tasks = []
    for prompt_idx, (prompt, naming_prompt) in enumerate(zip(prompts, naming_prompts)):
        for seed_idx in range(args.num_seeds):
            prompt_name = naming_prompt  # Use naming_prompt directly without sanitization
            save_path = os.path.join(args.output_folder, f"{prompt_name}-{seed_idx}.mp4")
            all_tasks.append((prompt_idx, prompt, naming_prompt, seed_idx, save_path))

    # Check which videos already exist and filter out completed tasks
    missing_tasks = []
    for task in all_tasks:
        prompt_idx, prompt, naming_prompt, seed_idx, save_path = task
        if not os.path.exists(save_path):
            missing_tasks.append(task)

    logging.info(
        f"Total tasks: {len(all_tasks)}, "
        f"Already completed: {len(all_tasks) - len(missing_tasks)}, "
        f"Missing: {len(missing_tasks)}"
    )

    if missing_tasks:
        # Print missing prompts for debugging
        logging.info("=== Missing videos ===")
        missing_by_prompt = {}
        for prompt_idx, prompt, naming_prompt, seed_idx, save_path in missing_tasks:
            if naming_prompt not in missing_by_prompt:
                missing_by_prompt[naming_prompt] = []
            missing_by_prompt[naming_prompt].append(seed_idx)
        for naming_prompt, seed_indices in sorted(missing_by_prompt.items()):
            logging.info(f"  {naming_prompt[:60]}...: seeds {sorted(seed_indices)}")
    else:
        logging.info("All videos already exist, nothing to do.")

    # Round-robin distribute only missing tasks across GPUs
    my_tasks = [
        task for task_idx, task in enumerate(missing_tasks)
        if task_idx % world_size == rank
    ]

    logging.info(
        f"Rank {rank}: Processing {len(my_tasks)} missing tasks "
        f"(out of {len(missing_tasks)} total missing)"
    )
    
    if missing_tasks:
        # Load the pipeline only if there are missing videos globally
        logging.info("Loading pipeline...")
        config = OmegaConf.load(args.config_path)
        # Initialize on CPU first to avoid GPU memory spike during checkpoint load
        pipe = BidirectionalInferencePipeline(config, device="cpu")

        # Load trained generator weights on CPU
        checkpoint_path = os.path.join(args.checkpoint_folder, "model.pt")
        pipe.load_checkpoint(checkpoint_path)

        # Move to GPU once (no duplicate weights in GPU memory)
        pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()

    # All ranks must participate in barrier before destroying process group,
    # even if they have no tasks. Never sys.exit() before barrier.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    for prompt_idx, prompt, naming_prompt, seed_idx, save_path in my_tasks:
        actual_seed = args.seed + seed_idx
        logging.info(
            f"Rank {rank}: prompt={prompt_idx}, seed_idx={seed_idx}, "
            f"seed={actual_seed}: {prompt[:60]}..."
        )

        video = pipe.inference(
            noise=torch.randn(
                1, 21, 16, 60, 104,
                generator=torch.Generator(device="cuda").manual_seed(actual_seed),
                dtype=torch.bfloat16,
                device="cuda",
            ),
            text_prompts=[prompt],
        )[0].permute(0, 2, 3, 1).cpu().numpy()

        export_to_video(video, save_path, fps=16)
        logging.info(f"Rank {rank}: Saved {save_path}")

    logging.info(f"Rank {rank}: Finished.")


if __name__ == "__main__":
    main()
