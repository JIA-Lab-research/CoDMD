"""
CoDMD Training Script — Copula-Aware Distribution Matching Distillation.

Supports two data modes controlled by config:
  - use_tar_data: true  → pre-computed T5 embeddings from tar shards + backward simulation
  - use_tar_data: false → TextDataset (txt prompt file) + backward simulation

Usage:
    torchrun --nnodes 4 --nproc_per_node=8 --rdzv_id=5235 \
        copula_dmd/train_dmd.py -- \
        --config_path configs/wan_dmd_tar.yaml
"""
from copula_dmd.data import create_rcm_tar_dataloader, TextDataset
from copula_dmd.models import get_block_class
from copula_dmd.util import (
    launch_distributed_job,
    set_seed,
    fsdp_wrap,
    cycle,
    fsdp_state_dict,
    barrier,
)
import torch.distributed as dist
from omegaconf import OmegaConf
from copula_dmd.dmd import DMD
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusers.utils import export_to_video
import time
import os
from datetime import datetime


class CoDMDTrainer:
    """CoDMD Trainer with copula-aware distillation loss."""

    def __init__(self, config):
        self.config = config

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            total_gpus = dist.get_world_size()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            model_path = getattr(config, "model_path", None)
            model_version = "14b" if (model_path and "14B" in model_path) else "1.3b"
            num_steps = len(config.denoising_step_list)
            global_bs = getattr(config, "batch_size", 1) * total_gpus

            exp_name = f"codmd_{model_version}_nfe{num_steps}_bs{global_bs}_gpu{total_gpus}_{timestamp}"
            self.output_path = os.path.join(
                getattr(config, "output_path", "./outputs"), exp_name)
            os.makedirs(self.output_path, exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.output_path)
            print(f"TensorBoard log dir: {self.output_path}")

            config_src = getattr(config, "config_path", None)
            if config_src and os.path.isfile(config_src):
                import shutil
                shutil.copy2(config_src, os.path.join(
                    self.output_path, os.path.basename(config_src)))

        # Initialize model
        self.distillation_model = DMD(config, device=self.device)

        # FSDP wrap all sub-models
        self.distillation_model.generator = fsdp_wrap(
            self.distillation_model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy)

        self.distillation_model.real_score = fsdp_wrap(
            self.distillation_model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy)

        self.distillation_model.fake_score = fsdp_wrap(
            self.distillation_model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy)

        self.distillation_model.text_encoder = fsdp_wrap(
            self.distillation_model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy)

        if not config.no_visualize:
            self.distillation_model.vae = self.distillation_model.vae.to(
                device=self.device, dtype=self.dtype)

        # Optimizer — generator
        generator_params = [
            p for p in self.distillation_model.generator.parameters()
            if p.requires_grad]
        self.generator_optimizer = torch.optim.AdamW(
            generator_params, lr=config.lr,
            betas=(config.beta1, config.beta2))

        # Optimizer — critic
        critic_params = [
            p for p in self.distillation_model.fake_score.parameters()
            if p.requires_grad]
        self.critic_optimizer = torch.optim.AdamW(
            critic_params, lr=config.lr / config.dfake_gen_update_ratio,
            betas=(config.beta1, config.beta2))

        # Dataloader
        self.use_tar_data = getattr(config, "use_tar_data", False)
        if self.use_tar_data:
            dataloader = create_rcm_tar_dataloader(
                tar_dir=config.tar_data_dir,
                batch_size=config.batch_size,
                num_workers=getattr(config, "num_workers", 4),
                shuffle_buffer=getattr(config, "shuffle_buffer", 1000),
                world_size=dist.get_world_size(),
                rank=dist.get_rank())
        else:
            dataset = TextDataset(config.data_path)
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.batch_size, sampler=sampler)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        state_dict = {
            "generator": fsdp_state_dict(self.distillation_model.generator),
            "critic": fsdp_state_dict(self.distillation_model.fake_score),
        }

        if self.is_main_process:
            try:
                save_dir = os.path.join(
                    self.output_path, f"checkpoint_model_{self.step:06d}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "model.pt")
                torch.save(state_dict, save_path)
                print(f"Model saved to {save_path}")
            except (OSError, IOError, RuntimeError) as save_error:
                print(f"[ERROR] Failed to save checkpoint: {save_error}")

    def save_video(self, latent, prefix, step):
        if self.config.no_visualize:
            return
        try:
            video_vis_dir = os.path.join(self.output_path, "video_vis")
            os.makedirs(video_vis_dir, exist_ok=True)
            with torch.no_grad():
                vae_dtype = next(self.distillation_model.vae.parameters()).dtype
                decoded = self.distillation_model.vae.decode_to_pixel(
                    latent.to(vae_dtype))
                decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
                video_np = decoded[0].cpu().permute(0, 2, 3, 1).numpy()
                video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
                export_to_video(
                    video_np,
                    os.path.join(video_vis_dir, f"{prefix}_{step}.mp4"),
                    fps=16)
        except (OSError, IOError, RuntimeError) as video_error:
            print(f"[ERROR] Failed to save video: {video_error}")

    def save_inference_video(self, conditional_dict, step):
        """Generate video using full multi-step inference and save.

        All ranks must participate in the forward pass (FSDP allgather).
        Only rank 0 saves the video.
        """
        if self.config.no_visualize:
            return

        with torch.no_grad():
            self.distillation_model.eval()
            generator = self.distillation_model.generator
            shape = list(self.config.image_or_video_shape)

            noisy_video = torch.randn(
                1, shape[1], shape[2], shape[3], shape[4],
                device=self.device, dtype=self.dtype,
                generator=torch.Generator(device=self.device).manual_seed(42))

            denoising_step_list = self.distillation_model.denoising_step_list
            scheduler = self.distillation_model.scheduler

            test_prompt = ("A stylish woman walks down a Tokyo street filled "
                           "with warm glowing neon and animated city signage.")
            inference_cond = self.distillation_model.text_encoder(
                text_prompts=[test_prompt])

            for index, current_timestep in enumerate(denoising_step_list):
                timestep = torch.ones(
                    (1, shape[1]), dtype=torch.long,
                    device=self.device) * current_timestep
                pred = generator(
                    noisy_image_or_video=noisy_video,
                    conditional_dict=inference_cond,
                    timestep=timestep)
                if index < len(denoising_step_list) - 1:
                    next_t = denoising_step_list[index + 1] * torch.ones(
                        (1, shape[1]), dtype=torch.long, device=self.device)
                    noisy_video = scheduler.add_noise(
                        pred.flatten(0, 1),
                        torch.randn_like(pred.flatten(0, 1)),
                        next_t.flatten(0, 1)).unflatten(0, (1, shape[1]))

            if self.is_main_process:
                self.save_video(pred, "infer", step)

    def train_one_step(self):
        self.distillation_model.eval()

        train_generator = (self.step % self.config.dfake_gen_update_ratio == 0)
        visualize_every = getattr(self.config, "log_iters", 200) // 4
        should_visualize = (
            self.step % visualize_every == 0 and not self.config.no_visualize)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Get the next batch of data
        if self.use_tar_data:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size
            clean_latent = None
        else:
            text_prompts = next(self.dataloader)
            clean_latent = None
            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size

        # Extract conditional embeddings
        with torch.no_grad():
            if self.use_tar_data:
                conditional_dict = {
                    "prompt_embeds": batch["prompt_embeds"].to(
                        device=self.device, dtype=self.dtype)}
            else:
                conditional_dict = self.distillation_model.text_encoder(
                    text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                self.unconditional_dict = {
                    k: v.detach() for k, v in unconditional_dict.items()}
            unconditional_dict = self.unconditional_dict

        # Train the generator
        if train_generator:
            generator_loss, generator_log_dict = (
                self.distillation_model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent))

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_grad_norm = (
                self.distillation_model.generator.clip_grad_norm_(
                    self.max_grad_norm))
            self.generator_optimizer.step()
        else:
            generator_log_dict = {}

        # Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = (
            self.distillation_model.fake_score.clip_grad_norm_(
                self.max_grad_norm))
        self.critic_optimizer.step()

        # Logging
        if self.is_main_process:
            self.writer.add_scalar(
                "loss/critic_loss", critic_loss.item(), self.step)
            self.writer.add_scalar(
                "norm/critic_grad_norm", critic_grad_norm.item(), self.step)

            if train_generator:
                self.writer.add_scalar(
                    "loss/generator_loss", generator_loss.item(), self.step)
                self.writer.add_scalar(
                    "norm/generator_grad_norm",
                    generator_grad_norm.item(), self.step)
                self.writer.add_scalar(
                    "norm/dmdtrain_gradient_norm",
                    generator_log_dict["dmdtrain_gradient_norm"].item(),
                    self.step)

                for key in ["copula_loss", "copula_batch_loss",
                             "copula_frame_loss"]:
                    if key in generator_log_dict:
                        self.writer.add_scalar(
                            f"loss/{key}",
                            generator_log_dict[key].item(), self.step)

        if should_visualize:
            self.save_inference_video(conditional_dict, self.step)

    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    self.writer.add_scalar(
                        "per_iteration_time",
                        current_time - self.previous_time, self.step)
                    self.previous_time = current_time

            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.config_path = args.config_path
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = CoDMDTrainer(config)
    trainer.train()

    if trainer.is_main_process:
        trainer.writer.close()


if __name__ == "__main__":
    main()
