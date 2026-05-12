"""
Copula-Aware Distillation Loss for CoDMD training.

Matches the relational structure (similarity matrix) between student generator
and teacher/critic score predictions via KL divergence on softmax similarity
matrices. This preserves the joint dependency (copula) structure across frames
and across batch samples during distillation.

Two granularities:
  - Batch-level: cross-GPU all_gather of pooled features, S: [B_global, B_global]
  - Frame-level: per-sample temporal structure, S: [F, F], local only

Usage:
    from copula_dmd.copula_loss import compute_copula_aware_loss
    loss, log = compute_copula_aware_loss(
        student_latent, real_score_latent, fake_score_latent,
        temperature=0.1, batch_weight=1.0, frame_weight=1.0,
    )
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

class AllGatherWithGrad(torch.autograd.Function):
    """Differentiable all_gather: gathers tensors from all GPUs and
    routes gradients back to each GPU's local chunk."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        ctx.world_size = world_size
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank()
        return grad_output.chunk(ctx.world_size, dim=0)[rank].contiguous()


@torch.no_grad()
def _all_gather_no_grad(tensor):
    """all_gather without gradient tracking (for teacher/critic features)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [tensor]
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


def _gather_with_grad(feat):
    """Gather features across GPUs with gradient support."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        return AllGatherWithGrad.apply(feat)
    return feat


def _gather_without_grad(feat):
    """Gather features across GPUs without gradient."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        return torch.cat(_all_gather_no_grad(feat), dim=0)
    return feat


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(features):
    """Compute cosine similarity matrix [N, N] from feature vectors [N, D]."""
    normed = F.normalize(features.float(), dim=-1)
    return normed @ normed.T


def kl_divergence_loss(target_prob, student_prob):
    """Row-wise KL(target_prob || student_prob)."""
    return F.kl_div(
        student_prob.log().float(),
        target_prob.float(),
        reduction='batchmean',
    )


def _pool_to_batch_features(latent):
    """Pool [B, F, C, H, W] -> [B, C] by averaging over F, H, W."""
    return latent.float().mean(dim=[1, 3, 4])


def _pool_to_frame_features(latent_single):
    """Pool [F, C, H, W] -> [F, C] by averaging over H, W."""
    return latent_single.float().mean(dim=[-2, -1])


def _zero_loss(device, dtype=torch.float32):
    return torch.tensor(0.0, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Copula-aware loss: batch-level and frame-level
# ---------------------------------------------------------------------------

def _batch_level_copula_loss(student_latent, real_score_latent,
                             fake_score_latent, temperature):
    """Batch-level copula-aware loss with cross-GPU all_gather.

    Constructs a target similarity distribution from the difference between
    the teacher (real_score) and critic (fake_score) cosine similarity matrices,
    then minimizes KL divergence between this target and the student's
    similarity distribution.
    """
    student_feat = _gather_with_grad(_pool_to_batch_features(student_latent))
    real_feat = _gather_without_grad(_pool_to_batch_features(real_score_latent))
    fake_feat = _gather_without_grad(_pool_to_batch_features(fake_score_latent))

    if student_feat.shape[0] < 2:
        return _zero_loss(student_latent.device, student_latent.dtype)

    real_cosine = cosine_similarity_matrix(real_feat)
    fake_cosine = cosine_similarity_matrix(fake_feat)
    student_cosine = cosine_similarity_matrix(student_feat)

    target_prob = F.softmax(
        (student_cosine - (fake_cosine - real_cosine)) / temperature, dim=-1,
    ).detach()
    student_prob = F.softmax(student_cosine / temperature, dim=-1)

    return kl_divergence_loss(target_prob, student_prob)


def _frame_level_copula_loss(student_latent, real_score_latent,
                             fake_score_latent, temperature):
    """Frame-level copula-aware loss (local, zero inter-GPU communication).

    Operates on per-sample temporal similarity matrices [F, F].
    """
    batch_size, num_frames = student_latent.shape[:2]
    if num_frames < 2:
        return _zero_loss(student_latent.device, student_latent.dtype)

    total_loss = _zero_loss(student_latent.device)
    for i in range(batch_size):
        student_feat = _pool_to_frame_features(student_latent[i])
        real_feat = _pool_to_frame_features(real_score_latent[i])
        fake_feat = _pool_to_frame_features(fake_score_latent[i])

        real_cosine = cosine_similarity_matrix(real_feat)
        fake_cosine = cosine_similarity_matrix(fake_feat)
        student_cosine = cosine_similarity_matrix(student_feat)

        target_prob = F.softmax(
            (student_cosine - (fake_cosine - real_cosine)) / temperature,
            dim=-1,
        ).detach()
        student_prob = F.softmax(student_cosine / temperature, dim=-1)

        total_loss = total_loss + kl_divergence_loss(target_prob, student_prob)

    return total_loss / batch_size


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_copula_aware_loss(
    student_latent, real_score_latent, fake_score_latent,
    temperature=0.1, batch_weight=1.0, frame_weight=1.0,
):
    """Compute copula-aware distillation loss.

    Matches the relational structure (copula) of the student generator's
    predictions to the joint structure implied by teacher and critic scores.

    Args:
        student_latent: [B, F, C, H, W] generator x0 prediction (has grad).
        real_score_latent: [B, F, C, H, W] teacher score x0 (detached).
        fake_score_latent: [B, F, C, H, W] critic score x0 (detached).
        temperature: softmax temperature controlling sharpness.
        batch_weight: weight for batch-level (cross-sample) copula loss.
        frame_weight: weight for frame-level (temporal) copula loss.

    Returns:
        loss: scalar copula-aware loss.
        log_dict: dict with detached sub-losses for logging.
    """
    loss = _zero_loss(student_latent.device)
    log_dict = {}

    if batch_weight > 0:
        batch_loss = _batch_level_copula_loss(
            student_latent, real_score_latent, fake_score_latent, temperature)
        loss = loss + batch_weight * batch_loss
        log_dict["copula_batch_loss"] = batch_loss.detach()

    if frame_weight > 0:
        frame_loss = _frame_level_copula_loss(
            student_latent, real_score_latent, fake_score_latent, temperature)
        loss = loss + frame_weight * frame_loss
        log_dict["copula_frame_loss"] = frame_loss.detach()

    return loss, log_dict
