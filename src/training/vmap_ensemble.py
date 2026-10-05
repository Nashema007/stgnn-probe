"""Vectorized ensemble trainer for independently-initialized per-node models.

``torch.func.vmap`` + ``functional_call`` lets N independently-initialized
model instances train as one batched computation instead of a Python loop
of N sequential ``.train()`` calls — same statistical result as the loop
(each node still gets its own independently-fit weights, no cross-node
parameter sharing), just scheduled as one fused forward/backward per batch
instead of N tiny ones. This matters when each node's model/batch is too
small to saturate the GPU on its own: the loop leaves the GPU mostly idle
between tiny per-node kernel launches, with the real bottleneck being
CPU-side Python/dispatch overhead repeated 207 times.

The per-step *update mechanism* — forward, BatchNorm running-stat updates,
gradient clipping per node, the Adam update rule, and per-node
``ReduceLROnPlateau`` scheduling — was checked element-by-element against
PyTorch's own implementations given identical starting parameters, and
matches to float32 precision. The one expected non-match there is a
parameter whose gradient is mathematically nullified by an immediately
following BatchNorm (e.g. a Linear's bias before BatchNorm) — both the
loop and the vmapped version compute a near-zero gradient for it that is
pure floating-point noise, which Adam's adaptive scaling can amplify into
visibly different (but functionally irrelevant) trajectories; this is an
artifact of the architecture, not of vmapping it.

That mechanism-level check does NOT imply bit-for-bit or seed-for-seed
reproducibility against the old per-node loop end to end: the old loop
reset the global RNG to ``seed`` via ``_seed_everything`` inside every
node's own ``GraphTrainer.train()`` call, so each node's training-phase
randomness (dropout masks, etc.) started from the same fixed seed. The
ensemble trainer instead seeds once before constructing all N models, so
each node's initial weights — and which RNG state training-phase
randomness draws from — depend on a single un-reset RNG stream rather
than being individually reset per node. Both produce valid, independently-fit
per-node models; neither is a substitute for the other run-for-run.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state, vmap


def _patch_tsl_gated_tanh() -> None:
    """Replace TSL's @torch.jit.script gated_tanh with a plain Python version.

    TorchScript-compiled callables internally access raw tensor storage pointers
    at dispatch time. Inside vmap, tensors are virtual/batched constructs with no
    concrete storage, so that access raises:
        RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

    A plain Python equivalent is behaviourally identical — the scripting only
    adds a minor JIT compile overhead that is irrelevant at this call frequency.
    We patch both the canonical module and the local binding in temporal_conv so
    that forward passes through TemporalConvNet use the patched version.
    """

    def _gated_tanh(input: Tensor, dim: int = -1) -> Tensor:
        out, gate = input.tensor_split(2, dim=dim)
        return torch.tanh(out) * torch.sigmoid(gate)

    try:
        tsl_func = importlib.import_module("tsl.nn.functional")
        vars(tsl_func)["gated_tanh"] = _gated_tanh
        tsl_conv = importlib.import_module("tsl.nn.layers.base.temporal_conv")
        vars(tsl_conv)["gated_tanh"] = _gated_tanh
    except ImportError:
        pass


_patch_tsl_gated_tanh()


@dataclass
class EnsembleState:
    """N independently-initialized models, stacked into batched param/buffer tensors.

    ``base`` lives on the ``meta`` device and is never itself run — it only
    supplies the module structure ``functional_call`` dispatches through.
    Toggle ``base.train()``/``base.eval()`` to switch the whole ensemble's
    forward behaviour (BatchNorm/Dropout), exactly like calling
    ``model.train()``/``model.eval()`` in the non-vmapped trainer.
    """

    params: dict[str, Tensor]
    buffers: dict[str, Tensor]
    base: nn.Module
    n: int


def build_ensemble(
    model_factory: Callable[[], nn.Module], n: int, device: torch.device
) -> EnsembleState:
    """Construct ``n`` independently-initialized models and stack their state.

    Each call to ``model_factory()`` should return a freshly, randomly
    initialized model (no shared weights across the ``n`` instances) —
    mirrors building a fresh model per node in the original per-node loop.
    """
    models = [model_factory() for _ in range(n)]
    for m in models:
        m.train()
    params, buffers = stack_module_state(models)
    params = {k: v.to(device).clone().detach().requires_grad_(True) for k, v in params.items()}
    buffers = {k: v.to(device).clone() for k, v in buffers.items()}
    base = model_factory().to("meta")
    return EnsembleState(params=params, buffers=buffers, base=base, n=n)


def ensemble_forward(state: EnsembleState, x: Tensor, **kwargs: object) -> Tensor:
    """Run all ``n`` models' forward pass in one vmapped call.

    ``x``: ``(N, B, ...)`` — one batch per node/model, stacked along dim 0.
    ``randomness="different"`` while training gives each node's Dropout its
    own independent mask (matching independent per-node training); eval
    mode has no randomness to manage (Dropout is a no-op, BatchNorm uses
    running stats), so it's left at vmap's default.
    """

    def _call(params: dict[str, Tensor], buffers: dict[str, Tensor], x: Tensor) -> Tensor:
        return functional_call(state.base, (params, buffers), (x,), kwargs)

    randomness = "different" if state.base.training else "error"
    return vmap(_call, in_dims=(0, 0, 0), randomness=randomness)(state.params, state.buffers, x)


def clip_grad_norm_per_node_(params: dict[str, Tensor], max_norm: float, n: int) -> Tensor:
    """Clip gradients independently per node (leading dim of each stacked param).

    Matches ``torch.nn.utils.clip_grad_norm_`` applied separately to each
    node's own parameter slice — NOT one global norm across every node's
    gradients combined, which would let one node's large gradient suppress
    every other node's update.

    A node whose gradient norm is non-finite (NaN/Inf — e.g. that one
    node's series hit a numerical instability) has its gradient zeroed out
    entirely instead of scaled. This isn't a true no-op for that node —
    ``PerNodeAdam.step()`` still treats it as a zero-gradient observation,
    nudging its ``exp_avg`` slightly toward zero — but a zero gradient is
    finite, so it can never inject NaN into ``exp_avg``/``exp_avg_sq`` (an
    exponential moving average, where NaN would otherwise persist forever
    once introduced). Other nodes' gradients are untouched — this is a
    per-node ``(N,)``-broadcast, not a cross-node reduction. Returns the
    per-node norm (NaN where non-finite)
    so callers can log/warn about which nodes hit it.
    """
    grads = [p.grad for p in params.values() if p.grad is not None]
    if not grads:
        return torch.zeros(n)
    total_sq = torch.zeros(n, device=grads[0].device)
    for g in grads:
        total_sq += g.reshape(n, -1).pow(2).sum(dim=1)
    total_norm = total_sq.sqrt()
    nonfinite = ~torch.isfinite(total_norm)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    # A non-finite node's clip_coef is itself NaN — zero its scale factor
    # instead, so that node's grad becomes exactly 0 (a skipped update) and
    # cannot poison its Adam moment estimates.
    clip_coef = torch.where(nonfinite, torch.zeros_like(clip_coef), clip_coef)
    for g in grads:
        g.mul_(clip_coef.view(*([n] + [1] * (g.dim() - 1))))
    return total_norm


class PerNodeAdam:
    """Adam with an independent, per-node learning rate.

    Mirrors ``torch.optim.Adam``'s update rule exactly (checked elementwise
    against it) — the only difference is ``lr`` is a per-node tensor of
    shape ``(N,)`` instead of one shared float, so each node's LR can be
    plateau-scheduled independently while every node's parameters live in
    one stacked tensor updated by a single step() call. This works because
    Adam's moment estimates (``exp_avg``/``exp_avg_sq``) depend only on the
    gradient, not the learning rate — scheduling ``lr`` per node doesn't
    disturb their accumulation or cross-contaminate other nodes' state.
    """

    def __init__(
        self,
        params: dict[str, Tensor],
        n: int,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.n = n
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        device = next(iter(params.values())).device
        self.lr = torch.full((n,), lr, device=device)
        self.exp_avg = {k: torch.zeros_like(v) for k, v in params.items()}
        self.exp_avg_sq = {k: torch.zeros_like(v) for k, v in params.items()}
        self.step_t = torch.zeros(n, device=device)

    @torch.no_grad()
    def step(self) -> None:
        self.step_t += 1
        beta1, beta2 = self.betas
        bias_correction1 = 1 - beta1**self.step_t
        bias_correction2 = 1 - beta2**self.step_t
        for k, p in self.params.items():
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p
            shape = [self.n] + [1] * (p.dim() - 1)
            ea = self.exp_avg[k]
            eas = self.exp_avg_sq[k]
            ea.mul_(beta1).add_(grad, alpha=1 - beta1)
            eas.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            step_size = (self.lr / bias_correction1).view(*shape)
            denom = (eas / bias_correction2.view(*shape)).sqrt().add_(self.eps)
            p.sub_(step_size * ea / denom)

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None

    def state_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "exp_avg": self.exp_avg,
            "exp_avg_sq": self.exp_avg_sq,
            "step_t": self.step_t,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.lr = cast(Tensor, state["lr"])
        self.exp_avg = cast(dict[str, Tensor], state["exp_avg"])
        self.exp_avg_sq = cast(dict[str, Tensor], state["exp_avg_sq"])
        self.step_t = cast(Tensor, state["step_t"])


class PerNodePlateau:
    """Per-node ``ReduceLROnPlateau`` (``mode="min"``).

    Mirrors ``torch.optim.lr_scheduler.ReduceLROnPlateau``'s ``step``/
    ``is_better``/``_reduce_lr`` (checked against its source), including
    ``cooldown`` and ``threshold_mode`` (``"rel"`` or ``"abs"``) — every
    kwarg ``scheduler_kwargs`` in a model YAML can set for the real
    ``ReduceLROnPlateau`` GraphTrainer uses, so a config that's valid for
    every other model in this codebase doesn't raise ``TypeError`` here —
    but tracks ``best``/``num_bad_epochs``/``cooldown_counter`` as ``(N,)``
    tensors instead of single floats, so each node's LR decays
    independently based on its own validation-loss history.
    """

    def __init__(
        self,
        adam: PerNodeAdam,
        n: int,
        factor: float,
        patience: int,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode!r}.")
        self.adam = adam
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        device = adam.lr.device
        self.best = torch.full((n,), float("inf"), device=device)
        self.num_bad_epochs = torch.zeros(n, dtype=torch.long, device=device)
        self.cooldown_counter = torch.zeros(n, dtype=torch.long, device=device)

    def _is_better(self, val_loss: Tensor) -> Tensor:
        if self.threshold_mode == "rel":
            return val_loss < self.best * (1.0 - self.threshold)
        return val_loss < self.best - self.threshold

    @torch.no_grad()
    def step(self, val_loss: Tensor) -> None:
        better = self._is_better(val_loss)
        self.best = torch.where(better, val_loss, self.best)
        num_bad = torch.where(
            better, torch.zeros_like(self.num_bad_epochs), self.num_bad_epochs + 1
        )
        # While in cooldown, a node's bad-epoch streak doesn't count —
        # mirrors ReduceLROnPlateau.step ignoring bad epochs during cooldown.
        in_cooldown = self.cooldown_counter > 0
        self.num_bad_epochs = torch.where(in_cooldown, torch.zeros_like(num_bad), num_bad)
        self.cooldown_counter = torch.clamp(self.cooldown_counter - 1, min=0)

        reduce = self.num_bad_epochs > self.patience
        new_lr = (self.adam.lr * self.factor).clamp(min=self.min_lr)
        apply = reduce & ((self.adam.lr - new_lr) > self.eps)
        self.adam.lr = torch.where(apply, new_lr, self.adam.lr)
        self.cooldown_counter = torch.where(
            reduce, torch.full_like(self.cooldown_counter, self.cooldown), self.cooldown_counter
        )
        self.num_bad_epochs = torch.where(
            reduce, torch.zeros_like(self.num_bad_epochs), self.num_bad_epochs
        )

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.best = state["best"]
        self.num_bad_epochs = state["num_bad_epochs"]
        self.cooldown_counter = state.get("cooldown_counter", torch.zeros_like(self.num_bad_epochs))


class PerNodeEpochScheduler:
    """LR schedule that's a pure function of epoch number, applied uniformly
    to every node — covers GraphTrainer's ``"multistep"``, ``"exponential"``,
    and ``"none"`` scheduler options (everything except ``"plateau"``, which
    needs ``PerNodePlateau`` since it depends on each node's own val loss).

    Unlike ``PerNodePlateau``, these schedules don't diverge per node (every
    node gets the same epoch-indexed LR), so there's no per-node state to
    track or checkpoint beyond the base ``lr`` baked in at construction —
    ``step(epoch)`` just recomputes ``adam.lr`` from scratch each call,
    which is naturally resume-safe (calling it with the same epoch after a
    crash reproduces the same lr, no counters to restore).
    """

    def __init__(
        self,
        adam: PerNodeAdam,
        mode: str,
        base_lr: float,
        milestones: list[int] | None = None,
        gamma: float = 0.1,
        decay_rate: float = 0.97,
    ) -> None:
        if mode not in ("multistep", "exponential", "none"):
            raise ValueError(f"mode must be 'multistep', 'exponential', or 'none', got {mode!r}.")
        self.adam = adam
        self.mode = mode
        self.base_lr = base_lr
        self.milestones = sorted(milestones or [50, 70, 100])
        self.gamma = gamma
        self.decay_rate = decay_rate

    @torch.no_grad()
    def step(self, epoch: int) -> None:
        if self.mode == "none":
            lr = self.base_lr
        elif self.mode == "exponential":
            lr = self.base_lr * (self.decay_rate**epoch)
        else:
            lr = self.base_lr * (self.gamma ** sum(epoch >= m for m in self.milestones))
        self.adam.lr.fill_(lr)

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        del state


class PerNodeEarlyStopping:
    """Tracks each node's best val loss / patience counter and freezes its
    best (params, buffers) snapshot the moment that node improves.

    Training keeps running every node every epoch (cheaper than masking
    per-node updates inside the vmapped forward/backward), but once a
    node's ``num_bad_epochs`` exceeds ``patience`` it's marked ``done`` —
    callers stop feeding it into the loss/metrics bookkeeping that decides
    when to stop, though it costs nothing to let its parameters keep
    drifting in the background since the final result always restores its
    frozen best snapshot, never its latest one (mirrors
    ``GraphTrainer``'s "reload best checkpoint" behaviour at the end of
    ``train()``).
    """

    def __init__(self, n: int, patience: int, device: torch.device) -> None:
        self.n = n
        self.patience = patience
        self.best_val_loss = torch.full((n,), float("inf"), device=device)
        self.num_bad_epochs = torch.zeros(n, dtype=torch.long, device=device)
        self.best_params: dict[str, Tensor] | None = None
        self.best_buffers: dict[str, Tensor] | None = None

    def update(
        self, val_loss: Tensor, params: dict[str, Tensor], buffers: dict[str, Tensor]
    ) -> None:
        if self.best_params is None:
            self.best_params = {k: v.detach().clone() for k, v in params.items()}
            self.best_buffers = {k: v.detach().clone() for k, v in buffers.items()}
        assert self.best_params is not None
        assert self.best_buffers is not None

        improved = val_loss < self.best_val_loss
        self.best_val_loss = torch.where(improved, val_loss, self.best_val_loss)
        self.num_bad_epochs = torch.where(
            improved, torch.zeros_like(self.num_bad_epochs), self.num_bad_epochs + 1
        )
        for k in params:
            mask = improved.view(*([self.n] + [1] * (params[k].dim() - 1)))
            self.best_params[k] = torch.where(mask, params[k].detach(), self.best_params[k])
        for k in buffers:
            mask = improved.view(*([self.n] + [1] * (buffers[k].dim() - 1)))
            self.best_buffers[k] = torch.where(mask, buffers[k].detach(), self.best_buffers[k])

    @property
    def all_done(self) -> bool:
        return bool((self.num_bad_epochs >= self.patience).all().item())

    def state_dict(self) -> dict[str, object]:
        return {
            "best_val_loss": self.best_val_loss,
            "num_bad_epochs": self.num_bad_epochs,
            "best_params": self.best_params,
            "best_buffers": self.best_buffers,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.best_val_loss = cast(Tensor, state["best_val_loss"])
        self.num_bad_epochs = cast(Tensor, state["num_bad_epochs"])
        self.best_params = cast(dict[str, Tensor] | None, state["best_params"])
        self.best_buffers = cast(dict[str, Tensor] | None, state["best_buffers"])
