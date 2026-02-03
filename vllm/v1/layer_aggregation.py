import torch

from typing import Any, Callable, Optional, Sequence

EPS = 1e-20


def aggregate_layer_logits(logits_list: Sequence[torch.Tensor],
                           eps: float = EPS) -> Optional[torch.Tensor]:
    if not logits_list:
        return None
    if len(logits_list) == 1:
        return logits_list[0]

    probs = [logits.softmax(dim=-1) for logits in logits_list]
    stacked = torch.stack(probs, dim=0)
    mean_probs = stacked.mean(dim=0)
    return torch.log(mean_probs.clamp_min(eps))


def aggregate_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    compute_logits_fn: Callable[[torch.Tensor, Optional[Any]],
                                Optional[torch.Tensor]],
    sampling_metadata: Optional[Any] = None,
    eps: float = EPS,
    reuse_last_logits: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    logits_list = []
    for i, hidden in enumerate(hidden_states):
        if reuse_last_logits is not None and i == len(hidden_states) - 1:
            logits = reuse_last_logits
        else:
            logits = compute_logits_fn(hidden, sampling_metadata)
        if logits is None:
            return None
        logits_list.append(logits)

    return aggregate_layer_logits(logits_list, eps=eps)
