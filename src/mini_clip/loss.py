import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, logit_scale):
    """Symmetric InfoNCE over the batch.

    Both feature tensors must already be L2 normalised. The diagonal of the
    similarity matrix is the positive pair, everything else in the batch is
    treated as a negative.
    """
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    targets = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2
