import torch


def retrieval_metrics(image_features, text_features, text2img, ks=(1, 5, 10)):
    """Recall@K in both directions.

    image_features: (N_img, D), L2 normalised
    text_features:  (N_txt, D), L2 normalised
    text2img:       (N_txt,) index of the image each caption describes
    """
    sims = text_features @ image_features.t()  # (N_txt, N_img)
    out = {}

    # text -> image: one correct image per caption
    ranks = []
    order = sims.argsort(dim=1, descending=True)
    for i in range(sims.size(0)):
        rank = (order[i] == text2img[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)
    for k in ks:
        out[f"t2i_R@{k}"] = (ranks < k).float().mean().item() * 100
    out["t2i_medr"] = ranks.median().item() + 1

    # image -> text: 5 captions are correct, we score the best ranked one
    sims_i = sims.t()  # (N_img, N_txt)
    order_i = sims_i.argsort(dim=1, descending=True)
    ranks_i = []
    for i in range(sims_i.size(0)):
        correct = (text2img[order_i[i]] == i).nonzero(as_tuple=True)[0]
        ranks_i.append(correct[0].item() if correct.numel() else sims_i.size(1))
    ranks_i = torch.tensor(ranks_i, dtype=torch.float)
    for k in ks:
        out[f"i2t_R@{k}"] = (ranks_i < k).float().mean().item() * 100
    out["i2t_medr"] = ranks_i.median().item() + 1

    out["rsum"] = sum(out[f"{d}_R@{k}"] for d in ("t2i", "i2t") for k in ks)
    return out
