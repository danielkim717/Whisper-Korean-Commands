"""원본 cell 14의 프로토타입 및 Unknown 거절 구현."""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_embeddings_from_loader(model, loader, device, l2_normalize=True):
    model.eval()
    E_list, y_list = [], []
    for audio, labels in loader:
        audio = audio.to(device)
        labels = labels.to(device)

        emb = model.forward_features(audio)  # [B, D]
        if l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)

        E_list.append(emb.detach().cpu())
        y_list.append(labels.detach().cpu())

    E = torch.cat(E_list, dim=0)  # [N, D]
    y = torch.cat(y_list, dim=0)  # [N]
    return E, y


def build_prototypes(E, y, num_classes):
    # E: [N,D]  (CPU 텐서)
    D = E.size(1)
    protos = torch.zeros(num_classes, D, dtype=E.dtype)
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        proto = E[idx].mean(dim=0)
        proto = F.normalize(proto, p=2, dim=0)
        protos[c] = proto
    return protos


class ModelWithUnknownRaw(nn.Module):
    def __init__(self, base_model, protos, threshold=0.125):
        super().__init__()
        self.base = base_model
        self.register_buffer("protos", protos)  # [C, D] (CPU/저장됨)
        self.threshold = float(threshold)

    def forward(self, audio):
        # logits -> softmax top1 prob
        logits = self.base(audio)                 # [B, C]
        probs = torch.softmax(logits, dim=1)      # [B, C]
        conf, pred = torch.max(probs, dim=1)      # [B], [B]

        # embedding -> cosine distance to prototypes
        emb = self.base.forward_features(audio)   # [B, D]
        emb = F.normalize(emb, p=2, dim=1)

        cos = torch.matmul(emb, self.protos.t())  # [B, C]
        dist = 1.0 - cos                          # cosine distance
        dmin, _ = torch.min(dist, dim=1)          # [B]

        # unknown이면 -1로 치환
        pred_out = pred.clone()
        unknown_mask = dmin > self.threshold
        pred_out = torch.where(
            unknown_mask,
            torch.full_like(pred_out, -1),
            pred_out
        )
        return pred_out, conf, dmin
