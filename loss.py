import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, neighbor_num, class_num, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.neighbor_num = neighbor_num
        self.class_num = class_num
        self.device = device
        self.mask = self._create_mask(batch_size).to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _create_mask(self, N):
        mask = torch.ones(N, N, dtype=torch.bool)
        mask.fill_diagonal_(False)
        half_N = N // 2
        mask[:half_N, half_N:] = torch.eye(half_N, dtype=torch.bool)
        mask[half_N:, :half_N] = torch.eye(half_N, dtype=torch.bool)
        return ~mask

    def forward(self, h_i, h_j, fused_graph):
        return self.neighborhood_loss(h_i, h_j, fused_graph, self.temperature)

    def neighborhood_loss(self, z1, z2, fused_graph, temperature):
        loss1 = self._contrastive_with_graph(z1, z2, fused_graph, temperature)
        loss2 = self._contrastive_with_graph(z2, z1, fused_graph, temperature)
        return 0.5 * (loss1 + loss2)

    def _contrastive_with_graph(self, z1, z2, fused_graph, temperature):
        intra_sim = torch.exp(torch.mm(z1, z1.t()) / temperature).clamp(min=1e-8, max=1e8)
        inter_sim = torch.exp(torch.mm(z1, z2.t()) / temperature).clamp(min=1e-8, max=1e8)
        pos_intra = (intra_sim * fused_graph).sum(dim=1)
        pos_inter = (inter_sim * fused_graph).sum(dim=1)
        pos_loss = pos_intra + pos_inter
        pos_self = inter_sim.diagonal().clamp(min=1e-8, max=1e8)
        neg_loss = (intra_sim.sum(dim=1) + inter_sim.sum(dim=1) - intra_sim.diagonal()).clamp(min=1e-8)
        loss = -torch.log((pos_self + pos_loss + 1e-8) / (neg_loss + 1e-8))
        return loss.mean()