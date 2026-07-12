from __future__ import annotations

import torch
from torch import nn


def make_sparse_adjacency(graph: dict[str, object], device: torch.device) -> torch.Tensor:
    row = torch.as_tensor(graph["adj_row"], dtype=torch.long, device=device)
    col = torch.as_tensor(graph["adj_col"], dtype=torch.long, device=device)
    value = torch.as_tensor(graph["adj_value"], dtype=torch.float32, device=device)
    num_nodes = int(graph["metadata"]["num_nodes"])
    return torch.sparse_coo_tensor(torch.stack([row, col], dim=0), value, (num_nodes, num_nodes)).coalesce()


class ContinuousTimeMambaCore(nn.Module):
    """Selective diagonal SSM with explicit delta-t discretization.

    For each step, the continuous transition A is discretized as:

    A_bar = exp(delta_t * A)
    B_bar = ((A_bar - 1) / A) * B_t

    where ``delta_t`` is the observed gap in hours multiplied by a positive
    input-dependent gate. This is the critical irregular-time path used by both
    the temporal backbone and the Hilbert-ordered spatial scan.
    """

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim * state_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim * state_dim)
        self.a_log = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.d_skip = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Run the scan.

        Args:
            x: Tensor with shape ``[batch, steps, items, hidden_dim]``.
            dt: Tensor with shape ``[batch, steps]`` or ``[batch, steps, items]``.
        """
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, steps, items, hidden_dim].")
        batch, steps, items, hidden = x.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden}.")
        if dt.ndim == 2:
            dt = dt[:, :, None].expand(batch, steps, items)
        elif dt.ndim != 3:
            raise ValueError("dt must have shape [batch, steps] or [batch, steps, items].")

        gate_in, value_in = self.in_proj(x).chunk(2, dim=-1)
        u = torch.nn.functional.silu(gate_in) * value_in
        a = -torch.exp(self.a_log).view(1, 1, self.hidden_dim, self.state_dim)
        state = x.new_zeros(batch, items, self.hidden_dim, self.state_dim)
        outputs = []

        for step in range(steps):
            u_t = u[:, step]
            delta_gate = torch.nn.functional.softplus(self.delta_proj(u_t)) + 1e-4
            delta = (dt[:, step].unsqueeze(-1) * delta_gate).clamp(min=1e-4, max=72.0)
            b_t = self.b_proj(u_t).view(batch, items, self.hidden_dim, self.state_dim)
            c_t = self.c_proj(u_t).view(batch, items, self.hidden_dim, self.state_dim)
            a_bar = torch.exp(delta.unsqueeze(-1) * a)
            b_bar = ((a_bar - 1.0) / a) * b_t
            state = a_bar * state + b_bar * u_t.unsqueeze(-1)
            y_t = (state * c_t).sum(dim=-1) + self.d_skip.view(1, 1, -1) * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return self.dropout(self.out_proj(y))


class SpatialGraphMambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.seq_core = ContinuousTimeMambaCore(hidden_dim, state_dim, dropout)
        self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, order: torch.Tensor, inverse_order: torch.Tensor) -> torch.Tensor:
        """Apply graph mixing and a Hilbert-ordered Mamba scan.

        Args:
            x: ``[batch, nodes, hidden_dim]``.
        """
        # CUDA sparse addmm does not support FP16.  Autocast may cast the dense
        # operand to FP16 (for example when BF16 is unavailable), so keep only
        # the sparse aggregation in FP32 and resume autocast for the projection.
        graph_messages = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            adj_float = adj.float()
            for item in x:
                graph_messages.append(torch.sparse.mm(adj_float, item.float()))
        graph_h = self.graph_proj(torch.stack(graph_messages, dim=0))

        ordered = x.index_select(dim=1, index=order)
        seq_dt = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        seq_h = self.seq_core(ordered.unsqueeze(2), seq_dt).squeeze(2)
        seq_h = seq_h.index_select(dim=1, index=inverse_order)

        mixed = self.mix(torch.cat([graph_h, seq_h], dim=-1))
        h = self.norm(x + mixed)
        return self.ffn_norm(h + self.ffn(h))


class UnifiedGraphMambaPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_dim: int,
        num_airnow: int,
        spatial_layers: int,
        temporal_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_airnow = num_airnow
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.spatial_layers = nn.ModuleList(
            [SpatialGraphMambaBlock(hidden_dim, state_dim, dropout) for _ in range(spatial_layers)]
        )
        self.temporal_layers = nn.ModuleList(
            [ContinuousTimeMambaCore(hidden_dim, state_dim, dropout) for _ in range(temporal_layers)]
        )
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(temporal_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, dt: torch.Tensor, adj: torch.Tensor, order: torch.Tensor, inverse_order: torch.Tensor) -> torch.Tensor:
        """Predict AirNow NO2 for the final window step.

        Args:
            x: ``[batch, time, nodes, input_dim]``.
            dt: ``[batch, time]`` actual scan gaps in hours.
        """
        batch, steps, nodes, _features = x.shape
        h = self.input_proj(x)
        spatial_steps = []
        for step in range(steps):
            h_t = h[:, step]
            for layer in self.spatial_layers:
                h_t = layer(h_t, adj, order, inverse_order)
            spatial_steps.append(h_t)
        h = torch.stack(spatial_steps, dim=1)

        for core, norm in zip(self.temporal_layers, self.temporal_norms):
            h = norm(h + core(h, dt))

        airnow_h = h[:, -1, : self.num_airnow, :]
        return self.head(airnow_h).squeeze(-1)


class GraphBoundPredictor(nn.Module):
    """Bind static graph tensors to a predictor.

    This wrapper lets DataParallel split only ``x`` and ``dt`` along the batch
    dimension while replicating the sparse graph adjacency and Hilbert ordering
    tensors on each GPU.
    """

    def __init__(
        self,
        predictor: UnifiedGraphMambaPredictor,
        adj: torch.Tensor,
        order: torch.Tensor,
        inverse_order: torch.Tensor,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.register_buffer("adj", adj)
        self.register_buffer("order", order)
        self.register_buffer("inverse_order", inverse_order)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return self.predictor(x, dt, self.adj, self.order, self.inverse_order)
