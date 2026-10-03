import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.spherical_convolution import SphericalConv


class MultiModalSFNO(nn.Module):
    """
    Input:
        (B, M, Cin, H, W)

    Output:
        (B, M, Cout, H, W)

    Architecture:

        modality-specific lifting
                ↓
             concat
                ↓
         shared SFNO trunk
                ↓
          shared latent
                ↓
      modality-specific projections
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modalities=1,
        hidden_channels=256,
        n_modes=(110, 128),
        n_layers=8,
        lifting_channel_ratio=1,
        projection_channel_ratio=1,
        non_linearity=F.gelu,
        norm=None,
        grid_embedding_dim=None,
        factorization="dense",
        rank=1.0,
    ):
        super().__init__()

        self.n_modalities = n_modalities
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        lifting_channels = int(lifting_channel_ratio * hidden_channels)

        projection_channels = int(projection_channel_ratio * hidden_channels)

        # --------------------------------------------------
        # modality-specific lifting heads
        # --------------------------------------------------

        self.liftings = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    hidden_channels=lifting_channels,
                    n_layers=2,
                    n_dim=2,
                    non_linearity=non_linearity,
                )
                for _ in range(n_modalities)
            ]
        )
        
        self.fusion_replica = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=2,
            n_dim=2,
            non_linearity=non_linearity,
        )

        # --------------------------------------------------
        # geometry mixing MLP
        #
        # (B, hidden + grid_embedding, H, W)
        #          ->
        # (B, hidden, H, W)
        # --------------------------------------------------

        if grid_embedding_dim is not None:
            self.geometry_mixing = ChannelMLP(
                in_channels=hidden_channels + grid_embedding_dim,
                out_channels=hidden_channels,
                hidden_channels=hidden_channels,
                n_layers=2,
                n_dim=2,
                non_linearity=non_linearity,
            )

        # --------------------------------------------------
        # shared SFNO trunk
        # --------------------------------------------------

        self.sfno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_layers=n_layers,
            non_linearity=non_linearity,
            norm=norm,
            factorization=factorization,
            rank=rank,
            conv_module=SphericalConv,
        )

        self.n_layers = n_layers

        # --------------------------------------------------
        # modality-specific projection heads
        # --------------------------------------------------

        self.projections = nn.ModuleList(
            [
                ChannelMLP(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    hidden_channels=projection_channels,
                    n_layers=2,
                    n_dim=2,
                    non_linearity=non_linearity,
                )
                for _ in range(n_modalities)
            ]
        )

    def forward(self, x, grid_embeddings=None):
        """
        x:
            (B, M, Cin, H, W)

        returns:
            (B, M, Cout, H, W)
        """

        B, M, Cin, H, W = x.shape

        assert (
            M == self.n_modalities
        ), f"Expected {self.n_modalities} modalities, got {M}"

        # ==================================================
        # modality-specific lifting
        # ==================================================

        lifted = []

        for m in range(M):

            xm = x[:, m]  # (B,Cin,H,W)

            hm = self.liftings[m](xm)  # (B,hidden,H,W)

            lifted.append(hm)

        # ==================================================
        # concatenate modalities
        # ==================================================

        h = torch.cat(
            lifted,
            dim=1,
        )

        # (B, M*hidden, H, W)
        
        h = self.fusion_replica(h)
        
        # ==================================================
        # geometry mixing
        # ==================================================

        if grid_embeddings is not None:
            h = self.geometry_mixing(
                torch.cat([h, grid_embeddings], dim=1)
            )

        # ==================================================
        # shared SFNO trunk
        # ==================================================

        for layer_idx in range(self.n_layers):

            h = self.sfno_blocks(
                h,
                index=layer_idx,
            )

        # ==================================================
        # modality-specific projections
        # ==================================================

        outputs = []

        for m in range(M):

            ym = self.projections[m](h)

            outputs.append(ym)

        y = torch.stack(
            outputs,
            dim=1,
        )

        # (B,M,Cout,H,W)

        return y