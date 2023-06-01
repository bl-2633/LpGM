import os
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir_path)
import torch
import numpy as np
from torch import nn
from graph_transformer_pytorch import GraphTransformer

class Coord_VAE_seq(nn.Module):
    def __init__(self, device = 0):
        super().__init__()
        self.in_chnnel = 19
        self.kernel_size = (3,3)
        self.channels = [32, 64, 128, 256]
        self.latent_dim = 64
        self.padding = (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[1] - 1) / 2))
        
        encoder_layers_2D = [nn.Sequential(
            nn.Conv2d(self.in_chnnel, self.channels[0], kernel_size = self.kernel_size, padding =self.padding),
            nn.InstanceNorm2d(self.channels[0]),
            nn.LeakyReLU(inplace = True)
        )]
        for i in range(len(self.channels) - 1):
            encoder_layers_2D.append(nn.Sequential(
                nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size = self.kernel_size, padding = self.padding),
                nn.InstanceNorm2d(self.channels[i+1]),
                nn.LeakyReLU(inplace = True)
            ))
        self.encoder_2D = nn.Sequential(*encoder_layers_2D)
        
        encoder_layers_1D = [nn.Sequential(
            nn.Conv1d(26, self.channels[0], kernel_size = self.kernel_size[0], padding =self.padding[0]),
            nn.InstanceNorm1d(self.channels[0]),
            nn.LeakyReLU(inplace = True)
        )]
        for i in range(len(self.channels) - 1):
            encoder_layers_1D.append(nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size = self.kernel_size[0], padding = self.padding[0]),
                nn.InstanceNorm1d(self.channels[i+1]),
                nn.LeakyReLU(inplace = True)
            ))
        self.encoder_1D = nn.Sequential(*encoder_layers_1D)

        self.fc_mu = nn.Linear(2 * self.channels[-1], self.latent_dim)
        self.fc_var= nn.Linear(2 * self.channels[-1], self.latent_dim)

        self.up_sample = nn.Linear(self.latent_dim, self.channels[-1]) 
        self.channels.reverse()
        decoder_layers_1D = []
        decoder_layers_2D = []
        for i in range(len(self.channels) - 1):
            decoder_layers_1D.append(nn.Sequential(
                nn.Conv1d(self.channels[i], self.channels[i+1], kernel_size = self.kernel_size[0], padding = self.padding[0]),
                nn.InstanceNorm1d(self.channels[i+1]),
                nn.LeakyReLU(inplace = True),
            ))
        self.decoder_1D = nn.Sequential(*decoder_layers_1D)
        for i in range(len(self.channels) - 1):
            decoder_layers_2D.append(nn.Sequential(
                nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size = self.kernel_size, padding = self.padding),
                nn.InstanceNorm2d(self.channels[i+1]),
                nn.LeakyReLU(inplace = True),
            ))
        self.decoder_2D = nn.Sequential(*decoder_layers_2D)
        
        self.Coords_Transformer = GraphTransformer(
            dim = self.channels[-1],
            depth = 6,
            with_feedforwards = True,
            gated_residual = True,
            rel_pos_emb = True      
        )

        self.coord_proj = nn.Sequential(
            nn.Linear(self.channels[-1], 12, bias = False),
            )
        self.seq_proj = nn.Sequential(
            nn.LayerNorm(self.channels[-1]),
            nn.Linear(self.channels[-1], 20),
            nn.GELU(),
            nn.Linear(20, 20)
            )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu        
        return z

    def forward(self, scalar_feat, edge_feat):
        device = scalar_feat.get_device()
        if device == -1:
            device = 'cpu'
        edge_feat = edge_feat.permute(0,3,1,2)
        scalar_feat = scalar_feat.permute(0,2,1)
 
        feat_1D = self.encoder_1D(scalar_feat)
        feat_2D = self.encoder_2D(edge_feat)
        latnet_feat = torch.cat([torch.mean(feat_2D, dim = 3).transpose(-1, -2), feat_1D.transpose(-1,-2)], axis = 2)
        mu, logvar = self.fc_mu(latnet_feat), self.fc_var(latnet_feat)
        z = self.reparameterize(mu, logvar)
        feat_1D = self.up_sample(z).transpose(-2,-1)
        feat_2D = torch.einsum('abc, abd -> abcd', feat_1D, feat_1D)

        out_2D = self.decoder_2D(feat_2D).permute(0, 3, 2, 1)
        out_1D = self.decoder_1D(feat_1D).permute(0, 2, 1)

        out_2D = (out_2D.transpose(1,2) + out_2D)
        mask = torch.ones(1, out_1D.size(1)).bool().to(device)
        scalar_out, _ = self.Coords_Transformer(out_1D, out_2D, mask = mask)
        coords = self.coord_proj(scalar_out).view(scalar_out.size(0), scalar_out.size(1), 4, 3)
        seqs = self.seq_proj(scalar_out)

        return coords, seqs, mu, logvar

    def get_latent(self, scalar_feat, edge_feat):
        edge_feat = edge_feat.permute(0,3,1,2)
        scalar_feat = scalar_feat.permute(0,2,1)
        feat_1D = self.encoder_1D(scalar_feat)
        feat_2D = self.encoder_2D(edge_feat)
        latnet_feat = torch.cat([torch.mean(feat_2D, dim = 3).transpose(-1, -2), feat_1D.transpose(-1,-2)], axis = 2)
        mu, logvar = self.fc_mu(latnet_feat), self.fc_var(latnet_feat)
        z = self.reparameterize(mu, logvar)
        return z

    def hallucinate(self, scalar_feat, edge_feat, num_decoys,relaxed = False):
        device = scalar_feat.get_device()
        if device == -1:
            device = 'cpu'
        edge_feat = edge_feat.permute(0,3,1,2)
        scalar_feat = scalar_feat.permute(0,2,1)
 
        feat_1D = self.encoder_1D(scalar_feat)
        feat_2D = self.encoder_2D(edge_feat)
        latnet_feat = torch.cat([torch.mean(feat_2D, dim = 3).transpose(-1, -2), feat_1D.transpose(-1,-2)], axis = 2)
        mu, logvar = self.fc_mu(latnet_feat), self.fc_var(latnet_feat)
        if relaxed:
            logvar = torch.zeros_like(logvar)
        z = torch.cat([self.reparameterize(mu, logvar) for _ in range(num_decoys)], dim = 0)
        feat_1D = self.up_sample(z).transpose(-2,-1)
        feat_2D = torch.einsum('abc, abd -> abcd', feat_1D, feat_1D)

        out_2D = self.decoder_2D(feat_2D).permute(0, 3, 2, 1)
        out_1D = self.decoder_1D(feat_1D).permute(0, 2, 1)

        out_2D = (out_2D.transpose(1,2) + out_2D)
        mask = torch.ones(num_decoys, out_1D.size(1)).bool().to(device)
        scalar_out, _ = self.Coords_Transformer(out_1D, out_2D, mask = mask)
        coords = self.coord_proj(scalar_out).view(scalar_out.size(0), scalar_out.size(1), 4, 3)
        seqs = self.seq_proj(scalar_out)
        
        return coords, z

    
if __name__ == '__main__':
    import tqdm
    import VAE_loss
    import CATH_dset
    from torch.utils import data as D
    
    dset = CATH_dset.CATH_feats(partition = 'train')
    loader = D.DataLoader(dset, num_workers = 10, batch_size = 1)
    device = torch.device('cuda:1')
    model = Coord_VAE_TFN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = VAE_loss.VAE_TFN_loss()
    pbar = tqdm.tqdm(total = dset.__len__())
    s = 0
    total_loss = 0
    for coords, scalar_feat, edge_feat, coords_feat  in loader:
        pbar.update()
        out_coords, mu, logvar = model(raw_coords = coords[:, :, 0, :].to(device), scalar_feat = scalar_feat.to(device), coords_feat = coords_feat.to(device), edge_feat = edge_feat.to(device))
        loss = loss_fn(coords.to(device), out_coords, mu, logvar)
        optim.zero_grad()
        loss.backward()
        optim.step()            
        total_loss += loss.item()
        s+=1
        pbar.set_description(str(total_loss/s)[:6])
