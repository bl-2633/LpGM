from ast import Gt
import os
from re import L
from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(dir_path)
import torch
import numpy as np
from torch import nn
import calc_torsion
import orientation_utils
from protein_learning.networks.loss.coord_loss import FAPELoss

class VAE_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fape_loss = FAPELoss()
    
    def __sphererical_projection(self, phi, psi, omega):
        x = torch.cos(phi) * torch.cos(psi) * torch.cos(omega)
        y = torch.sin(phi) * torch.cos(psi) * torch.cos(omega)
        z = torch.sin(psi) * torch.cos(omega)
        t = torch.sin(omega)
        projected_coords = torch.cat([x, y, z, t])
        return projected_coords
    
    def __binned_distance_BCE(self, pred_bin, gt_dist):
        bounds = (2,22)
        n_bins = 64
        
        bin_edges = torch.linspace(bounds[0], bounds[1], steps = n_bins + 1)
        bin_edges = [(bin_edges[i].item(),bin_edges[i+1].item()) for i in range(len(bin_edges)-1)]
        quantized = torch.stack([torch.bitwise_and(edge[0]<gt_dist, gt_dist<edge[1]).float() for edge in bin_edges], dim = -1)        
        loss = torch.nn.functional.binary_cross_entropy(pred_bin, quantized)
        return loss
    
    def __torsion_loss(self, pred_torsion, gt_torsion):
        pred_phi = pred_torsion[:,:,:2] / torch.linalg.norm(pred_torsion[:,:,:2])
        pred_psi = pred_torsion[:,:,2:4] / torch.linalg.norm(pred_torsion[:,:,2:4])
        pred_omega = pred_torsion[:,:,4:6] / torch.linalg.norm(pred_torsion[:,:,4:6])
        
        phi_loss = torch.mean(torch.linalg.norm(pred_phi - gt_torsion[:,:,:2], dim = -1))
        psi_loss = torch.mean(torch.linalg.norm(pred_psi - gt_torsion[:,:,2:4], dim = -1))
        omega_loss = torch.mean(torch.linalg.norm(pred_omega - gt_torsion[:,:,4:6], dim = -1))
        loss_1 = torch.mean(torch.stack([phi_loss, psi_loss, omega_loss]))
        loss_2 = torch.mean(torch.stack([torch.abs(1 - torch.linalg.norm(pred_torsion[:,:,:2])), 
                                         torch.abs(1 - torch.linalg.norm(pred_torsion[:,:,2:4])),
                                         torch.abs(1 - torch.linalg.norm(pred_torsion[:,:,4:6]))]))
        return loss_1 + 0.02 * loss_2

    def forward(self, gt, pred, gt_torsion, mu, logvar, pred_bin_dist, pred_torsion, beta = 1e-4):
        gt_CA = gt[:,:,0,:]
        pred_CA = pred[:,:,0,:]
        gt_N = gt[:,:,1,:]
        pred_N = pred[:,:,1,:]
        gt_C = gt[:,:,2,:]
        pred_C = pred[:,:,2,:]
        gt_dist = torch.cdist(gt_CA, gt_CA)
        pred_dist = torch.cdist(pred_CA, pred_CA)

        #dist_loss = self.__binned_distance_BCE(pred_bin_dist, gt_dist)
        #torsion_loss = self.__torsion_loss(pred_torsion, gt_torsion)        
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = -1))
        coord_loss = self.fape_loss(pred_coords=pred, true_coords=gt)
        return  coord_loss + beta * kl_loss, coord_loss
    





        