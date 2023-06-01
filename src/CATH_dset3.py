import torch
import numpy as np
from torch.utils import data as D
import json
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/'.join(dir_path.split('/')[:-2]) + '/data')
from protein_learning.protein_utils.align.per_residue import impute_beta_carbon
from protein_learning.protein_utils.dihedral.orientation_utils import get_tr_rosetta_orientation_mats
from Bio.PDB import PDBParser, Polypeptide
import calc_torsion

def seq2onehot(seq):
    '''
    Function that takes a raw sequence and return a one-hot encoded matrix
    seq: raw sequence as a string with length L
    matrix: one-hot encoded seqeunce as 25*L matrix
    '''
    alphabet = {
     'A': 0,
     'R': 1,
     'N': 2,
     'D': 3,
     'C': 4,
     'Q': 5,
     'E': 6,
     'G': 7,
     'H': 8,
     'I': 9,
     'L': 10,
     'K': 11,
     'M': 12,
     'F': 13,
     'P': 14,
     'S': 15,
     'T': 16,
     'W': 17,
     'Y': 18,
     'V': 19,
    }
    length = len(seq) #get the length of the input sequence
    matrix = torch.zeros(size = (length, 20)) #initialize the matrix
    # populating the one-hot mateix given the input sequence
    for i, aa in enumerate(seq):
        try:
            matrix[i][alphabet[aa]] = 1
        except:
            pass
    return matrix

def pdb2coords(pdb_path, read_chain = 'A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_path)
    coords = []
    seq = ''
    for model in structure:
        for chain in model:
            if chain.id == read_chain:
                for residue in chain:
                    if not Polypeptide.is_aa(residue, standard = True):
                        continue
                    res_coords = torch.zeros((4,3))
                    try:
                        res_coords[0,:] = torch.from_numpy(residue['N'].get_vector().get_array())
                        res_coords[1,:] = torch.from_numpy(residue['CA'].get_vector().get_array())
                        res_coords[2,:] = torch.from_numpy(residue['C'].get_vector().get_array())
                        try:
                            res_coords[3,:] = torch.from_numpy(residue['O'].get_vector().get_array())
                        except:
                            res_coords[3,:] = torch.from_numpy(residue['OG'].get_vector().get_array()) 
                        coords.append(res_coords)
                        seq += Polypeptide.three_to_one(residue.resname)
                    except:
                        continue
        break
    coords = torch.stack(coords, axis = 0)
    return coords, seq

class CATH_feats(D.Dataset):
    '''
    Dataset class for CATH data processed by https://arxiv.org/abs/2009.01411 this will load the train/val/test set with coordinates
    '''

    def __init__(self, partition='train', mask_rate = 0, seq = False):
        #self.chain_dict = json.load(open('/'.join(dir_path.split('/')[:-2]) + '/data/CATH4.2/' + 'chain_set_splits.json'))
        self.coord_dict = dict()
        self.mask = True if mask_rate>0 else False
        self.mask_rate = mask_rate
        self.seq = seq  
       
        #self.chain_list = self.chain_dict[partition]

    def __getitem__(self,index):
        chain_id = self.chain_list[index]
        sample = torch.load('/'.join(dir_path.split('/')[:-2]) + '/data/CATH4.2/seq_feats/'+ chain_id + '.pt')
        valid_index = self.select_valid(sample)
        seq = sample['seq']
        seq = ''.join([seq[i] for i in valid_index])
        seq_feat = seq2onehot(seq)
        
        C_coords = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['C']))).T.float()
        N_coords = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['N']))).T.float()
        CA_coords  = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['CA']))).T.float()
        O_coords  = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['O']))).T.float()
        
        _backbone_coords = torch.index_select(torch.stack([N_coords, CA_coords, C_coords, O_coords]).permute(2,0,1), dim = 0, index = valid_index)
        coord_mask = torch.ones_like(_backbone_coords)
        seq_mask = torch.ones_like(seq_feat)
        if self.mask:
            if np.random.choice([1,0],p = [self.mask_rate, 1 - self.mask_rate]):
                mask_mode = np.random.choice([1,2,3])
                mask_mode = 3
                if mask_mode == 1: # linear mask            
                    mask_len = 10 #int(np.random.uniform(0.1, 0.15) * len(seq))
                    mask_index = np.random.choice(range(0, len(seq) - mask_len - 1))
                    coord_mask[mask_index:mask_index+mask_len, :, :] = 0
                    seq_mask[mask_index:mask_index+mask_len, :] = 0
                elif mask_mode == 2: #spatial mask
                    radius = 12
                    max_neighbor = 60
                    seed_index = np.random.choice(range(0, len(seq)))
                    dist_map = torch.cdist(_backbone_coords[:,1,:], _backbone_coords[:,1,:])
                    neighbors = (dist_map[seed_index,:] < radius).nonzero()
                    if len(neighbors) > 60:
                        neighbors = neighbors[:60]
                    coord_mask[neighbors,:,:] = 0
                    seq_mask[neighbors,:] = 0
                elif mask_mode == 3: # random mask
                    mask_prob = np.random.uniform(0,0.5, len(seq))
                    mask_idx = torch.tensor([np.random.choice([0,1], p =[1-p, p]) for p in mask_prob]).nonzero()
                    coord_mask[mask_idx,:,:] = 0
                    seq_mask[mask_idx,:] = 0

        backbone_coords = _backbone_coords * coord_mask
        _seq_feat = seq_mask * seq_feat

        edge_feat = self.get_edge_feat(backbone_coords[:,1,:].T)
        scalar_feat = self.get_torsion_embeddings(backbone_coords[:,0,:].T, backbone_coords[:,1,:].T, backbone_coords[:,2,:].T)
        coords_feat = self.get_coords_feat(backbone_coords)
        CB_coords = torch.nan_to_num(impute_beta_carbon(backbone_coords[:,:3,:]))
        oris = torch.stack(get_tr_rosetta_orientation_mats(backbone_coords[:,0,:], backbone_coords[:,1,:], CB_coords), dim = -1)
        edge_feat = torch.cat([edge_feat, oris], dim = -1)
        if self.seq:
            scalar_feat = torch.cat([scalar_feat, _seq_feat], dim=-1)
        return _backbone_coords, scalar_feat, edge_feat, coords_feat, sample['name'], seq#sample['CATH'][0].split('.')[0] #sample['name'] #seq_mask#sample['CATH'][0].split('.')[0]

    def load_id(self, chain_id):
        sample = torch.load('/'.join(dir_path.split('/')[:-2]) + '/data/CATH4.2/seq_feats/'+ chain_id + '.pt')
        valid_index = self.select_valid(sample)
        seq = sample['seq']
        seq = ''.join([seq[i] for i in valid_index])
        seq_feat = seq2onehot(seq)
        
        C_coords = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['C']))).T.float()
        N_coords = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['N']))).T.float()
        CA_coords  = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['CA']))).T.float()
        O_coords  = torch.from_numpy(np.nan_to_num(np.array(sample['coords']['O']))).T.float()
        
        _backbone_coords = torch.index_select(torch.stack([N_coords, CA_coords, C_coords, O_coords]).permute(2,0,1), dim = 0, index = valid_index)
        coord_mask = torch.ones_like(_backbone_coords)
        seq_mask = torch.ones_like(seq_feat)
        if self.mask:
            if np.random.choice([1,0],p = [self.mask_rate, 1 - self.mask_rate]):
                mask_mode = np.random.choice([1,2,3])
                mask_mode = 3
                if mask_mode == 1: # linear mask            
                    mask_len = 10 #int(np.random.uniform(0.1, 0.15) * len(seq))
                    mask_index = np.random.choice(range(0, len(seq) - mask_len - 1))
                    coord_mask[mask_index:mask_index+mask_len, :, :] = 0
                    seq_mask[mask_index:mask_index+mask_len, :] = 0
                elif mask_mode == 2: #spatial mask
                    radius = 12
                    max_neighbor = 60
                    seed_index = np.random.choice(range(0, len(seq)))
                    dist_map = torch.cdist(_backbone_coords[:,1,:], _backbone_coords[:,1,:])
                    neighbors = (dist_map[seed_index,:] < radius).nonzero()
                    if len(neighbors) > 60:
                        neighbors = neighbors[:60]
                    coord_mask[neighbors,:,:] = 0
                    seq_mask[neighbors,:] = 0
                elif mask_mode == 3: # random mask
                    mask_prob = np.random.uniform(0,0.5, len(seq))
                    mask_idx = torch.tensor([np.random.choice([0,1], p =[1-p, p]) for p in mask_prob]).nonzero()
                    coord_mask[mask_idx,:,:] = 0
                    seq_mask[mask_idx,:] = 0

        backbone_coords = _backbone_coords * coord_mask
        _seq_feat = seq_mask * seq_feat

        edge_feat = self.get_edge_feat(backbone_coords[:,1,:].T)
        scalar_feat = self.get_torsion_embeddings(backbone_coords[:,0,:].T, backbone_coords[:,1,:].T, backbone_coords[:,2,:].T)
        coords_feat = self.get_coords_feat(backbone_coords)
        CB_coords = torch.nan_to_num(impute_beta_carbon(backbone_coords[:,:3,:]))
        oris = torch.stack(get_tr_rosetta_orientation_mats(backbone_coords[:,0,:], backbone_coords[:,1,:], CB_coords), dim = -1)
        edge_feat = torch.cat([edge_feat, oris], dim = -1)
        if self.seq:
            scalar_feat = torch.cat([scalar_feat, _seq_feat], dim=-1)
        return _backbone_coords, scalar_feat, edge_feat, coords_feat, sample['name'], seq

    def get_constraints(self, backbone_coords, seq, constraints):
        edge_feat = self.get_edge_feat(backbone_coords[:,1,:].T)
        scalar_feat = self.get_torsion_embeddings(backbone_coords[:,0,:].T, backbone_coords[:,1,:].T, backbone_coords[:,2,:].T)
        coords_feat = self.get_coords_feat(backbone_coords)
        CB_coords = torch.nan_to_num(impute_beta_carbon(backbone_coords[:,:3,:]))
        oris = torch.stack(get_tr_rosetta_orientation_mats(backbone_coords[:,0,:], backbone_coords[:,1,:], CB_coords), dim = -1)
        edge_feat = torch.cat([edge_feat, oris], dim = -1)
        seq_feat = seq2onehot(seq).to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
        scalar_feat = torch.cat([scalar_feat, seq_feat], dim=-1)
        
        pair_mask = torch.zeros_like(edge_feat).to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
        scalar_mask = torch.zeros_like(scalar_feat).to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
        helix_index = constraints['H']
        sheet_index = constraints['E']
        for (s1,e1) in helix_index:
            for (s2,e2) in helix_index:
                pair_mask[s1:e1,s2:e2,:] = 1
                scalar_mask[s1:e1,:] = 1
            for (s2,e2) in sheet_index:
                #pair_mask[s1:e1,s2:e2,:] = 1
                #pair_mask[s2:e2,s1:e1,:] = 1
                scalar_mask[s1:e1,:] = 1

        
        for (s1,e1) in sheet_index:
            for (s2,e2) in sheet_index:               
                pair_mask[s1:e1,s2:e2,:] = 1
                scalar_mask[s1:e1,:] = 1

        return (1-pair_mask, 1-scalar_mask), (edge_feat * pair_mask, scalar_mask * scalar_mask)
        
    def to_feats(self, backbone_coords, seq, constraints = None):
        edge_feat = self.get_edge_feat(backbone_coords[:,1,:].T)
        scalar_feat = self.get_torsion_embeddings(backbone_coords[:,0,:].T, backbone_coords[:,1,:].T, backbone_coords[:,2,:].T)
        coords_feat = self.get_coords_feat(backbone_coords)
        CB_coords = torch.nan_to_num(impute_beta_carbon(backbone_coords[:,:3,:]))
        oris = torch.stack(get_tr_rosetta_orientation_mats(backbone_coords[:,0,:], backbone_coords[:,1,:], CB_coords), dim = -1)
        edge_feat = torch.cat([edge_feat, oris], dim = -1)
        seq_feat = seq2onehot(seq).to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
        scalar_feat = torch.cat([scalar_feat, seq_feat], dim=-1)

        if constraints:
            edge_feat = constraints[0][0].to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device()) * edge_feat + constraints[1][0].to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
            scalar_feat = constraints[0][1].to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device()) * scalar_feat + constraints[1][1].to('cpu' if backbone_coords.get_device() <0 else backbone_coords.get_device())
        

        return scalar_feat, edge_feat

    def get_coords_feat(self, bb_coords):
        return torch.stack(
            [
                bb_coords[:,0,:] - bb_coords[:, 1, :],
                bb_coords[:,1,:] - bb_coords[:, 1, :],
                bb_coords[:,2,:] - bb_coords[:, 1, :],
                bb_coords[:,3,:] - bb_coords[:, 1, :]
            ]
        ).permute(1,0,2)

    def get_edge_feat(self, CA_coords):
        D_min=0.
        D_max=20.
        D_count=16
        dist = torch.cdist(CA_coords.T, CA_coords.T)
        D_mu = torch.linspace(D_min, D_max, D_count).to('cpu' if CA_coords.get_device() <0 else CA_coords.get_device())
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(dist, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF
    
    def get_torsion_embeddings(self, N_coords, CA_coords, C_coords):
        torsions = list(calc_torsion.get_bb_dihedral(N_coords.T, CA_coords.T, C_coords.T))
        torsion_embed = [
            torch.sin(torsions[0]),torch.cos(torsions[0]),
            torch.sin(torsions[1]),torch.cos(torsions[1]),
            torch.sin(torsions[2]),torch.cos(torsions[2])
        ]
        return torch.stack(torsion_embed).T
    
    def select_valid(self, sample):
        invalid_index1 = []
        invalid_index2 = []
        invalid_index3 = []
        invalid_index4 = []
        for i in range(len(sample['coords']['C'])):
            if np.isnan(sample['coords']['C'][i]).all():
               invalid_index1.append(i)
            if np.isnan(sample['coords']['N'][i]).all():
               invalid_index2.append(i)
            if np.isnan(sample['coords']['CA'][i]).all():
               invalid_index3.append(i)
            if np.isnan(sample['coords']['O'][i]).all():
               invalid_index4.append(i)
        invalid_index = set(invalid_index1) | set(invalid_index2) | set(invalid_index3) | set(invalid_index4)
        valid_index = list(set(list(range(len(sample['coords']['C'])))) - invalid_index)
        return torch.tensor(valid_index)
    
    def feat_from_pdb(self, pdb_path, chain = 'A'):
        backbone_coords, seq = pdb2coords(pdb_path, read_chain = chain) 
        seq_feat = seq2onehot(seq)

        coord_mask = torch.ones_like(backbone_coords)
        seq_mask = torch.ones_like(seq_feat)

        _seq_feat = seq_mask * seq_feat

        edge_feat = self.get_edge_feat(backbone_coords[:,1,:].T)
        scalar_feat = self.get_torsion_embeddings(backbone_coords[:,0,:].T, backbone_coords[:,1,:].T, backbone_coords[:,2,:].T)
        CB_coords = torch.nan_to_num(impute_beta_carbon(backbone_coords[:,:3,:]))
        oris = torch.stack(get_tr_rosetta_orientation_mats(backbone_coords[:,0,:], backbone_coords[:,1,:], CB_coords), dim = -1)
        edge_feat = torch.cat([edge_feat, oris], dim = -1)
        if self.seq:
            scalar_feat = torch.cat([scalar_feat, _seq_feat], dim=-1)

        return backbone_coords, scalar_feat, edge_feat, seq 

    def __len__(self):
        return len(self.chain_list)


if __name__ == '__main__':
    dset = CATH_feats(partition = 'test')
    loader = D.DataLoader(dset, num_workers = 5, batch_size = 1)
    for i in loader:
        for j in i:
            print(j.size())
        break
        
        
    
