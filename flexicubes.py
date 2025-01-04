# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from .tables import *
from kaolin.utils.testing import check_tensor

__all__ = [
    'FlexiCubes'
]


class BitCounter:
    def __init__(self, device="cuda"):
        # Pre-compute lookup table for all possible 8-bit values
        self.lookup = torch.tensor([bin(i).count('1') for i in range(256)], 
                                 dtype=torch.uint8, 
                                 device=device)
    def __call__(self, tensor):
        """
        Count bits in a uint8 tensor efficiently using lookup table.
        Args:
            tensor: torch.Tensor of dtype torch.uint8
        Returns:
            torch.Tensor with same shape as input containing bit counts
        """
        if tensor.dtype != torch.uint8:
            raise ValueError("Input tensor must be torch.uint8")
        # Ensure tensor values don't exceed 255 by masking
        tensor = tensor & 0xFF
        #cast to long, becasue PyTorch interprets uint8 tensors as boolean masks when indexing.
        #long needed for lookup. Don't worry The lookup table is tiny (just 256 entries)
        return self.lookup[tensor.long()] 
    

class FlexiCubes:
    def __init__(self, device="cuda"):

        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.long, device=device, requires_grad=False)
        self.num_vd_table = torch.tensor(num_vd_table,
                                         dtype=torch.long, device=device, requires_grad=False)
        self.check_table = torch.tensor(
            check_table,
            dtype=torch.long, device=device, requires_grad=False)

        self.tet_table = torch.tensor(tet_table, dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_train = torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device, requires_grad=False)

        self.cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
                                         1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float, device=device)
        self.cube_corners_idx = torch.pow(2, torch.arange(8, requires_grad=False))
        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, device=device, requires_grad=False)

        self.edge_dir_table = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1],
                                           dtype=torch.long, device=device)
        self.dir_faces_table = torch.tensor([
            [[5, 4], [3, 2], [4, 5], [2, 3]],
            [[5, 4], [1, 0], [4, 5], [0, 1]],
            [[3, 2], [1, 0], [2, 3], [0, 1]]
        ], dtype=torch.long, device=device)
        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device)
        self.bit_counter = BitCounter(device)

    def __call__(self, voxelgrid_vertices, scalar_field, cube_idx, resolution, qef_reg_scale=1e-3,
                 weight_scale=0.99, beta=None, alpha=None, gamma_f=None, voxelgrid_colors=None, training=False,
                 dtype=None): #dtype so that it can work with half precision.
        """
        Optionally specify 'dtype' to unify all float inputs (e.g. torch.float16).
        If 'dtype' is None, we infer from voxelgrid_vertices or default to float32.
        """
        # unify floating tensors to a chosen dtype, so it can work with half, etc:
        if dtype is None:
            if voxelgrid_vertices.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                dtype = voxelgrid_vertices.dtype
            else:
                dtype = torch.float32
        #cast all to dtype, so that it works with half-precision:
        voxelgrid_vertices = voxelgrid_vertices.to(dtype)  # Cast the geometry to the chosen dtype
        scalar_field = scalar_field.to(dtype)
        if beta is not None:
            beta = beta.to(dtype)
        if alpha is not None:
            alpha = alpha.to(dtype)
        if gamma_f is not None:
            gamma_f = gamma_f.to(dtype)
        if voxelgrid_colors is not None:
            voxelgrid_colors = voxelgrid_colors.to(dtype)

        assert torch.is_tensor(voxelgrid_vertices) and \
            check_tensor(voxelgrid_vertices, (None, 3), throw=False), \
            "'voxelgrid_vertices' should be a tensor of shape (num_vertices, 3)"
        num_vertices = voxelgrid_vertices.shape[0]
        assert torch.is_tensor(scalar_field) and \
            check_tensor(scalar_field, (num_vertices,), throw=False), \
            "'scalar_field' should be a tensor of shape (num_vertices,)"
        assert torch.is_tensor(cube_idx) and \
            check_tensor(cube_idx, (None, 8), throw=False), \
            "'cube_idx' should be a tensor of shape (num_cubes, 8)"
        num_cubes = cube_idx.shape[0]
        assert beta is None or (
            torch.is_tensor(beta) and
            check_tensor(beta, (num_cubes, 12), throw=False)
        ), "'beta' should be a tensor of shape (num_cubes, 12)"
        assert alpha is None or (
            torch.is_tensor(alpha) and
            check_tensor(alpha, (num_cubes, 8), throw=False)
        ), "'alpha' should be a tensor of shape (num_cubes, 8)"
        assert gamma_f is None or (
            torch.is_tensor(gamma_f) and
            check_tensor(gamma_f, (num_cubes,), throw=False)
        ), "'gamma_f' should be a tensor of shape (num_cubes,)"

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device, dtype=dtype),
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros((0), device=self.device, dtype=dtype),
                torch.zeros((0, voxelgrid_colors.shape[-1]), device=self.device, dtype=dtype) if voxelgrid_colors is not None else None
            )
        beta, alpha, gamma_f = self._normalize_weights(
            beta, alpha, gamma_f, surf_cubes, weight_scale)
        
        if voxelgrid_colors is not None:
            voxelgrid_colors = torch.sigmoid(voxelgrid_colors)

        case_ids = self._get_case_id(occ_fx8, surf_cubes, resolution)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        vd, L_dev, vd_gamma, vd_idx_map, vd_color = self._compute_vd(
            voxelgrid_vertices, cube_idx[surf_cubes], surf_edges, scalar_field,
            case_ids, surf_cubes, beta, alpha, gamma_f, idx_map, qef_reg_scale, voxelgrid_colors)
        vertices, faces, s_edges, edge_indices, vertices_color = self._triangulate(
            scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask, training, vd_color)
        return vertices, faces, L_dev, vertices_color

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad


    def _normalize_weights(self, beta, alpha, gamma_f, surf_cubes, weight_scale, chunk_size=100_000):
        """
        Normalizes the given weights to be non-negative, processing in chunks to reduce memory usage.
        """
        n_cubes = surf_cubes.shape[0]
        device = self.device
        dtype = beta.dtype if beta is not None else torch.float

        # Initialize output tensors for surf_cubes only
        surf_count = surf_cubes.sum().item()
        beta_out = torch.empty((surf_count, 12), dtype=dtype, device=device)
        alpha_out = torch.empty((surf_count, 8), dtype=dtype, device=device)
        gamma_out = torch.empty((surf_count,), dtype=dtype, device=device)
        
        # Get indices of surface cubes
        surf_indices = torch.nonzero(surf_cubes, as_tuple=False).squeeze(1)
        
        # Process in chunks
        start_idx = 0
        for chunk_start in range(0, surf_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, surf_count)
            chunk_indices = surf_indices[chunk_start:chunk_end]
            
            # Process beta
            if beta is not None:
                beta_chunk = beta[chunk_indices]
                beta_chunk = (torch.tanh(beta_chunk) * weight_scale + 1)
                beta_out[chunk_start:chunk_end] = beta_chunk
            else:
                beta_out[chunk_start:chunk_end] = 1
                
            # Process alpha
            if alpha is not None:
                alpha_chunk = alpha[chunk_indices]
                alpha_chunk = (torch.tanh(alpha_chunk) * weight_scale + 1)
                alpha_out[chunk_start:chunk_end] = alpha_chunk
            else:
                alpha_out[chunk_start:chunk_end] = 1
                
            # Process gamma_f
            if gamma_f is not None:
                gamma_chunk = gamma_f[chunk_indices]
                gamma_chunk = torch.sigmoid(gamma_chunk) * weight_scale + (1 - weight_scale) / 2
                gamma_out[chunk_start:chunk_end] = gamma_chunk
            else:
                gamma_out[chunk_start:chunk_end] = 1
                
            # Free memory explicitly
            if 'beta_chunk' in locals(): del beta_chunk
            if 'alpha_chunk' in locals(): del alpha_chunk
            if 'gamma_chunk' in locals(): del gamma_chunk
            torch.cuda.empty_cache()
            
        return beta_out, alpha_out, gamma_out
    

    @torch.no_grad()
    def _get_case_id(self, occ_fx8_bits, surf_cubes, res):
        """
        Instead of building problem_config_full, we store the ambiguous config
        in a single 1D buffer with chunked processing for memory efficiency.
        """
        device = occ_fx8_bits.device
        num_cubes = surf_cubes.size(0)
        rx, ry, rz = res if isinstance(res, (list, tuple)) else (res, res, res)

        # Step 1: compute case_ids from occ_fx8_bits
        case_ids = torch.zeros((num_cubes,), dtype=torch.long, device=device)
        case_ids[surf_cubes] = (occ_fx8_bits[surf_cubes].long() * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)

        # Step 2: read problem_config for each cube in chunks
        chunk_size = 50_000
        problem_cfg = torch.zeros((num_cubes, 5), dtype=torch.int16, device=device)
        
        for start in range(0, num_cubes, chunk_size):
            end = min(start + chunk_size, num_cubes)
            chunk_case_ids = case_ids[start:end]
            chunk_cfg = self.check_table[chunk_case_ids]
            problem_cfg[start:end] = chunk_cfg
            del chunk_cfg
            torch.cuda.empty_cache()

        # Step 3: collect ambiguous cubes
        is_ambig = (problem_cfg[:, 0] == 1)

        # Step 4: precompute coordinates
        rx_t = torch.arange(rx, device=device, dtype=torch.int32)
        ry_t = torch.arange(ry, device=device, dtype=torch.int32)
        rz_t = torch.arange(rz, device=device, dtype=torch.int32)
        
        # Generate coordinates in chunks to save memory
        x_flat = torch.zeros(num_cubes, dtype=torch.int32, device=device)
        y_flat = torch.zeros(num_cubes, dtype=torch.int32, device=device)
        z_flat = torch.zeros(num_cubes, dtype=torch.int32, device=device)
        
        spatial_chunk_size = min(32, rx, ry, rz)  # Process in smaller spatial blocks
        for x in range(0, rx, spatial_chunk_size):
            x_end = min(x + spatial_chunk_size, rx)
            for y in range(0, ry, spatial_chunk_size):
                y_end = min(y + spatial_chunk_size, ry)
                for z in range(0, rz, spatial_chunk_size):
                    z_end = min(z + spatial_chunk_size, rz)
                    
                    X, Y, Z = torch.meshgrid(
                        rx_t[x:x_end],
                        ry_t[y:y_end],
                        rz_t[z:z_end],
                        indexing='ij'
                    )
                    
                    # Calculate start index in flattened array
                    start_idx = (x * ry * rz + y * rz + z)
                    chunk_size = (x_end-x) * (y_end-y) * (z_end-z)
                    end_idx = start_idx + chunk_size
                    
                    x_flat[start_idx:end_idx] = X.reshape(-1)
                    y_flat[start_idx:end_idx] = Y.reshape(-1)
                    z_flat[start_idx:end_idx] = Z.reshape(-1)
                    
                    del X, Y, Z
                    torch.cuda.empty_cache()

        # Step 5: Process ambiguous cubes in chunks
        ambig_idx = torch.nonzero(is_ambig, as_tuple=False).squeeze(1)
        chunk_size = 50_000
        
        for start in range(0, ambig_idx.numel(), chunk_size):
            end = start + chunk_size
            c_idx = ambig_idx[start:end]

            c_cfg = problem_cfg[c_idx, :]
            dx = c_cfg[:, 1].to(torch.int32)
            dy = c_cfg[:, 2].to(torch.int32)
            dz = c_cfg[:, 3].to(torch.int32)

            cx = x_flat[c_idx]
            cy = y_flat[c_idx]
            cz = z_flat[c_idx]

            nx = cx + dx
            ny = cy + dy
            nz = cz + dz

            in_range = (nx >= 0) & (nx < rx) & \
                    (ny >= 0) & (ny < ry) & \
                    (nz >= 0) & (nz < rz)
            
            if not in_range.any():
                del c_cfg, dx, dy, dz, cx, cy, cz, nx, ny, nz
                torch.cuda.empty_cache()
                continue

            good = in_range.nonzero(as_tuple=False).squeeze(1)
            
            # Calculate neighbor indices carefully to avoid overflow
            n_idx = (nx[good].long() + 
                    ny[good].long() * rx + 
                    nz[good].long() * (rx * ry))
            this_idx = c_idx[good]

            both_ambig = (problem_cfg[n_idx, 0] == 1)
            to_invert = good[both_ambig]

            if to_invert.numel() > 0:
                new_id = problem_cfg[this_idx[to_invert], -1]
                n_new_id = problem_cfg[n_idx[to_invert], -1]
                
                case_ids[this_idx[to_invert]] = new_id
                case_ids[n_idx[to_invert]] = n_new_id

            del c_cfg, dx, dy, dz, cx, cy, cz, nx, ny, nz, n_idx
            torch.cuda.empty_cache()

        return case_ids


    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=cube_idx.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_idx.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx, chunk_size=100_000):
        """
        Identifies grid cubes that intersect the surface by checking if corners are not all the same sign.
        Bit-packs 8 corner sign bits into one uint8 per cube for memory efficiency.
        """
        device = scalar_field.device
        num_cubes = cube_idx.shape[0]
        
        all_occ_bits = []
        all_surf = []
        
        for start in range(0, num_cubes, chunk_size):
            end = start + chunk_size
            chunk = cube_idx[start:end]  # shape [B, 8]
            
            # Get signs for all corners
            chunk_occ_n = (scalar_field[chunk.reshape(-1)] < 0)
            
            # Pack 8 sign bits into uint8, making sure to stay within uint8 range
            chunk_bits = torch.zeros((chunk.shape[0],), dtype=torch.uint8, device=device)
            for i in range(8):  # we have exactly 8 corners
                # Ensure each bit operation stays within uint8 range
                corner_mask = chunk_occ_n[i::8].to(torch.uint8) & 1  # ensure it's 0 or 1
                chunk_bits = chunk_bits | (corner_mask << i)  # shift and OR
                
            # Count bits using lookup table
            chunk_count = self.bit_counter(chunk_bits)
            
            # Surface exists if some but not all corners are negative
            chunk_surf = (chunk_count > 0) & (chunk_count < 8)
            
            all_occ_bits.append(chunk_bits)
            all_surf.append(chunk_surf)
        
        occ_fx8_bits = torch.cat(all_occ_bits, dim=0)
        surf_cubes = torch.cat(all_surf, dim=0)
        
        return surf_cubes, occ_fx8_bits

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -
                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)]
                                 , edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _solve_vd_QEF(self, p_bxnx3, norm_bxnx3, c_bx3, qef_reg_scale):
        p_bxnx3 = p_bxnx3.reshape(-1, 7, 3)
        norm_bxnx3 = norm_bxnx3.reshape(-1, 7, 3)
        c_bx3 = c_bx3.reshape(-1, 3)
        A = norm_bxnx3
        B = ((p_bxnx3) * norm_bxnx3).sum(-1, keepdims=True)

        A_reg = (torch.eye(3, device=p_bxnx3.device) * qef_reg_scale).unsqueeze(0).repeat(p_bxnx3.shape[0], 1, 1)
        B_reg = (qef_reg_scale * c_bx3).unsqueeze(-1)
        A = torch.cat([A, A_reg], 1)
        B = torch.cat([B, B_reg], 1)
        dual_verts = torch.linalg.lstsq(A, B).solution.squeeze(-1)
        return dual_verts

    def _compute_vd(
        self,
        voxelgrid_vertices,   # shape [num_surf_vertices, 3]
        cube_idx_surf,       # shape [num_surf_cubes, 8] => from cube_idx[surf_cubes]
        surf_edges,          # shape [num_surf_edges, 2]
        scalar_field,        # shape [num_vertices]
        case_ids,            # shape [num_cubes]
        surf_cubes,          # bool mask of shape [num_cubes]
        beta, alpha, gamma_f,# shape [num_surf_cubes, 12], [num_surf_cubes,8], [num_surf_cubes], respectively
        idx_map,             # shape [num_surf_cubes, 12]
        qef_reg_scale,
        voxelgrid_colors
    ):
        """
        A revised version that only allocates 'vd_idx_map_surf' for the surface cubes.
        """
        device = self.device
        dtype = beta.dtype  # e.g. float16 or float32

        # Number of surface cubes
        num_surf_cubes = cube_idx_surf.shape[0]
        # Extract the case IDs just for those cubes
        case_ids_surf = case_ids[surf_cubes]  # shape [num_surf_cubes]

        # index into self.num_vd_table for those cubes
        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids_surf, dim=0).int()
        # or do: num_vd = self.num_vd_table[case_ids_surf].int()

        # shape [num_surf_cubes, 12], storing dual-vertex index or -1 if not used
        vd_idx_map_surf = torch.full(
            (num_surf_cubes, 12),
            fill_value=-1,
            dtype=torch.int32,     # 32-bit is usually enough
            device=device,
            requires_grad=False
        )

        # We’ll build big lists for edge_group, edge_group_to_vd, etc.
        edge_group = []
        edge_group_to_vd = []
        edge_group_to_cube = []
        vd_num_edges = []
        vd_gamma = []

        # We'll do an integer total counter of how many dual vertices
        total_num_vd = 0

        # If you have many unique values in 'num_vd', you can chunk them:
        unique_v = torch.unique(num_vd)  # typical range 0..12
        # Usually that's small, so we won't chunk further. But if needed, you can chunk here.

        for val in unique_v:
            if val < 1:
                continue  # skip if no vertices generated
            # Which cubes have 'num_vd' == val?
            # shape [num_surf_cubes] boolean
            cur_cubes_mask = (num_vd == val)
            # Optionally convert to indices:
            cur_cubes_idx = torch.nonzero(cur_cubes_mask, as_tuple=False).squeeze(1)
            if cur_cubes_idx.numel() == 0:
                continue

            # how many dual verts are we generating for these cubes?
            #  => #cubes_in_group * val
            curr_num_vd = cur_cubes_idx.numel() * val

            # gather the DMC edges for these cubes
            # shape [N, val*7], where N = number of cubes in cur_cubes_idx
            # we do self.dmc_table[case_ids_surf[cur_cubes_idx], :val]
            dmc_subset = torch.index_select(
                input=self.dmc_table, 
                index=case_ids_surf[cur_cubes_idx].long(),  # cast to long if needed
                dim=0
            )[:, :val]
            dmc_subset = dmc_subset.reshape(-1, val*7)  # shape [N, val*7]

            # We'll create a range [total_num_vd .. total_num_vd + curr_num_vd -1]
            # shape [curr_num_vd, 1] => each repeated 7 times
            group_to_vd = torch.arange(curr_num_vd, device=device, dtype=torch.int32)
            group_to_vd = group_to_vd.unsqueeze(-1).repeat(1, 7)  # shape [curr_num_vd, 7]

            # Now, we need to map from "which cubes" to "which entry" in vd_idx_map_surf
            # We'll build a repeated version as well
            # shape [N, val*7] => we want to flatten => length N*val*7 => same as group_to_vd
            # but first let's shape out how many cubes we have:
            N = cur_cubes_idx.numel()
            # We'll replicate each cube index val*7 times
            cubes_repeated = torch.arange(N, device=device, dtype=torch.int32).unsqueeze(-1)
            cubes_repeated = cubes_repeated.repeat(1, val*7).reshape(-1)
            # Then we add offset 'start_in_global = total_num_vd' for group_to_vd

            # We also need the mask for edges != -1
            mask = (dmc_subset != -1)

            # Now flatten dmc_subset => shape [N*val*7]
            dmc_flat = dmc_subset.reshape(-1)
            group_to_vd_flat = group_to_vd.reshape(-1)

            # We want to keep only those with != -1
            edge_group.append(torch.masked_select(dmc_flat, mask.reshape(-1)))
            edge_group_to_vd.append(torch.masked_select(group_to_vd_flat, mask.reshape(-1)))

            # For "edge_group_to_cube", we need the "global cube index" in [0..num_surf_cubes)
            # i.e. cur_cubes_idx[ c ] => c in [0..N-1]
            # but repeated val*7 times
            cubes_indexed = torch.index_select(cur_cubes_idx, 0, cubes_repeated)
            edge_group_to_cube.append(torch.masked_select(cubes_indexed, mask.reshape(-1)))

            # Summation for each dual vertex => how many edges => we'll do that later
            # Or we collect them as in your original code
            # For L_dev, we also store gamma
            gamma_chunk = torch.index_select(gamma_f, 0, cur_cubes_idx)
            # shape [N] => repeat it val times => shape [N*val]
            gamma_chunk = gamma_chunk.unsqueeze(-1).repeat(1, val).reshape(-1)
            # Now we have gamma for each dual vertex => shape [N*val]
            # But each dual vertex corresponds to 7 edges => we store the same gamma repeated 7 times
            gamma_chunk_7 = gamma_chunk.unsqueeze(-1).repeat(1,7).reshape(-1)
            # We'll mask it
            vd_gamma.append(torch.masked_select(gamma_chunk_7, mask.reshape(-1)))

            # We'll build vd_num_edges later (like in your code).
            # We'll store that mask in a shape [N, val, 7] => sum => how many edges per dual vertex
            mask_3d = mask.reshape(N, val, 7)
            # sum along last dim => shape [N, val]
            edges_count_per_vd = mask_3d.sum(dim=-1, keepdim=False)  # [N, val]
            vd_num_edges.append(edges_count_per_vd.reshape(-1))  # [N*val]

            # Finally update total
            total_num_vd += curr_num_vd

            # free memory
            del dmc_subset, group_to_vd, group_to_vd_flat, cubes_repeated
            del mask, dmc_flat, gamma_chunk, gamma_chunk_7, mask_3d, edges_count_per_vd
            torch.cuda.empty_cache()

        # Now we cat all partial lists
        edge_group = torch.cat(edge_group, dim=0)                # shape ~ [sum_all_edges]
        edge_group_to_vd = torch.cat(edge_group_to_vd, dim=0)    # same shape
        edge_group_to_cube = torch.cat(edge_group_to_cube, dim=0)# same shape
        vd_gamma = torch.cat(vd_gamma, dim=0)
        vd_num_edges = torch.cat(vd_num_edges, dim=0)

        # Next we allocate the final arrays:
        vd = torch.zeros((total_num_vd, 3), device=device, dtype=dtype)
        beta_sum = torch.zeros((total_num_vd, 1), device=device, dtype=dtype)

        # Also define a color buffer if needed:
        if voxelgrid_colors is not None:
            C = voxelgrid_colors.shape[-1]
            vd_color = torch.zeros((total_num_vd, C), device=device, dtype=dtype)
        else:
            vd_color = None

        # We'll also build a final index map for surface cubes => shape [num_surf_cubes,12]
        # Start it at -1
        vd_idx_map_surf = torch.full(
            (num_surf_cubes, 12),
            fill_value=-1,
            dtype=torch.int32,
            device=device
        )

        # idx_map is shape [num_surf_cubes,12], from _identify_surf_edges
        # We flatten for easier gather
        idx_map_flat = idx_map.reshape(-1)  # shape [num_surf_cubes*12]

        # Now we gather from that
        #   index = edge_group_to_cube * 12 + edge_group
        #   => where edge_group is in [0..11], edge_group_to_cube is in [0..num_surf_cubes-1]
        # We'll do the gather of idx_map_flat:
        gather_index = edge_group_to_cube * 12 + edge_group  # shape [sum_all_edges]
        idx_group = torch.gather(idx_map_flat, dim=0, index=gather_index.long())  # shape [sum_all_edges]

        # Create the edge coordinates and scalar values
        surf_edges_x = torch.index_select(
            input=voxelgrid_vertices, 
            index=surf_edges.reshape(-1), 
            dim=0
        ).reshape(-1, 2, 3)  # shape [num_surf_edges, 2, 3]

        surf_edges_s = torch.index_select(
            input=scalar_field, 
            index=surf_edges.reshape(-1), 
            dim=0
        ).reshape(-1, 2, 1)  # shape [num_surf_edges, 2, 1]

        # Next we do linear interpolation:
        #   surf_edges_x, surf_edges_s, etc. shape [N_edges, ...]
        # We pick out by idx_group
        #   x_group => shape [sum_all_edges, 2, 3]
        x_group = torch.index_select(surf_edges_x, 0, idx_group)
        s_group = torch.index_select(surf_edges_s, 0, idx_group)

        # Then we do zero_crossing => shape [N_edges, 3]
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)  
        # but we also need the subset
        zero_crossing_group = torch.index_select(zero_crossing, 0, idx_group)

        # alpha starts as [num_surf_cubes, 8] (one value per corner)
        # cube_edges is [24] containing pairs of indices (12 edges × 2 endpoints)
        # First expand alpha using cube_edges to get values for both endpoints of each edge
        alpha_nx12x2 = torch.index_select(
            input=alpha,
            index=self.cube_edges,  # This maps from 8 corners to 24 values (12 edges × 2 endpoints)
            dim=1
        ).reshape(-1, 12, 2)  # Reshape to [num_surf_cubes, 12, 2]

        # Now flatten to [num_surf_cubes * 12, 2]
        alpha_nx12x2_flat = alpha_nx12x2.reshape(-1, 2)

        # gather_index selects the specific edges we need
        # it has shape [sum_all_edges] where each value is in range [0, num_surf_cubes * 12)
        alpha_group = torch.index_select(
            input=alpha_nx12x2_flat,
            index=gather_index.long(),
            dim=0
        ).reshape(-1, 2, 1)  # Final shape: [sum_all_edges, 2, 1]

        # Calculate ue_group using linear interpolation
        ue_group = self._linear_interp(s_group * alpha_group, x_group)  # shape [sum_all_edges, 3]

        # Free memory
        del alpha_nx12x2, alpha_nx12x2_flat
        torch.cuda.empty_cache()

        # same with beta
        beta_flat = beta.reshape(-1)
        beta_group = torch.gather(beta_flat, dim=0, index=gather_index.long()).reshape(-1, 1)

        # Accumulate into vd
        beta_sum.index_add_(0, edge_group_to_vd, beta_group)
        vd.index_add_(0, edge_group_to_vd, ue_group * beta_group)
        vd /= torch.clamp_min(beta_sum, 1e-12)  # for safety

        # If color
        if vd_color is not None:
            surf_edges_c = torch.index_select(voxelgrid_colors, 0, surf_edges.reshape(-1)).reshape(-1, 2, C)
            c_group = torch.index_select(surf_edges_c, 0, idx_group)
            uc_group = self._linear_interp(s_group * alpha_group, c_group)
            vd_color.index_add_(0, edge_group_to_vd, uc_group * beta_group)
            vd_color /= torch.clamp_min(beta_sum, 1e-12)

        # Next L_dev:
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges.unsqueeze(1))

        # Build final index map => we scatter the dual-vertex index into [surf_cubes,12]
        # v_idx is [0..total_num_vd-1]
        v_idx = torch.arange(total_num_vd, device=device, dtype=torch.int32)
        # we scatter
        vd_idx_map_surf = vd_idx_map_surf.reshape(-1)
        vd_idx_map_surf.scatter_(
            dim=0,
            index=(edge_group_to_cube * 12 + edge_group).long(),
            src=v_idx[edge_group_to_vd]
        )
        vd_idx_map_surf = vd_idx_map_surf.reshape(num_surf_cubes, 12)

        return vd, L_dev, vd_gamma, vd_idx_map_surf, vd_color


    def _triangulate(self, scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map, 
                    vd_idx_map, surf_edges_mask, training, vd_color, chunk_size=100_000):
        """
        Memory-efficient implementation of mesh triangulation.
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into 
        triangles based on the gamma parameter, as described in Section 4.3.
        Args:
            scalar_field: Scalar field values at grid vertices
            surf_edges: Surface-intersecting edges
            vd: Dual vertices
            vd_gamma: Gamma values for dual vertices
            edge_counts: Number of edges per cube
            idx_map: Mapping from cube edges to unique indices
            vd_idx_map: Mapping from cube edges to dual vertex indices
            surf_edges_mask: Boolean mask for surface edges
            training: Whether in training mode
            vd_color: Optional vertex colors
            chunk_size: Size of chunks for memory-efficient processing
        """
        device = self.device

        with torch.no_grad():
            total_size = edge_counts.shape[0]
            group_list = []
            vd_idx_list = []
            
            # Process in chunks, matching the actual data size
            for start in range(0, total_size, chunk_size):
                # Calculate actual available size for this chunk
                available_size = vd_idx_map[start:].shape[0]  # Get remaining data size
                actual_chunk_size = min(chunk_size, available_size)
                end = start + actual_chunk_size
                
                # Create mask of correct size
                chunk_mask = (edge_counts[start:end] == 4) & surf_edges_mask[start:end]
                if not chunk_mask.any():
                    continue
                
                # Use mask indices instead of boolean indexing
                mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]
                
                # Index using the indices
                chunk_group = idx_map[start:end][mask_indices]
                chunk_vd_idx = vd_idx_map[start:end][mask_indices]
                
                group_list.append(chunk_group)
                vd_idx_list.append(chunk_vd_idx)
                
                del chunk_mask, mask_indices, chunk_group, chunk_vd_idx
                torch.cuda.empty_cache()
            
            # Handle empty case
            if not group_list:
                return (vd, 
                        torch.zeros((0, 3), dtype=torch.long, device=device),
                        torch.zeros((0, 2), dtype=scalar_field.dtype, device=device),
                        torch.zeros((0,), dtype=torch.long, device=device),
                        vd_color)
                
            # Concatenate results
            group = torch.cat(group_list, dim=0)
            vd_idx = torch.cat(vd_idx_list, dim=0)
            del group_list, vd_idx_list
            torch.cuda.empty_cache()
            
            # Sort edges and reorder vertices accordingly
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)
            del group, vd_idx, indices
            torch.cuda.empty_cache()

            # Process edge directions in chunks
            s_edges_chunks = []
            for start in range(0, edge_indices.shape[0], chunk_size):
                actual_chunk_size = min(chunk_size, edge_indices.shape[0] - start)
                end = start + actual_chunk_size
                chunk_indices = edge_indices[start:end]
                
                chunk_s_edges = scalar_field[
                    surf_edges[chunk_indices.reshape(-1, 4)[:, 0]].reshape(-1)
                ].reshape(-1, 2)
                s_edges_chunks.append(chunk_s_edges)
                
            s_edges = torch.cat(s_edges_chunks, dim=0)
            del s_edges_chunks
            torch.cuda.empty_cache()

            # Reorder vertices based on scalar field signs
            quad_vd_idx = quad_vd_idx[:len(s_edges)]  # Make sure it matches s_edges length
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((
                quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]
            ))
            
            del flip_mask
            torch.cuda.empty_cache()

        # Process gamma values for vertex placement
        # First, handle the -1 indices
        valid_mask = quad_vd_idx >= 0
        reshaped_idx = quad_vd_idx.reshape(-1)
        valid_idx = reshaped_idx[valid_mask.reshape(-1)]

        # Get gamma values for valid indices
        valid_gamma = torch.index_select(
            input=vd_gamma,
            index=valid_idx,
            dim=0
        )

        # Create output tensor filled with default value (maybe 1.0 or 0.0 depending on your needs)
        quad_gamma = torch.ones_like(reshaped_idx, dtype=vd_gamma.dtype).to(device=vd_gamma.device)
        quad_gamma[valid_mask.reshape(-1)] = valid_gamma
        quad_gamma = quad_gamma.reshape(-1, 4)

        gamma_02 = quad_gamma[:, 0] * quad_gamma[:, 2]
        gamma_13 = quad_gamma[:, 1] * quad_gamma[:, 3]

        if not training:
            # Generate triangles based on gamma comparison
            mask = (gamma_02 > gamma_13)
            faces = torch.zeros(
                (quad_gamma.shape[0], 6), 
                dtype=torch.int32, #int32 to match quad_vd_idx
                device=quad_vd_idx.device
            )
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)
        else:
            # Training mode: compute vertex positions with gamma weighting
            vd_quad = torch.index_select(
                input=vd, 
                index=quad_vd_idx.reshape(-1), 
                dim=0
            ).reshape(-1, 4, 3)
            vd_02 = (vd_quad[:, 0] + vd_quad[:, 2]) / 2
            vd_13 = (vd_quad[:, 1] + vd_quad[:, 3]) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = (
                vd_02 * gamma_02.unsqueeze(-1) + 
                vd_13 * gamma_13.unsqueeze(-1)
            ) / weight_sum.unsqueeze(-1)
            
            # Handle colors if present
            if vd_color is not None:
                color_quad = torch.index_select(
                    input=vd_color, 
                    index=quad_vd_idx.reshape(-1), 
                    dim=0
                ).reshape(-1, 4, vd_color.shape[-1])
                color_02 = (color_quad[:, 0] + color_quad[:, 2]) / 2
                color_13 = (color_quad[:, 1] + color_quad[:, 3]) / 2
                color_center = (
                    color_02 * gamma_02.unsqueeze(-1) + 
                    color_13 * gamma_13.unsqueeze(-1)
                ) / weight_sum.unsqueeze(-1)
                vd_color = torch.cat([vd_color, color_center])
            
            # Generate final vertices and faces
            vd_center_idx = torch.arange(
                vd_center.shape[0], 
                device=device
            ) + vd.shape[0]
            vd = torch.cat([vd, vd_center])
            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)
            faces = torch.cat([
                faces, 
                vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)
            ], -1).reshape(-1, 3)

        return vd, faces, s_edges, edge_indices, vd_color