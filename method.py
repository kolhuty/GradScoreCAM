import torch
import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM

class GradScoreCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None, topk_ratio=0.05):
        super(GradScoreCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform=reshape_transform,
                                       uses_gradients = True)
        self.topk_ratio = topk_ratio
        self.sel_activations = torch.Tensor().to(self.device)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations_full,
                        grads):

        grads = torch.from_numpy(grads).to(self.device)
        with torch.no_grad():

            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:]
            )

            activation_tensor_full = torch.from_numpy(activations_full).to(self.device) #[B, C, H, W]

            # importance scores for each batch
            hires_scores = torch.sum(activation_tensor_full * grads, dim=(2, 3)) #[B, C]
            B, C = hires_scores.shape

            # select k the most important activation maps
            k = max(1, int(C * self.topk_ratio))
            topk_indices = torch.topk(hires_scores, k=k, dim=1).indices  # [B, k]

            # collect maps for each batch
            selected_activations_list = []
            for b in range(B):
                selected = activation_tensor_full[b, topk_indices[b]]  # [k, H, W]
                selected_activations_list.append(selected)
            self.sel_activations = torch.stack(selected_activations_list)  # [B, k, H, W]

            # upsample
            upsampled = upsample(self.sel_activations)
            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(self.sel_activations.shape[0], self.sel_activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: list,
        activations_full: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations_full, grads)
        activations_for_weighted_sum = self.sel_activations.cpu().detach().numpy()

        # 2D conv
        if len(activations_for_weighted_sum.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations_for_weighted_sum
        # 3D conv
        elif len(activations_for_weighted_sum.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations_for_weighted_sum
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations_for_weighted_sum.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

