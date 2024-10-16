import torch
import torch.nn as nn


class KGRModule(nn.Module):
    def __init__(self, module, momentum=None, zscore=0.0, perturbation=0.2,
                 is_module_affine=True, is_input_skewed=True, disabled=False):
        super().__init__()
        self.module = module
        self.momentum = momentum
        self.zscore = zscore
        self.perturbation = perturbation
        self.is_module_affine = is_module_affine
        self.is_input_skewed = is_input_skewed
        self.disabled = disabled

        # Define running buffers
        self.register_buffer("running_center", None)
        self.register_buffer("running_width", None)
        self.register_buffer("batch_count", torch.tensor(0.0))

        # Check if the module has bias
        if hasattr(self.module, "bias") and self.module.bias is not None:
            # Register the bias as a trainable parameter
            self.bias = nn.Parameter(self.module.bias.detach().clone())
            # Remove the bias from the original module
            self.module.bias = None
        else:
            self.bias = None

    def forward(self, x):
        if self.disabled:
            # KGR disabled
            return self.module(x) if self.bias is None else self.module(x) + self.bias

        if self.training:
            if self.is_input_skewed:
                # Use median and MAD if input is skewed, scale MAD to be on the same scale as std
                batch_center = torch.median(x, dim=0).values.detach()
                batch_width = torch.median(torch.abs(x - batch_center), dim=0).values.detach() / 0.6745
            else:
                # Use mean and standard deviation if input is not skewed
                batch_center = torch.mean(x, dim=0).detach()
                batch_width = torch.std(x, dim=0).detach()

            # Update the batch count
            batch_size = x.size(0)  # Number of samples in the current batch
            self.batch_count += batch_size

            # Momentum
            momentum = self.momentum
            if momentum is None:
                # Later batches have smaller and smaller impact
                momentum = batch_size / self.batch_count

            # If this is the first batch, initialize running center and width
            if self.running_center is None or self.running_width is None:
                self.running_center = batch_center
                self.running_width = batch_width
            else:
                self.running_center = self.running_center * (1 - momentum) + batch_center * momentum  # noqa
                self.running_width = self.running_width * (1 - momentum) + batch_width * momentum  # noqa

            # Sampling knots with some noise for training
            knot = torch.normal(self.running_center, self.running_width * self.zscore)
        else:
            # Sampling knots deterministically for inference
            knot = self.running_center

        if not self.is_module_affine:
            # Determine bias based on knot gathering
            bias_kgr = -self.module(knot.unsqueeze(0))

            # Perturbation knot by original bias
            if self.bias is not None:
                bias = bias_kgr * (1 - self.perturbation) + self.bias * self.perturbation
            else:
                bias = bias_kgr

            # Pass the input through the module
            out = self.module(x) + bias
        else:
            # Avoid double forward if module is affine
            if self.bias is not None:
                out = self.module(x - knot * (1 - self.perturbation)) + self.bias * self.perturbation
            else:
                out = self.module(x - knot)
        return out
    