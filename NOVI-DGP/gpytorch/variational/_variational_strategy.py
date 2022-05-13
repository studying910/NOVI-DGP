#!/usr/bin/env python3

from abc import ABC, abstractproperty

import torch
import time

from .. import settings
from ..distributions import Delta, MultivariateNormal
from ..module import Module, _validate_module_outputs
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached, clear_cache_hook
from .stein_generator import SteinGenerator
from .stein_discriminator import SteinDiscriminator
from ..kernels import RBFKernel, ScaleKernel, CosineKernel, MaternKernel, RQKernel


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def rbf_kernel(x1, x2, lengthscale=1):
    # It should return a matrix with size [m1, m2]
    if x1.shape[1] != x2.shape[1]:
        raise ValueError('size of x1 should be [m1, n], size of x2 should be [m2, n]')
    x_list = []
    for i in range(0, x1.shape[0]):
        x1_list = []
        for j in range(0, x2.shape[0]):
            dot = torch.mm((x1[i, :] - x2[j, :]).unsqueeze(0), (x1[i, :] - x2[j, :]).unsqueeze(1))
            x1_list.append(dot)
        dot_tensor = torch.stack(x1_list, dim=1)
        x_list.append(dot_tensor)
    dot_matrix = torch.stack(x_list, dim=0)
    scale_matrix = dot_matrix / (2 * lengthscale ^ 2)
    rbf_matrix = torch.exp(-scale_matrix).to(x1.device)
    return rbf_matrix


class _VariationalStrategy(ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    def __init__(self, model, inducing_points, variational_distribution, stein_type=False, batch_shape=torch.Size([])):
        super(_VariationalStrategy, self).__init__()

        # Model
        object.__setattr__(self, "model", model)

        # TODO: Just use this value to define kernel_method, it will be corrected
        self.batch_shape = batch_shape
        self.inducing_points = inducing_points
        # Inducing points
        # inducing_points = inducing_points.clone()  # identity-mapping, support gradient backward
        # if inducing_points.dim() == 1:
        #     inducing_points = inducing_points.unsqueeze(-1)
        # if learn_inducing_locations:  # learn the locations of z as parameters
        #     torch.nn.Module.register_parameter(self, "inducing_points", torch.nn.Parameter(inducing_points))
        # else:
        #     torch.nn.Module.register_buffer(self, "inducing_points", inducing_points)

        # Variational distribution q(u)
        self.stein = stein_type
        self.print_time = model.print_time
        self.print_param = model.print_param
        self.sample_times = model.sample_times
        if self.stein:
            self.model_stein = model
            self.generator = SteinGenerator(self.model_stein, self.inducing_points)  # generator
            if self.print_param:
                print('Generator:', get_parameter_number(self.generator))
            self.discriminator = SteinDiscriminator(self.model_stein)  # discriminator
            if self.print_param:
                print('Discriminator:', get_parameter_number(self.discriminator))
            # register a kernel method
            self.kernel_method = ScaleKernel(
                RQKernel(batch_shape=torch.Size([]), ard_num_dims=self.inducing_points.shape[-1]),
                batch_shape=torch.Size([]), ard_num_dims=None
            )
            if self.print_param:
                print('Kernel method:', get_parameter_number(self.kernel_method))
        else:
            self._variational_distribution = variational_distribution
            torch.nn.Module.register_buffer(self, "variational_params_initialized", torch.tensor(0))

    def _clear_cache(self):
        clear_cache_hook(self)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        if not self.stein:
            return self._variational_distribution()
        else:
            return None

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param torch.Tensor x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param ~gpytorch.lazy.LazyTensor variational_inducing_covar: If the distribuiton :math:`q(\mathbf u)`
            is Gaussian, then this variable is the covariance matrix of that Gaussian. Otherwise, it will be
            :attr:`None`.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        if self.stein:
            raise NotImplementedError("Can not compute the kl divergence since q(u) is implicit")
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, x, prior=False, train_type=True, disc_type=True, hyp_type=True,
                 trace_layer_list=None, norm_layer_list=None,
                 u_layer_list=None, fiu_layer_list=None, glogpfu_layer_list=None, shared_list=None,
                 transformed_list=None, **kwargs):
        # Set our own default size as follows:
        # size of x: [batch_size, input_dim] -> data or input from previous layer
        # size of z: [inducing_size, input_dim] -> inducing points
        # size of u: [inducing_size * output_dim, 1] -> function value of inducing points
        # size of f: [batch_size, output_dim] -> output value of this layer, obey distribution q(f)
        # size of e: [inducing_size * noise_dim, 1] -> noise sampled from low-dimensional space
        # size of f_u: [inducing_size * output_dim, 1] -> stein function value of u

        # If we're in prior mode, then we're done!
        # if prior:
        #     return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        # if self.training:
        #     self._clear_cache()
        # (Maybe) initialize variational distribution q(U) as p(U)
        # if not self.stein:
        #     if not self.variational_params_initialized.item():
        #         prior_dist = self.prior_distribution
        #         self._variational_distribution.initialize_variational_distribution(prior_dist)
        #         self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size (task size)
        # size of inducing_points: [output_dim, inducing_size, input_dim]
        # TODO: adjust the size of inducing_points to [inducing_size, input_dim] for each layer
        # inducing_points = self.inducing_points  # size of z: [inducing_size, input_dim]
        # if not self.stein:
        #     if inducing_points.shape[:-2] != x.shape[:-2]:
        #         x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get p(u) or q(u)
        variational_dist_u = None
        if not self.stein:
            variational_dist_u = self.variational_distribution

        # Generate u
        # TODO: change here to sample K times to estimate the expectation of q(u) and average loss
        begin = time.time()
        # u = self.inducing_points  # initialize u
        # u_matrix = self.inducing_points
        u_list = []
        u_matrix_list = []
        z_posterior = None
        if self.stein:  # size of u: [inducing_size, output_dim]
            # u and z_posterior are already on cuda
            u_list, z_posterior = self.generator.forward(shared_list,
                                                         transformed_list)  # function value of inducing points
            u_matrix_list = []
            for i in range(0, self.sample_times):
                u_matrix = torch.reshape(u_list[i], (z_posterior.shape[0], -1))  # store the matrix type of u
                u_matrix_list.append(u_matrix)
            if u_layer_list is None:
                u_layer_list = []
                u_layer_list.append(u_list)
            else:
                u_layer_list.append(u_list)  # contain multiple u in each layer instead of a single u
        end = time.time()
        if self.print_time:
            print('Generate u time:', str(end - begin), 's')

        # No matter train or not, we both need to compute the kernel matrix
        kxx = self.kernel_method.forward(x, x)
        kxz = self.kernel_method.forward(x, z_posterior)
        kzx = self.kernel_method.forward(z_posterior, x)
        kzz = self.kernel_method.forward(z_posterior, z_posterior)

        # change here to distinguish train and evaluate
        if train_type:
            # Transform u to f_u and store it in fiu_layer_list
            # f_u = u
            trace_mean = 0.
            f_u_list = []
            if self.stein:
                # f_u, trace are already on cuda
                f_u_list, trace_mean = self.discriminator.forward(
                    u_list)  # stein function value: f_u, trace of loss: trace
                # TODO: examine if the type is list or not (not hurry, finish it after the whole program)
                if fiu_layer_list is None:
                    fiu_layer_list = []
                    fiu_layer_list.append(f_u_list)
                else:
                    fiu_layer_list.append(f_u_list)  # contain multiple f_u in each layer

            # Compute the trace of jacobian matrix using hutchinson estimator, default sample size: 1
            # begin = time.time()
            # trace = 0.
            if self.stein:
                # TODO: add a loop to sample more w to obtain an unbiased estimation
                # w = torch.randn(u_grad.shape[1], 1)
                # jvp = torch.mm(w.t(), u_grad)
                # trace = torch.mm(jvp, w)  # obtain the final estimated trace of this layer
                if trace_layer_list is None:
                    trace_layer_list = []
                    trace_layer_list.append(trace_mean)
                else:
                    trace_layer_list.append(trace_mean)
            # end = time.time()
            # print('Compute trace time:', str(end - begin), 's')

            # Compute the norm of f_u
            begin = time.time()
            norm_u_final = 0.
            norm_u_list = []
            if self.stein:
                for i in range(0, self.sample_times):
                    norm_u = torch.mm(f_u_list[i].unsqueeze(0),
                                      f_u_list[i].unsqueeze(1))  # obtain the dot product of f_u^T and f_u
                    norm_u_list.append(norm_u)
                norm_u_tensor = torch.stack(norm_u_list, dim=0)
                norm_u_final = norm_u_tensor.mean(0)  # average over self.sample
                if norm_layer_list is None:
                    norm_layer_list = []
                    norm_layer_list.append(norm_u_final)
                else:
                    norm_layer_list.append(norm_u_final)
            end = time.time()
            if self.print_time:
                print('Compute norm time:', str(end - begin), 's')

            # Compute gradient of U^T*Kzz^(-1)*U == Kzz^(-1)*U
            begin = time.time()
            glogp_fu = torch.randn(1, requires_grad=True).to(x.device)  # just assign memory, do nothing
            if self.stein:
                # TODO: Integrate this computation using LazyTensor
                # Use representation() to form the original Tensor
                # Or use evaluate() ?
                self.kernel_method = self.kernel_method.to(x.device)
                # kzz = kzz.representation()[0].to(x.device)  # not sure whether kzz is on cuda or not
                # kzz = kzz.evaluate().to(x.device)
                # kzz = rbf_kernel(z_posterior, z_posterior)
                # kzz_detach = kzz.detach()
                kzz_inv = torch.linalg.inv(kzz)
                # u_matrix_detach = u_matrix.detach()
                # u_matrix_detach.requires_grad = True
                # u_matrix is already on cuda
                glogp_fu_list = []
                for i in range(0, self.sample_times):
                    # Compute prior loss directly
                    '''
                    u_matrix_t = torch.transpose(u_matrix_list[i], 0, 1)
                    logpu_jvp = torch.mm(u_matrix_t, kzz_inv)
                    logpu_matrix = torch.mm(logpu_jvp, u_matrix_list[i])
                    logp_u = torch.trace(logpu_matrix)
                    glogpu_matrix = torch.autograd.grad(logp_u, u_matrix_list[i], retain_graph=True, create_graph=True)[
                        0]
                    glogp_u_t = torch.reshape(glogpu_matrix, (1, u_list[i].shape[0]))  # column vector
                    f_u_u = torch.unsqueeze(f_u_list[i], dim=1)
                    glogp_fu = torch.mm(glogp_u_t, f_u_u)  # return the scalar value of dot product
                    glogp_fu_list.append(glogp_fu)
                    '''

                    # Compute prior loss after simplification
                    logpu_matrix = torch.mm(kzz_inv, u_matrix_list[i])
                    glogp_u_t = torch.reshape(logpu_matrix,
                                              (1, u_list[i].shape[0]))  # size of [1, inducing_size * output_dim]
                    f_u_u = torch.unsqueeze(f_u_list[i], dim=1)
                    glogp_fu = torch.mm(glogp_u_t, f_u_u)  # return the scalar value of dot product
                    glogp_fu_list.append(glogp_fu)
                glogp_fu_tensor = torch.stack(glogp_fu_list, dim=0)
                glogp_final_fu = glogp_fu_tensor.mean(0)  # average over self.sample_times
                # contain the dot product between the gradient of logp_u and f_u
                if glogpfu_layer_list is None:
                    glogpfu_layer_list = []
                    glogpfu_layer_list.append(glogp_final_fu)
                else:
                    glogpfu_layer_list.append(glogp_final_fu)
            end = time.time()
            if self.print_time:
                print('Compute prior time:', str(end - begin), 's')

            # update parameter of which network
            if self.stein:
                if disc_type:
                    self.generator.requires_grad_(False)
                    self.generator.inducing_points.requires_grad_(False)
                    self.discriminator.requires_grad_(True)
                    self.kernel_method.requires_grad_(False)
                else:
                    self.generator.requires_grad_(True)
                    self.generator.inducing_points.requires_grad_(False)
                    self.discriminator.requires_grad_(False)
                    self.kernel_method.requires_grad_(False)
        else:
            if hyp_type:
                self.generator.requires_grad_(False)
                self.generator.inducing_points.requires_grad_(True)
                self.discriminator.requires_grad_(False)
                self.kernel_method.requires_grad_(True)
            else:
                self.generator.requires_grad_(False)
                self.generator.inducing_points.requires_grad_(False)
                self.discriminator.requires_grad_(False)
                self.kernel_method.requires_grad_(False)

        # Get q(f)
        if self.stein:  # implement our stein-based method or not
            # Now x, z_posterior and u are already on cuda
            outputs = self.forward_stein(
                x,  # x: input from the previous layer
                z_posterior,  # inducing_points: inducing points sampled in this layer
                inducing_function_values_list=u_list,
                # inducing_function_values: function value of inducing points transformed from sampled noise
                kernel_matrix=(kxx, kxz, kzx, kzz),
                disc_type=disc_type,
                **kwargs,
            )
            # change here to add more returns
            return outputs, trace_layer_list, norm_layer_list, u_layer_list, fiu_layer_list, glogpfu_layer_list
            # return the trace with q(f) and the norm of f_u
        else:
            if isinstance(variational_dist_u, MultivariateNormal):
                return super().__call__(
                    x,
                    self.inducing_points,
                    inducing_values=variational_dist_u.mean,
                    variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                    disc_type=disc_type,
                    **kwargs,
                )
            elif isinstance(variational_dist_u, Delta):
                return super().__call__(
                    x, self.inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None,
                    **kwargs
                )
            else:
                raise RuntimeError(
                    f"Invalid variational distribuition ({type(variational_dist_u)}). "
                    "Expected a multivariate normal or a delta distribution."
                )
