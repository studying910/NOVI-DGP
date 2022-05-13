#!/usr/bin/env python3

import warnings

import torch
import time

from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify, \
    lazify
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.cholesky import psd_safe_cholesky
from ..utils.errors import CachingError
from ..utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from ._variational_strategy import _VariationalStrategy
from ..kernels import RBFKernel, ScaleKernel, CosineKernel, MaternKernel, RQKernel


def _ensure_updated_strategy_flag_set(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


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


class VariationalStrategy(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param torch.Tensor inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :type learn_inducing_locations: `bool`, optional

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(self, model, inducing_points, variational_distribution, stein_type=False, batch_shape=torch.Size([])):
        super().__init__(model, inducing_points, variational_distribution, stein_type=stein_type,
                         batch_shape=batch_shape)
        # self.register_buffer("updated_strategy", torch.tensor(True))
        # self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.input_dim = model.input_dim
        self.output_dim = model.output_dim
        if self.output_dim is None:
            self.output_dim = 1
        self.batch_shape = torch.Size([self.output_dim])
        self.print_time = model.print_time
        self.print_param = model.print_param
        self.diagonal_type = model.diagonal_type  # use diagonal matrix of q(f) instead of full kernel matrix
        self.expect_mean = model.expect_mean  # set True to compute mean of q(f) by average over q(u)

        # register a kernel method
        # TODO: Fail: RBFKernel, CosineKernel, MaternKernel
        '''
        self.kernel_method = ScaleKernel(
            RQKernel(batch_shape=torch.Size([]), ard_num_dims=self.input_dim),
            batch_shape=torch.Size([]), ard_num_dims=None
        )
        if self.print_param:
            print('Kernel method for predicting f:', get_parameter_number(self.kernel_method))
        '''

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    # x: the input from the previous layer, inducing_points: Z sampling in this layer
    # inducing_values: mean of the initial variational distribution q(U) in this layer
    # variational_inducing_covar: covariance of the initial variational distribution q(U) in this layer
    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)  # concat Z_{l} with F_{l-1} in num_inducing dim
        full_output = self.model.forward(full_inputs, **kwargs)  # MultivariateNormal distribution variable
        full_covar = full_output.lazy_covariance_matrix  # covariance matrix as a lazytensor

        # Covariance terms
        num_induc = inducing_points.size(-2)  # number of inducing points sampled at layer l
        test_mean = full_output.mean[..., num_induc:]  # mean of x
        induc_induc_covar = full_covar[..., :num_induc,
                            :num_induc].add_jitter()  # upper-left [num_induc,num_induc] matrix
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()  # upper-right matrix
        # evaluate: switch lazytensor to the vanilla tensor
        data_data_covar = full_covar[..., num_induc:, num_induc:]  # lower-right matrix

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)  # cholesky decomposition
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook for this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean
        # @: matrix multiplication, A @ C == A.matmul(C)

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                    data_data_covar.add_jitter(1e-4).evaluate()
                    + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution q(f)
        return MultivariateNormal(predictive_mean, predictive_covar)

    def forward_stein(self, x, inducing_points, inducing_function_values_list, kernel_matrix, disc_type=True, **kwargs):
        # return a MultitaskMultivariateNormal type variable that represents q(f)
        # x: data or input from the previous layer, size: [batch_size, input_dim]
        # inducing_points: inducing points sampled in this layer, size: [inducing_size, input_dim]
        # inducing_function_values: function value of inducing points in this layer, size: [inducing_size, output_dim]

        # Kernel Method used to compute mean and covariance
        # TODO: examine the size of kernel matrix
        begin = time.time()
        # self.kernel_method = self.kernel_method.to(x.device)
        # compute kernel matrix
        # TODO: Integrate with LazyTensor or simply represent it with Tensor
        # Use representation() to form the original Tensor
        # Not sure whether kernel matrix are on cuda or not
        kxx, kxz, kzx, kzz = kernel_matrix
        '''
        kxz = self.kernel_method.forward(x, inducing_points)
        kzz = self.kernel_method.forward(inducing_points, inducing_points)
        kxx = self.kernel_method.forward(x, x)
        kzx = self.kernel_method.forward(inducing_points, x)
        '''
        '''
        # Transform LazyTensor to Tensor
        kxz = kxz.representation()[0].to(x.device)
        kzz = kzz.representation()[0].to(x.device)
        kxx = kxx.representation()[0].to(x.device)
        kzx = kzx.representation()[0].to(x.device)
        '''
        # kxz = kxz.evaluate().to(x.device)
        # kzz = kzz.evaluate().to(x.device)
        # kxx = kxx.evaluate().to(x.device)
        # kzx = kzx.evaluate().to(x.device)

        # kxz = rbf_kernel(x, inducing_points)
        # kzz = rbf_kernel(inducing_points, inducing_points)
        # kxx = rbf_kernel(x, x)
        # kzx = rbf_kernel(inducing_points, x)

        # TODO: Use MatmulLazyTensor
        # compute predictive_mean
        # TODO: inv_matmul can not work with LazyEvaluatedKernelTensor
        inter_kernel = torch.linalg.inv(kzz)
        inter_kernel = torch.mm(kxz, inter_kernel)  # Kxz*Kzz^{-1}
        if not self.expect_mean:
            # only use the first element of inducing_function_values_list
            inducing_function_values_matrix = torch.reshape(inducing_function_values_list[0], (
                inducing_points.shape[0], -1))  # [inducing_size, output_dim]
            predictive_final_mean = torch.mm(inter_kernel,
                                             inducing_function_values_matrix).t()  # [output_dim, batch_size]
        else:
            predictive_mean_list = []
            for i in range(0, self.sample_times):
                inducing_function_values_matrix = torch.reshape(inducing_function_values_list[i],
                                                                (inducing_points.shape[0], -1))
                predictive_mean = torch.mm(inter_kernel,
                                           inducing_function_values_matrix).t()  # [output_dim, batch_size]
                predictive_mean_list.append(predictive_mean)
            predictive_mean_tensor = torch.stack(predictive_mean_list, dim=0)
            predictive_final_mean = predictive_mean_tensor.mean(0)  # average over self.sample_times

        # compute predictive_covar
        predictive_covar_full = kxx - torch.mm(inter_kernel, kzx)
        noise_covar = torch.ones_like(predictive_covar_full).mul(1e-4)  # add a noise to satisfy positive-definite
        if self.diagonal_type:
            predictive_covar_diag = torch.diag(predictive_covar_full)
            predictive_covar = torch.diag_embed(predictive_covar_diag)
            predictive_covar = torch.abs(predictive_covar).add(noise_covar).unsqueeze(0)
            #  inducing_function_values.shape[0] / inducing_points.shape[0] == output_dim
            predictive_covar = predictive_covar.repeat(self.output_dim, 1, 1)
        else:
            predictive_covar_full = torch.abs(predictive_covar_full).add(noise_covar).unsqueeze(0)
            predictive_covar = predictive_covar_full.repeat(self.output_dim, 1, 1)
        # [output_dim, batch_size, batch_size]

        # return the distribution q(f)
        # TODO: integrate with MultitaskMultivariateNormal
        # treat task of q(f) as batch, so initialize as MultivariateNormal
        end = time.time()
        if self.print_time:
            print('Predict f time:', str(end - begin), 's')

        # Froze the kernel method when updating discriminator D
        '''
        if disc_type:
            self.kernel_method.requires_grad_(False)
        else:
            self.kernel_method.requires_grad_(True)
        '''

        # TODO: just return tuple: (predictive_final_mean, predictive_covar)
        return predictive_final_mean, predictive_covar
        # return MultivariateNormal(predictive_final_mean, predictive_covar)

    def __call__(self, x, prior=False, train_type=True, disc_type=True, hyp_type=True,
                 trace_layer_list=None, norm_layer_list=None,
                 u_layer_list=None, fiu_layer_list=None, glogpfu_layer_list=None,
                 shared_list=None, transformed_list=None, **kwargs):
        '''
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                whitened_mean = L.inv_matmul(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()
                covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                whitened_covar = RootLazyTensor(L.inv_matmul(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)
        '''

        return super().__call__(x, prior=prior, train_type=train_type, disc_type=disc_type,
                                hyp_type=hyp_type, trace_layer_list=trace_layer_list,
                                norm_layer_list=norm_layer_list, u_layer_list=u_layer_list,
                                fiu_layer_list=fiu_layer_list, glogpfu_layer_list=glogpfu_layer_list,
                                shared_list=shared_list, transformed_list=transformed_list, **kwargs)
