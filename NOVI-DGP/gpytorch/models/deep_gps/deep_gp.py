import warnings

import torch

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor
from gpytorch.likelihoods import Likelihood

from ..approximate_gp import ApproximateGP
from ..gp import GP


class _DeepGPVariationalStrategy(object):
    def __init__(self, model):
        self.model = model

    @property
    def sub_variational_strategies(self):
        if not hasattr(self, "_sub_variational_strategies_memo"):
            self._sub_variational_strategies_memo = [
                module.variational_strategy for module in self.model.modules() if isinstance(module, ApproximateGP)
            ]
        return self._sub_variational_strategies_memo

    def kl_divergence(self):
        return sum(strategy.kl_divergence().sum() for strategy in self.sub_variational_strategies)


class DeepGPLayer(ApproximateGP):
    """
    Represents a layer in a deep GP where inference is performed via the doubly stochastic method of
    Salimbeni et al., 2017. Upon calling, instead of returning a variational distribution q(f), returns samples
    from the variational distribution.

    See the documentation for __call__ below for more details below. Note that the behavior of __call__
    will change to be much more elegant with multiple batch dimensions; however, the interface doesn't really
    change.

    :param ~gpytorch.variational.VariationalStrategy variational_strategy: Strategy for
        changing q(u) -> q(f) (see other VI docs)
    :param int input_dims`: Dimensionality of input data expected by each GP
    :param int output_dims: (default None) Number of GPs in this layer, equivalent to
        output dimensionality. If set to `None`, then the output dimension will be squashed.

    Forward data through this hidden GP layer. The output is a MultitaskMultivariateNormal distribution
    (or MultivariateNormal distribution is output_dims=None).

    If the input is >=2 dimensional Tensor (e.g. `n x d`), we pass the input through each hidden GP,
    resulting in a `n x h` multitask Gaussian distribution (where all of the `h` tasks represent an
    output dimension and are independent from one another).  We then draw `s` samples from these Gaussians,
    resulting in a `s x n x h` MultitaskMultivariateNormal distribution.

    If the input is a >=3 dimensional Tensor, and the `are_samples=True` kwarg is set, then we assume that
    the outermost batch dimension is a samples dimension. The output will have the same number of samples.
    For example, a `s x b x n x d` input will result in a `s x b x n x h` MultitaskMultivariateNormal distribution.

    The goal of these last two points is that if you have a tensor `x` that is `n x d`, then

        >>> hidden_gp2(hidden_gp(x))

    will just work, and return a tensor of size `s x n x h2`, where `h2` is the output dimensionality of
    hidden_gp2. In this way, hidden GP layers are easily composable.
    """

    def __init__(self, variational_strategy, input_dims, output_dims):
        super(DeepGPLayer, self).__init__(variational_strategy)
        self.input_dims = input_dims
        self.output_dims = output_dims

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, inputs, are_samples=False, train_type=True, disc_type=True, hyp_type=True,
                 trace_layer_list=None, norm_layer_list=None,
                 u_layer_list=None, fiu_layer_list=None, glogpfu_layer_list=None,
                 shared_list=None, transformed_list=None, **kwargs):
        # for now, size of x: [batch_size, input_dim], if not dist of q(f), no action need to be taken
        deterministic_inputs = not are_samples  # default: True
        # transform distribution q(f) to samples f from q(f) ~ N(mu, Kff)
        # TODO: change here to make size of f samples: [input_dim, batch_size] and transpose it
        # if isinstance(inputs, MultivariateNormal):  # for all layers except the first layer
        if type(inputs) == tuple:
            # input_covar = inputs.lazy_covariance_matrix.representation()[0]  # [input_dim, batch_size, batch_size]
            means, covariances = inputs
            input_list = []
            # for i in range(0, input_covar.shape[0]):
            for i in range(0, covariances.shape[0]):
                # inputs.mean: [input_dim, batch_size]
                # input_i = torch.distributions.MultivariateNormal(loc=inputs.mean[i, :],
                #                                                  covariance_matrix=input_covar[i, :, :]).rsample(
                #     torch.Size([]))
                input_i = torch.distributions.MultivariateNormal(loc=means[i, :],
                                                                 covariance_matrix=covariances[i, :, :]).rsample(
                    torch.Size([]))
                input_list.append(input_i)
            inputs = torch.stack(input_list, dim=1)  # [batch_size, input_dim]
            # input_diag = torch.diag(input_covar)
            # input_noise = torch.randn_like(inputs.mean)
            # inputs = torch.mul(input_noise, input_diag) + inputs.mean
            # inputs = torch.distributions.MultivariateNormal(loc=inputs.mean,
            #                                                 covariance_matrix=input_covar.sqrt()).rsample()
            deterministic_inputs = False

        if settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateNormal or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.shape[-1] != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        # Repeat the input for all possible outputs
        # TODO: change it to make the size of x: [batch_size, input_dim] and remove the output_dim
        # if self.output_dims is not None:
        #    inputs = inputs.unsqueeze(-3)
        #    inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        # Now run samples through the GP
        # Now size of inputs x: [batch_size, input_dim]
        output, trace_layer_list, norm_layer_list, u_layer_list, fiu_layer_list, glogpfu_layer_list = \
            ApproximateGP.__call__(
                self,
                inputs,
                train_type=train_type,
                disc_type=disc_type,
                hyp_type=hyp_type,
                trace_layer_list=trace_layer_list,
                norm_layer_list=norm_layer_list,
                u_layer_list=u_layer_list,
                fiu_layer_list=fiu_layer_list,
                glogpfu_layer_list=glogpfu_layer_list,
                shared_list=shared_list,
                transformed_list=transformed_list)  # core code
        # if self.output_dims is not None:
        #    mean = output.loc.transpose(-1, -2)
        #    covar = BlockDiagLazyTensor(output.lazy_covariance_matrix, block_dim=-3)
        #    output = MultitaskMultivariateNormal(mean, covar, interleaved=False)

        # Maybe expand inputs?
        # if deterministic_inputs:
        #    output = output.expand(torch.Size([settings.num_likelihood_samples.value()]) + output.batch_shape)

        return output, trace_layer_list, norm_layer_list, u_layer_list, fiu_layer_list, glogpfu_layer_list


class DeepGP(GP):
    """
    A container module to build a DeepGP.
    This module should contain :obj:`~gpytorch.models.deep.DeepGPLayer`
    modules, and can also contain other modules as well.
    """

    def __init__(self):
        super().__init__()
        self.variational_strategy = _DeepGPVariationalStrategy(self)

    def forward(self, x):
        raise NotImplementedError


class DeepLikelihood(Likelihood):
    """
    A wrapper to make a GPyTorch likelihood compatible with Deep GPs

    Example:
        >>> deep_gaussian_likelihood = gpytorch.likelihoods.DeepLikelihood(gpytorch.likelihood.GaussianLikelihood)
    """

    def __init__(self, base_likelihood):
        super().__init__()
        warnings.warn(
            "DeepLikelihood is now deprecated. Use a standard likelihood in conjunction with a "
            "gpytorch.mlls.DeepApproximateMLL. See the DeepGP example in our documentation.",
            DeprecationWarning,
        )
        self.base_likelihood = base_likelihood

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        return self.base_likelihood.expected_log_prob(observations, function_dist, *params, **kwargs).mean(dim=0)

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        return self.base_likelihood.log_marginal(observations, function_dist, *params, **kwargs).mean(dim=0)

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.base_likelihood.__call__(*args, **kwargs)
