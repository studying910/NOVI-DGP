import torch
import torch.nn as nn
import tqdm
import time
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal, Delta
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP

import urllib.request
import os
from scipy.io import loadmat
from math import floor

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

if not smoke_test and not os.path.isfile('data/elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
#    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(1000, 3), torch.randn(1000)
else:
    data = torch.Tensor(loadmat('data/elevators.mat')['data'])
    X = data[:, :-1]  # the last column is label
    X = X - X.min(0)[0]  # X.min(0)[0]: min value of every feature
    X = 2 * (X / X.max(0)[0]) - 1  # pre-preocess to [-1,1]
    y = data[:, -1]
    y -= y.mean()
    y /= y.std()  # pre-process to N(0,1)

train_n = int(floor(0.8 * len(X)))  # split train set, ratio: 0.8
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.to(device), train_y.to(device), test_x.to(device), test_y.to(device)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


from torch.utils.data import TensorDataset, DataLoader

loader_batch_size = 1024
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)

print_time = False  # set True to print module time
print_norm = False  # set True to print norm of f_u (the perfect value of norm after training: 0)
print_metric = True  # set True to print metric per epoch
print_param = True  # set True to print parameter size of each network


class ToyDeepGPHiddenLayer(DeepGPLayer):  # input_dims: size of feature dim
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', stein=False,
                 noise_share=False, noise_dim=1, share_noise=None, multi_head=False, transformed_noise=None):
        # TODO: adjust the size of inducing_points to [inducing_size, input_dim] for each layer
        if stein is False:
            if output_dims is None:
                inducing_points = torch.randn((num_inducing, input_dims),
                                              requires_grad=True)  # sample from gaussian dist
                batch_shape = torch.Size([])
            else:  # inducing_points-dim correspond to: output_dims, num_inducing and input_dims
                inducing_points = torch.randn((output_dims, num_inducing, input_dims), requires_grad=True)
                batch_shape = torch.Size([output_dims])
        else:
            if output_dims is None:
                inducing_points = torch.randn((num_inducing, input_dims),
                                              requires_grad=True)  # sample from gaussian dist
                batch_shape = torch.Size([])
            else:  # inducing_points-dim correspond to: [num_inducing, input_dims]
                inducing_points = torch.randn((num_inducing, input_dims), requires_grad=True)
                batch_shape = torch.Size([output_dims])

        self.input_dim = input_dims
        self.output_dim = output_dims
        self.inducing_size = num_inducing
        '''
        if noise_share is False:
            if output_dims is None:
                self.noise_dim = 1
            else:
                self.noise_dim = round(output_dims/2) # dim of noise e that sampled to generate u
        else:
            self.noise_dim = 5 # fine-tune this term or directly learn as a prior
        '''
        self.noise_dim = noise_dim
        self.concat_type = True  # set to True to concat e with z in generator G
        self.noise_add = True  # Set False to input z only to generate u
        self.noise_share = noise_share
        '''
        if noise_share:  # only set to True when num_inducing and self.noise_dim is the same for all layers
            self.share_noise = torch.randn((num_inducing, self.noise_dim), requires_grad=True)
        else:
            self.share_noise = None
        '''
        self.share_noise = share_noise  # noise shared by all layers
        self.multi_head = multi_head
        self.transformed_noise = transformed_noise

        self.hutch_times = 1  # number for hutchison estimation of trace
        self.bottleneck_trick = True  # Set True to introduce decomposition of jacobian to reduce variance
        self.rademacher_type = False  # Set True to sample from Radamecher dist to compute trace

        self.print_time = print_time
        self.print_param = print_param

        # if stein=True, there is no need to generate variational_distribution
        if stein is False:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=num_inducing,
                batch_shape=batch_shape
            )
        else:
            variational_distribution = None

        variational_strategy = VariationalStrategy(  # transform q(U) to q(F)
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            stein_type=stein,
            batch_shape=batch_shape
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )  # ard_num_dims: seperate lengthscale for each feature
        # batch_shape: seperate lengthscale for each batch of input

    def forward(self, x):
        mean_x = self.mean_module(x)  # mean_x: [batch_size,1]
        covar_x = self.covar_module(x)  # covar_x: [batch_size,batch_size]
        return MultivariateNormal(mean_x, covar_x)

    # will be called when run the forward function of class DeepGP
    def __call__(self, x, train_type=True, disc_type=True, trace_layer_list=None, norm_layer_list=None,
                 u_layer_list=None,
                 fiu_layer_list=None, glogpfu_layer_list=None, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)), train_type=train_type, disc_type=disc_type,
                                trace_layer_list=trace_layer_list, norm_layer_list=norm_layer_list,
                                u_layer_list=u_layer_list, fiu_layer_list=fiu_layer_list,
                                glogpfu_layer_list=glogpfu_layer_list)


num_output_dims = 2 if smoke_test else 10  # number of output_dim in hidden layer
num_layer_inducing = 128  # we can fine-tune it to obtain a better result


class DeepGP(DeepGP):  # define the noise outside the layer
    def __init__(self, train_x_shape, stein=False, noise_share=False, multi_head=False):  # L=2
        if noise_share is False and multi_head:
            raise ValueError('Only can use Multi-head Mechanism when noise is shared across layer')
        super().__init__()
        self.noise_share = noise_share  # share the noise across layer or not
        if self.noise_share is False:
            self.noise_dim = 1  # no meaning
        else:
            self.noise_dim = 5  # fine-tune this term or directly learn as a prior
        if self.noise_share:  # only set to True when num_inducing and self.noise_dim is the same for all layers
            self.share_noise = torch.randn((num_layer_inducing, self.noise_dim))
        else:
            self.share_noise = None
        self.multi_head = multi_head
        if self.multi_head:
            self.back_bone = nn.Sequential(
                nn.Linear(in_features=self.noise_dim, out_features=32),
                nn.Tanh(),
                nn.Linear(in_features=32, out_features=64),
                nn.PReLU(),
                nn.Linear(in_features=64, out_features=128),
                nn.PReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.Tanh(),
                nn.Linear(in_features=64, out_features=32),
                nn.PReLU(),
                nn.Linear(in_features=32, out_features=self.noise_dim)
            )
            if print_param:
                print('Generator backbone:', get_parameter_number(self.back_bone))
        else:
            self.back_bone = None
        if self.multi_head:
            self.transformed_noise = self.back_bone(self.share_noise)
        else:
            self.transformed_noise = None

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            num_inducing=num_layer_inducing,
            mean_type='linear',
            stein=stein,
            noise_share=self.noise_share,
            noise_dim=self.noise_dim,
            share_noise=self.share_noise,
            multi_head=self.multi_head,
            transformed_noise=self.transformed_noise,
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dim,
            output_dims=None,
            num_inducing=num_layer_inducing,
            mean_type='constant',
            stein=stein,
            noise_share=self.noise_share,
            noise_dim=self.noise_dim,
            share_noise=self.share_noise,
            multi_head=self.multi_head,
            transformed_noise=self.transformed_noise,
        )

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()  # likelihood of p(Y|F)

    def forward(self, inputs):
        # set train_type=False when evaluate, do not compute the loss to faster the process
        inputs, train_type, disc_type, trace_layer_list, norm_layer_list, u_layer_list, fiu_layer_list, glogpfu_layer_list = inputs
        hidden_rep1, trace_layer_list, norm_layer_list, u_layer_list, fiu_layer_list, glogpfu_layer_list = self.hidden_layer(
            inputs, train_type=train_type, disc_type=disc_type,
            trace_layer_list=trace_layer_list,
            norm_layer_list=norm_layer_list,
            u_layer_list=u_layer_list,
            fiu_layer_list=fiu_layer_list,
            glogpfu_layer_list=glogpfu_layer_list)
        # get a prob distribution, not a deterministic vector value
        output = self.last_layer(hidden_rep1, train_type=train_type, disc_type=disc_type,
                                 trace_layer_list=trace_layer_list,
                                 norm_layer_list=norm_layer_list, u_layer_list=u_layer_list,
                                 fiu_layer_list=fiu_layer_list, glogpfu_layer_list=glogpfu_layer_list)
        return output

    def predict(self, test_loader):
        #        with torch.no_grad():
        mus = []
        variances = []
        lls = []
        for x_batch, y_batch in test_loader:
            x_batch_f, _, _, _, _, _ = self.forward((x_batch, False, True, [], [], [], [], []))
            preds = self.likelihood(x_batch_f)  # self(x_batch): f(X), preds: Y=f(X)+e
            mus.append(preds.mean)
            variances.append(preds.variance)
            lls.append(model.likelihood.log_marginal(y_batch, x_batch_f))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


# NOTE: When strange error occur (especially wrong error line), restart the kernel
# If multi_head=True, Transform shared noise by a network and concat with layer-specified inducing points
model = DeepGP(train_x.shape, stein=True, noise_share=True, multi_head=False)
print('Test whether you have cuda to run the process or not:', torch.cuda.is_available())
if torch.cuda.is_available():
    model = model.to(device)

from gpytorch.settings import num_likelihood_samples

# this is for running the notebook in our testing framework
num_epochs = 1 if smoke_test else 1
num_nc = 1 if smoke_test else 1  # use to train discriminator per epoch
num_samples = 3 if smoke_test else 10
lamda = 0.1 if smoke_test else 0.1  # hyper-parameter for entire loss
num_likelihood_fsamples = 1 if smoke_test else 10  # number of samples S drawn from q(f)

print(get_parameter_number(model))

# optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.001)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
# mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2])) # marginal log-likelihood

torch.autograd.set_detect_anomaly(True)

# training stage
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:  # change here to add adversarial training
    print('New epoch!!!')
    mus_disc = []  # contain all tensors in disc loop
    variance_disc = []
    lls_disc = []
    # fix hyperparameter v and generator G, update discriminator D num_nc times
    disc_iter = tqdm.tqdm(range(num_nc), desc='Train Discriminator', leave=False)
    for j in disc_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter_disc = tqdm.tqdm(train_loader, desc="Minibatch-Inside", leave=False)
        mus_disc_j = []
        variance_disc_j = []
        lls_disc_j = []
        for x_batch_disc, y_batch_disc in minibatch_iter_disc:
            with num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                trace_layer_list_disc = []
                norm_layer_list_disc = []
                u_layer_list_disc = []
                fiu_layer_list_disc = []
                glogpfu_layer_list_disc = []
                # trace_layer_list: contain the final trace in each layer
                # norm_layer_list: contain the norm in each layer
                # u_layer_list: contain u in each layer
                # fiu_layer_list: contain f_u in each layer
                # glogpu_layer_list: contain dot product of gradient of logp_u with f_u in each layer
                # disc_type: determine update discriminator or generator
                output_disc, trace_layer_list_disc, norm_layer_list_disc, u_layer_list_disc, fiu_layer_list_disc, glogpfu_layer_list_disc = model.forward(
                    (x_batch_disc, True, True, trace_layer_list_disc, norm_layer_list_disc, u_layer_list_disc,
                     fiu_layer_list_disc, glogpfu_layer_list_disc))
                # the second loss term
                trace_layer_tensor_disc = torch.stack(trace_layer_list_disc, dim=0)
                trace_loss_disc = trace_layer_tensor_disc.sum(0)
                # the third loss term
                norm_layer_tensor_disc = torch.stack(norm_layer_list_disc, dim=0)
                norm_loss_disc = norm_layer_tensor_disc.mean(0)
                if print_norm:
                    print('DISC Norm of f_u:', norm_loss_disc.item())
                norm_loss_disc = lamda * norm_loss_disc  # need to negativate this loss
                # the first term in the first loss term
                glogpfu_tensor_disc = torch.stack(glogpfu_layer_list_disc, dim=0)
                glogpu_loss_disc = glogpfu_tensor_disc.sum(0)
                glogpu_loss_disc = -0.5 * glogpu_loss_disc
                # the second term in the first loss term
                # f_mean = output.mean
                # f_covar = output.covar.representation()[0]  # tensor type of covariance matrix
                # call the API instead of setting our own function
                logpy_f_disc = model.likelihood.log_marginal(y_batch_disc, output_disc).mean(
                    dim=1)  # may need to divide loader_batch_size
                glogpyf_layer_list_disc = []
                # glogpyf_layer_list: gradient of logpy_f in each layer
                begin = time.time()
                for l in range(0, len(u_layer_list_disc)):
                    glogpyf_l_disc = \
                        torch.autograd.grad(logpy_f_disc, u_layer_list_disc[l], retain_graph=True, create_graph=True)[0]
                    glogpyf_layer_list_disc.append(glogpyf_l_disc)
                glogpyffu_layer_list_disc = []
                # glogpyffu_layer_list: the second term in the first loss term in each layer
                for l in range(0, len(u_layer_list_disc)):
                    glogpyffu_l_disc = torch.mm(glogpyf_layer_list_disc[l].unsqueeze(0),
                                                fiu_layer_list_disc[l].unsqueeze(1))
                    glogpyffu_layer_list_disc.append(glogpyffu_l_disc)
                end = time.time()
                if print_time:
                    print('Compute likelihood time:', str(end - begin), 's')
                glogpyffu_tensor_disc = torch.stack(glogpyffu_layer_list_disc, dim=0)
                glogpyffu_loss_disc = glogpyffu_tensor_disc.sum(0)
                # the first loss term
                score_loss_disc = glogpu_loss_disc + glogpyffu_loss_disc
                # change here to form our own loss
                # loss = -mll(output, y_batch)
                loss_disc = score_loss_disc + trace_loss_disc - norm_loss_disc
                torch.autograd.backward(loss_disc)
                optimizer.step()
                minibatch_iter_disc.set_postfix(loss=loss_disc.item())

                preds_disc = model.likelihood(output_disc)
                mus_disc_j.append(preds_disc.mean)
                variance_disc_j.append(preds_disc.variance)
                lls_disc_j.append(model.likelihood.log_marginal(y_batch_disc, output_disc))

        means_disc_j = torch.cat(mus_disc_j, dim=-1)
        variances_disc_j = torch.cat(variance_disc_j, dim=-1)
        test_lls_disc_j = torch.cat(lls_disc_j, dim=-1).mean()

        mus_disc.append(means_disc_j)
        variance_disc.append(variances_disc_j)
        lls_disc.append(test_lls_disc_j)

    rmse_disc_list = []
    for j in range(0, num_nc):
        rmse_disc_j = torch.mean(torch.pow(mus_disc[j].mean(0) - train_y, 2)).sqrt()
        rmse_disc_list.append(rmse_disc_j)
    rmse_disc = torch.stack(rmse_disc_list, dim=0).mean(0)
    test_lls_disc = torch.stack(lls_disc, dim=0)
    if print_metric:
        print(f"DISC_RMSE: {rmse_disc.item()}, DISC_NLL: {-test_lls_disc.mean().item()}")

    # Within each iteration, we will go over each minibatch of data
    minibatch_iter_gen = tqdm.tqdm(train_loader, desc="Minibatch-Outside", leave=False)
    mus_gen = []
    variance_gen = []
    lls_gen = []
    # fix discriminator D and train hyperparameter v with generator G
    for x_batch_gen, y_batch_gen in minibatch_iter_gen:
        with num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            trace_layer_list_gen = []
            norm_layer_list_gen = []
            u_layer_list_gen = []
            fiu_layer_list_gen = []
            glogpfu_layer_list_gen = []
            output_gen, trace_layer_list_gen, norm_layer_list_gen, u_layer_list_gen, fiu_layer_list_gen, glogpfu_layer_list_gen = model.forward(
                (x_batch_gen, True, True, trace_layer_list_gen,
                 norm_layer_list_gen, u_layer_list_gen, fiu_layer_list_gen,
                 glogpfu_layer_list_gen))
            trace_layer_tensor_gen = torch.stack(trace_layer_list_gen, dim=0)
            trace_loss_gen = trace_layer_tensor_gen.sum(0)

            norm_layer_tensor_gen = torch.stack(norm_layer_list_gen, dim=0)
            norm_loss_gen = norm_layer_tensor_gen.sum(0)
            norm_loss_gen = lamda * norm_loss_gen  # need to negativate this loss

            glogpfu_tensor_gen = torch.stack(glogpfu_layer_list_gen, dim=0)
            glogpu_loss_gen = glogpfu_tensor_gen.sum(0)
            glogpu_loss_gen = -0.5 * glogpu_loss_gen  # occur wrong when update generator G

            # the second term in the first loss term
            # f_mean = output.mean
            # f_covar = output.covar.representation()[0]  # tensor type of covariance matrix
            # call the API instead of setting our own function
            logpy_f_gen = model.likelihood.log_marginal(y_batch_gen, output_gen).mean(
                dim=1)  # may need to divide loader_batch_size
            glogpyf_layer_list_gen = []
            # glogpyf_layer_list: gradient of logpy_f in each layer
            for l in range(0, len(u_layer_list_gen)):
                glogpyf_l_gen = \
                    torch.autograd.grad(logpy_f_gen, u_layer_list_gen[l], retain_graph=True, create_graph=True)[0]
                glogpyf_layer_list_gen.append(glogpyf_l_gen)
            glogpyffu_layer_list_gen = []
            # glogpyffu_layer_list: the second term in the first loss term in each layer
            for l in range(0, len(u_layer_list_gen)):
                glogpyffu_l_gen = torch.mm(glogpyf_layer_list_gen[l].unsqueeze(0), fiu_layer_list_gen[l].unsqueeze(1))
                glogpyffu_layer_list_gen.append(glogpyffu_l_gen)
            glogpyffu_tensor_gen = torch.stack(glogpyffu_layer_list_gen, dim=0)
            glogpyffu_loss_gen = glogpyffu_tensor_gen.sum(0)
            # the first loss term
            score_loss_gen = glogpu_loss_gen + glogpyffu_loss_gen  # glogpu_loss +
            # change here to form our own loss
            # loss = -mll(output, y_batch)
            loss_gen = score_loss_gen + trace_loss_gen - norm_loss_gen
            torch.autograd.backward(loss_gen)
            optimizer.step()
            minibatch_iter_gen.set_postfix(loss=loss_gen.item())

            preds_gen = model.likelihood(output_gen)
            mus_gen.append(preds_gen.mean)
            variance_gen.append(preds_gen.variance)
            lls_gen.append(model.likelihood.log_marginal(y_batch_gen, output_gen))

    means_gen = torch.cat(mus_gen, dim=-1)
    variances_gen = torch.cat(variance_gen, dim=-1)
    test_lls_gen = torch.cat(lls_gen, dim=-1)

    rmse_gen = torch.mean(torch.pow(means_gen.mean(0) - train_y, 2)).sqrt()
    if print_metric:
        print(f"GEN_RMSE: {rmse_gen.item()}, GEN_NLL: {-test_lls_gen.mean().item()}")

test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=loader_batch_size)

# model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
