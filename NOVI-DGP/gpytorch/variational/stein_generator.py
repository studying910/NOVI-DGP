#!/usr/bin/env python3

import torch
import torch.nn as nn


class SteinGenerator(nn.Module):
    """"
    When called, first sample from a prior distribution p(e), then transform it to function value of inducing points
    e ~ p(e), u = G(e;\theta)
    """

    def __init__(self, model, inducing_points):
        super(SteinGenerator, self).__init__()
        self.sample_times = model.sample_times  # determine sampling number of u
        self.noise_dim = model.noise_dim  # size of input_dim for noise e
        self.inducing_size = model.inducing_size  # number of inducing points z
        self.vector_type = model.vector_type  # Set True to generate u using vector-based network
        self.noise_share = model.noise_share  # share the noise across layer or not
        # self.share_noise = model.share_noise  # the noise shared by all layers (maybe None to ban it)
        self.multi_head = model.multi_head  # determine to use multi-head or not
        # self.transformed_noise = model.transformed_noise  # noise transformed by back-bone network
        self.input_dim = model.input_dim
        self.output_dim = model.output_dim
        self.concat_type = model.concat_type  # concat e with z or not
        self.noise_add = model.noise_add  # add e or not
        self.device = model.device  # move data to cuda of specified device
        if self.output_dim is None:
            self.output_dim = 1

        self.learn_inducing_locations = model.learn_inducing_locations
        if self.learn_inducing_locations:  # treat location of z as parameter or not
            self.register_parameter(name="inducing_points", param=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer(name="inducing_points", tensor=inducing_points)
        # self.inducing_points = self.inducing_points.to(self.device)  # move z to cuda

        # TODO: We first focus on using vector-based network since it provides a connection between inducing point
        if self.vector_type:
            if self.noise_add:
                if self.concat_type is False:
                    self.transform = nn.Sequential(
                        nn.Linear(in_features=self.noise_dim, out_features=32),
                        # We may add nn.BatchNorm1d to reduce over-fit
                        # We may use Tanh to smooth the process
                        # nn.PReLU(),
                        nn.Sigmoid(),
                        # nn.Linear(in_features=128, out_features=256),
                        # nn.Tanh(),
                        # nn.Linear(in_features=256, out_features=256),
                        # nn.PReLU(),
                        # nn.Linear(in_features=256, out_features=128),
                        # nn.PReLU(),
                        # nn.PReLU(),
                        nn.Linear(in_features=32, out_features=self.inducing_size * self.output_dim),
                        # nn.Sigmoid()
                    )
                elif self.concat_type is True:  # concat e with z to introduce dependency of z for u
                    self.transform = nn.Sequential(
                        nn.Linear(in_features=self.noise_dim + self.input_dim * self.inducing_size, out_features=32),
                        # We may add nn.BatchNorm1d to reduce over-fit
                        # nn.PReLU(),
                        # nn.Tanh(),
                        nn.Sigmoid(),
                        # nn.Linear(in_features=128, out_features=256),
                        # nn.Tanh(),
                        # nn.Linear(in_features=256, out_features=256),
                        # nn.PReLU(),
                        # nn.Linear(in_features=256, out_features=128),
                        # nn.PReLU(),
                        # nn.PReLU(),
                        nn.Linear(in_features=32, out_features=self.inducing_size * self.output_dim),
                        # TODO: Should we add sigmoid function as output of u?
                        # nn.Sigmoid()
                    )
            else:
                self.transform = nn.Sequential(
                    nn.Linear(in_features=self.inducing_size * self.input_dim, out_features=32),
                    # nn.Tanh(),
                    # nn.Linear(in_features=128, out_features=256),
                    # nn.Tanh(),
                    nn.Sigmoid(),
                    # nn.Linear(in_features=256, out_features=256),
                    # nn.PReLU(),
                    # nn.Linear(in_features=256, out_features=128),
                    # nn.PReLU(),
                    nn.Linear(in_features=32, out_features=self.inducing_size * self.output_dim),
                    # nn.Sigmoid()
                )
        else:  # Use matrix-based network to generate u
            if self.noise_add:
                if self.concat_type is False:
                    self.transform = nn.Sequential(
                        nn.Linear(in_features=self.noise_dim, out_features=128),
                        # We may add nn.BatchNorm1d to reduce over-fit
                        # We may use Tanh to smooth the process
                        # nn.PReLU(),
                        nn.Sigmoid(),
                        # nn.Linear(in_features=128, out_features=256),
                        # nn.Tanh(),
                        # nn.Linear(in_features=256, out_features=256),
                        # nn.PReLU(),
                        # nn.Linear(in_features=256, out_features=128),
                        # nn.PReLU(),
                        # nn.PReLU(),
                        nn.Linear(in_features=128, out_features=self.output_dim),
                        # nn.Sigmoid()
                    )
                elif self.concat_type is True:  # concat e with z to introduce dependency of z for u
                    self.transform = nn.Sequential(
                        nn.Linear(in_features=self.noise_dim + self.input_dim, out_features=16),
                        # We may add nn.BatchNorm1d to reduce over-fit
                        # nn.PReLU(),
                        # nn.Tanh(),
                        nn.Sigmoid(),
                        # nn.Linear(in_features=128, out_features=256),
                        # nn.Tanh(),
                        # nn.Linear(in_features=256, out_features=256),
                        # nn.PReLU(),
                        # nn.Linear(in_features=256, out_features=128),
                        # nn.PReLU(),
                        # nn.PReLU(),
                        nn.Linear(in_features=16, out_features=self.output_dim),
                        # TODO: Should we add sigmoid function as output of u?
                        # nn.Sigmoid()
                    )
            else:
                self.transform = nn.Sequential(
                    nn.Linear(in_features=self.input_dim, out_features=16),
                    # nn.Tanh(),
                    # nn.Linear(in_features=128, out_features=256),
                    # nn.Tanh(),
                    nn.Sigmoid(),
                    # nn.Linear(in_features=256, out_features=256),
                    # nn.PReLU(),
                    # nn.Linear(in_features=256, out_features=128),
                    # nn.PReLU(),
                    nn.Linear(in_features=16, out_features=self.output_dim),
                    # nn.Sigmoid()
                )

    def forward(self, shared_list, transformed_list):  # change here to add shared noise and transformed noise
        # TODO: Generate u using different e
        if self.noise_add:
            if not self.noise_share:
                e_list = []
                if self.vector_type:
                    for i in range(0, self.sample_times):
                        e = torch.randn((self.noise_dim,), requires_grad=True).to(self.device)
                        e_list.append(e)
                else:
                    for i in range(0, self.sample_times):
                        e = torch.randn((self.inducing_size, self.noise_dim), requires_grad=True).to(
                            self.device)  # low-dimensional noise e
                        e_list.append(e)
            elif self.multi_head:  # noise transformed through backbone network
                e_list = transformed_list
                if e_list is None:
                    raise ValueError('If multi_head == True, transformed_list must contain element')
            else:  # noise shared across layer
                e_list = shared_list
                if e_list is None:
                    raise ValueError('If noise_share == True, shared_list must contain element')
        else:
            e_list = None

        # transformation
        inducing_points = self.inducing_points.to(self.device)
        inducing_points.requires_grad_(True)
        # e.requires_grad_(True)  # Set True to allow u_list to calculate grad
        if self.vector_type:
            inducing_points = torch.reshape(inducing_points, (self.inducing_size * self.input_dim,))
        else:
            inducing_points = torch.reshape(inducing_points, (self.inducing_size, self.input_dim))
        u_list = []  # a list used to contain u
        for i in range(0, self.sample_times):
            if self.noise_add:
                if self.concat_type is False:
                    u_inter = self.transform(e_list[i])  # function value transformed from noise e
                else:
                    if self.vector_type:
                        e_z = torch.cat((e_list[i], inducing_points), dim=0).to(self.device)
                    else:
                        e_z = torch.cat((e_list[i], inducing_points), dim=1).to(self.device)
                    u_inter = self.transform(e_z)
            else:
                u_inter = self.transform(inducing_points)  # only use z to generate u
            u = torch.reshape(u_inter, (self.inducing_size * self.output_dim,))
            u_list.append(u)
        inducing_points = torch.reshape(inducing_points, (self.inducing_size, self.input_dim))
        return u_list, inducing_points  # return an (inducing_size x output_dim) size vector and posterior z
