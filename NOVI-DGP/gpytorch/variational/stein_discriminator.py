#!/usr/bin/env python3

import torch
import torch.nn as nn
import time


class SteinDiscriminator(nn.Module):
    """
    When called, transform function value of inducing points u to stein function value f_u
    """

    def __init__(self, model):
        super(SteinDiscriminator, self).__init__()
        self.output_dim = model.output_dim
        self.inducing_size = model.inducing_size
        if self.output_dim is None:
            self.output_dim = 1
        self.hutch_times = model.hutch_times  # number of sample w to estimate trace
        self.bottleneck_trick = model.bottleneck_trick  # bottleneck trick to reduce variance
        self.rademacher_type = model.rademacher_type  # dist of noise to compute trace
        self.print_time = model.print_time
        self.device = model.device
        if not self.bottleneck_trick:
            self.transform = nn.Sequential(
                nn.Linear(in_features=self.output_dim * self.inducing_size,
                          out_features=16),
                # nn.PReLU(),  # need to add activate function to obtain solid results
                # We consider to use Tanh function to smooth the process
                nn.Tanh(),
                # nn.Linear(in_features=128, out_features=128),
                # nn.PReLU(),
                # nn.BatchNorm1d() may not be used if torch.autograd.functional.jacobian does not support it
                nn.Linear(in_features=16, out_features=self.output_dim * self.inducing_size),
            )
        else:
            self.transform_h = nn.Sequential(
                nn.Linear(in_features=self.output_dim * self.inducing_size,
                          out_features=16),
                nn.Tanh(),
            )
            self.transform_g = nn.Sequential(
                nn.Linear(in_features=16, out_features=self.output_dim * self.inducing_size),
            )

    # dimension of u: [inducing_size * output_dim, 1]
    def forward(self, u_list):
        # u_matrix = u.reshape(self.inducing_size, -1)
        # f_u_matrix = self.transform(u_matrix)
        # f_u = f_u_matrix.reshape(self.inducing_size * self.output_dim)

        begin = time.time()
        h_u_list = []
        f_u_list = []
        for j in range(0, len(u_list)):
            if not self.bottleneck_trick:
                h_u = u_list[j]
                f_u = self.transform(u_list[j])
            else:
                h_u = self.transform_h(u_list[j])
                f_u = self.transform_g(h_u)
            h_u_list.append(h_u)
            f_u_list.append(f_u)
        end = time.time()
        if self.print_time:
            print('Generate f_u time:', str(end - begin), 's')
        # carefully set vectorize=True to faster training, may result in failure
        trace__final_mean = 0.
        begin = time.time()
        # TODO: We need to average over self.sample_times to obtain the final trace_mean
        trace_mean_list = []  # use to contain trace to average
        for j in range(0, len(u_list)):
            if not self.bottleneck_trick:
                trace_list = []
                for i in range(0, self.hutch_times):
                    if not self.rademacher_type:
                        w_t = torch.randn(u_list[j].shape[0]).to(self.device)
                    else:
                        w_t = torch.randint(low=0, high=2, size=(u_list[j].shape[0],)).to(u_list[j].dtype) * 2 - 1
                        w_t = w_t.to(self.device)
                    u_grad = \
                        torch.autograd.grad(f_u_list[j], u_list[j], grad_outputs=w_t, retain_graph=True,
                                            create_graph=True)[0]
                    u_grad_vec = torch.unsqueeze(u_grad, dim=0)
                    '''
                    if not self.rademacher_type:
                        w = torch.randn(u_list[j].shape[0], 1).to(self.device)
                    else:
                        w = torch.randint(low=0, high=2, size=(u_list[j].shape[0], 1)).to(u_list[j].dtype) * 2 - 1
                        w = w.to(self.device)
                    '''
                    w = torch.transpose(w_t.unsqueeze(0), 0,
                                        1)  # We need to keep w_t and w are vectors with the same value
                    trace = torch.mm(u_grad_vec, w)
                    trace_list.append(trace)
                trace_tensor = torch.stack(trace_list, dim=0)
                trace_mean = trace_tensor.mean(0)
                trace_mean_list.append(trace_mean)
            else:
                trace_list = []
                for i in range(0, self.hutch_times):
                    if not self.rademacher_type:
                        w_t = torch.randn(h_u_list[j].shape[0]).to(self.device)
                    else:
                        w_t = torch.randint(low=0, high=2, size=(h_u_list[j].shape[0])).to(u_list[j].dtype) * 2 - 1
                        w_t = w_t.to(self.device)
                    u_grad_left = \
                        torch.autograd.grad(h_u_list[j], u_list[j], grad_outputs=w_t, retain_graph=True,
                                            create_graph=True)[0]
                    u_grad_right = \
                        torch.autograd.grad(f_u_list[j], h_u_list[j], grad_outputs=u_grad_left, retain_graph=True,
                                            create_graph=True)[0]
                    u_grad_rightvec = torch.unsqueeze(u_grad_right, dim=0)
                    '''
                    if not self.rademacher_type:
                        w = torch.randn(h_u_list[j].shape[0], 1).to(self.device)
                    else:
                        w = torch.randint(low=0, high=2, size=(h_u_list[j].shape[0], 1)).to(u_list[j].dtype) * 2 - 1
                        w = w.to(self.device)
                    '''
                    w = torch.transpose(w_t.unsqueeze(0), 0, 1)
                    trace = torch.mm(u_grad_rightvec, w)
                    trace_list.append(trace)
                trace_tensor = torch.stack(trace_list, dim=0)
                trace_mean = trace_tensor.mean(0)
                trace_mean_list.append(trace_mean)

        trace_final_tensor = torch.stack(trace_mean_list, dim=0)
        trace_final_mean = trace_final_tensor.mean(0)  # average over self.sample_times
        # u_grad = torch.autograd.functional.jacobian(self.transform, u, create_graph=True, strict=True)
        end = time.time()
        if self.print_time:
            print('Compute Trace time:', str(end - begin), 's')
        return f_u_list, trace_final_mean  # u_grad  # add to return the grad of u
