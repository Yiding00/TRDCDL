import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class encoder_TRDCDL(nn.Module):
    def __init__(self, n_in, n_hid, n_node, do_prob=0.):
        super(encoder_TRDCDL, self).__init__()
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.fc = nn.Linear(n_hid, n_in)
        self.graph_kernel = nn.Parameter(torch.ones([1, n_node], dtype=torch.float))

    def graph_conv(self, inputs):
        x = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        A = torch.repeat_interleave(self.graph_kernel.unsqueeze(0), inputs.size(0), dim=0)
        x = torch.bmm(A, x)
                               
        return x.reshape(inputs.size(0), inputs.size(2), inputs.size(3))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1,3)
        x = inputs.reshape(inputs.size(0), -1, inputs.size(3))
        x = self.mlp1(x)
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        x = self.graph_conv(x)
        x = self.mlp2(x)
        x = self.fc(x)

        return x.reshape(inputs.size(0), inputs.size(2), inputs.size(3))



class DCV(nn.Module):
    def __init__(self, graph_kernel, n_in, n_hid, n_out=1, do_prob=0.):
        super(DCV, self).__init__()
        self.gru = nn.GRU(n_in, n_hid, batch_first=True)
        self.mlp = MLP(n_hid, 2 * n_hid, n_hid, do_prob)
        self.fc = nn.Linear(n_hid, 4*n_out)
        self.bn = nn.BatchNorm1d(n_hid)
        self.graph_kernel = nn.Parameter(graph_kernel.clone().detach())
        self.graphconv = GraphConvolution(n_hid, n_hid)

    def batch_norm(self, inputs):
        x = inputs.reshape(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.reshape(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1,3)
        x = inputs.reshape(-1, inputs.size(2), inputs.size(3))
        x, _ = self.gru(x)
        x = self.batch_norm(x)
        x = x.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        x = self.graphconv(self.graph_kernel, x)
        x = F.elu(x)
        x = self.graphconv(self.graph_kernel, x)
        x = F.elu(x)
        x.reshape(x.size(0), -1, x.size(3))
        x2 = self.fc(x)
        self.x3 = x2.reshape(inputs.size(0), inputs.size(1), inputs.size(2), -1)
        u_mean = self.x3[:,:,:,0].unsqueeze(3)
        u_var = self.x3[:,:,:,1].unsqueeze(3)
        v_mean = self.x3[:,:,:,2].unsqueeze(3)
        v_var = self.x3[:,:,:,3].unsqueeze(3)
        u_temp = torch.normal(mean=0, std=1, size=u_mean.size()).cuda()
        v_temp = torch.normal(mean=0, std=1, size=u_mean.size()).cuda()
        u = u_mean + u_temp*u_var
        v = v_mean + v_temp*v_var
        out = torch.sigmoid(u*v)
        return u, v, out


def mask_inputs(mask, inputs):
    temp = torch.repeat_interleave(mask, inputs.size(-1), 3)
    inputs = temp * inputs

    return inputs

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):#,dropout
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs):
        support = torch.matmul(inputs, self.weight)
        x = inputs.reshape(support.size(0), support.size(1), -1)
        output = torch.matmul(adj, x)
        if self.bias is not None:
            return output.reshape(inputs.size(0), inputs.size(1),inputs.size(2), inputs.size(3)) + self.bias
        else:
            return output.reshape(inputs.size(0), inputs.size(1),inputs.size(2), inputs.size(3))
