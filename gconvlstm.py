import torch.nn as nn
import torch
import warnings
try:
    from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
except ImportError:
    warnings.warn("Warning: 'groupy' package not found. Please install it if you need to use group equivariant CNN.")

def _hidden_z_2_p4(x):
    '''
        x: [B, C, H, W]
    '''
    embd_dim = x.shape[1]
    conv = P4ConvZ2(in_channels=embd_dim, out_channels=embd_dim, kernel_size=1, bias=False)
    conv.weight.requires_grad = False
    conv.weight[:, :, 0, 0, 0] = torch.eye(embd_dim)
    conv.to(x.device)

    return conv(x)

class GConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_input = P4ConvZ2(in_channels=self.input_dim,
                              out_channels=self.input_dim,
                              kernel_size=self.kernel_size[0],
                              padding=self.padding,
                              bias=self.bias)
        self.conv_fi = P4ConvP4(in_channels=self.input_dim + self.hidden_dim * 2,
                              out_channels=2 * self.hidden_dim,
                              kernel_size=self.kernel_size[0],
                              padding=self.padding,
                              bias=self.bias)
        self.conv_g = P4ConvP4(in_channels=self.input_dim + self.hidden_dim * 1,
                              out_channels=1 * self.hidden_dim,
                              kernel_size=self.kernel_size[0],
                              padding=self.padding,
                              bias=self.bias)
        self.conv_o = P4ConvP4(in_channels=self.input_dim + self.hidden_dim * 2,
                              out_channels=1 * self.hidden_dim,
                              kernel_size=self.kernel_size[0],
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # transform x to the g-space
        # x = self.conv_input(input_tensor)
        x = input_tensor

        combined_xh = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_xhc = torch.cat([combined_xh, c_cur], dim=1)

        # input and forget gate
        combined_conv_if = self.conv_fi(combined_xhc)
        cc_i, cc_f = torch.split(combined_conv_if, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        # c_{t+1}
        cc_g = self.conv_g(combined_xh)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        
        # output
        # print(combined_xh.shape, c_next.shape)
        combined_xhc_next = torch.cat([combined_xh, c_next], dim=1)
        cc_o = self.conv_o(combined_xhc_next)
        o = torch.sigmoid(cc_o)

        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_fi.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_fi.weight.device))


class GConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(GConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_states=None):
        """

        Parameters
        ----------

        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_states is not None:
        # check hidden state
            for hidden_state in hidden_states:
                if len(hidden_state) != 2:
                    raise ValueError('`hidden_state` must contain both hidden and cell values')
        # TODO: check size of hidden state 
        else:
            # Since the init is done in forward. Can send image size here
            hidden_states = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            # convert h, c to the g-space
            h, c = hidden_states[layer_idx]
            h = h.unsqueeze(2).repeat(repeats=(1,1,4,1,1))
            c = c.unsqueeze(2).repeat(repeats=(1,1,4,1,1))
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            # print("bingo")
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or 
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    

if __name__ == "__main__":
    x = torch.randn([1, 2, 11, 6])
    # y = _hidden_z_2_p4(x)
    # print(y.shape)
    # print((y[:, :, 0, ...] - x).max())

    model = GConvLSTM(input_dim=1,
                 hidden_dim=1,
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True, 
                 bias=True,
                 return_all_layers=False)
    

    B, T, C, H, W = 1, 7, 1, 3, 3
    x = torch.randn([B, T, C, 4, H, W])
    # print(x.shape)

    x_ = torch.rot90(x, k=1, dims=(-2, -1))
    y1 = model(x)[0][0]
    # y2 = model(x_)[0][0]
    # print("che")
    print(y1.shape)
    # print(y1[0, -1, 0])
    # print(y2[0, -1, 0])