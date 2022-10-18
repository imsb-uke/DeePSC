import torch
import torch.nn as nn
import torchvision.models as tvmodels


class SVCNN(nn.Module):
    def __init__(
        self,
        nclasses=1,
        in_chans=3,
        drop_rate=0.5,
    ):
        super(SVCNN, self).__init__()

        self.nclasses = nclasses
        self.in_chans = in_chans
        self.drop_rate = drop_rate

        self.net = tvmodels.squeezenet1_1(
            pretrained=True,
            num_classes=1000,
        )
        self.net = self._as_sequential_squeezenet(self.net)

    def forward(self, x):
        return self.net(x)

    def _as_sequential_squeezenet(self, model):
        layers = list(model.features.children())
        global_pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten(1)
        dropout = nn.Dropout(self.drop_rate)
        classifier = nn.Linear(in_features=512, out_features=self.nclasses, bias=True)
        layers.extend([global_pool, flatten, dropout, classifier])
        return nn.Sequential(*layers)


class MVCNN(nn.Module):
    def __init__(
        self,
        single_model,
        num_views=7,
    ):
        super(MVCNN, self).__init__()

        self.num_views = num_views
        self.drop_rate = single_model.drop_rate

        self.net_1 = single_model.net[:-2]
        self.view_fusion_layer = AttentionFusionLayer(512)
        self.net_2 = single_model.net[-2:]

    def forward(self, x):

        y = self.net_1(x)
        y = y.contiguous().view(-1, self.num_views, *y.shape[2:])  # BxNxL
        y, _ = self.view_pooling_layer(y)
        y = self.net_2(y)

        return y


class AttentionFusionLayer(nn.Module):
    """
    Based on https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
    et k = 1 if only label for the entire bag is relevant, set k=n_patches/instances to get one prediction per instance

    input shape: BxNxL
    output shape: BxKxL
    """

    def __init__(self, l, d=64, k=1, **kwargs):
        super().__init__(**kwargs)
        self.l = l
        self.d = d
        self.k = k
        self.attention_dense1 = nn.Linear(l, d)
        self.attention_tanh = nn.Tanh()
        self.attention_dense2 = nn.Linear(d, self.k)
        self.attention_softmax = nn.Softmax(dim=2)

    def forward(self, inputs):

        # FC layer is applied on last dimension of input only (can be otherwise arbitrary shape)
        alpha = self.attention_dense1(inputs)  # -> BxNxD
        alpha = self.attention_tanh(alpha)  # -> BxNxD
        alpha = self.attention_dense2(alpha)  # -> BxNxK
        alpha = torch.transpose(alpha, 2, 1)  # -> BxKxN
        alpha = self.attention_softmax(alpha)  # -> BxKxN

        y = torch.bmm(alpha, inputs)  # BxKxL

        return y, alpha