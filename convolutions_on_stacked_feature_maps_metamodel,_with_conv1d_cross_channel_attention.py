class Meta_model(nn.Module):
    def __init__(self, input_channels, img_height, img_width, base_model_specs, token_dim=None, meta_channels=2, meta_kernel=3, meta_dilation=1, attention_kernel=1, meta_fc_layers=[10], meta_dropout=0.0):
        super().__init__()

        self.token_dim = token_dim
        self.base_model_specs = base_model_specs
        self.num_base_models = len(base_model_specs)
        self.model_specs = nn.ModuleList()
        self.meta_channels = meta_channels
        self.meta_kernel_size = meta_kernel
        self.attention_kernel = attention_kernel
        self.meta_dropout=meta_dropout
        self.meta_dilation = meta_dilation
        channel_cardinalities = []

        for model in base_model_specs:
            cnn = CNN(input_channels, img_height, img_width, block_specs=model[0],
                      pooling_kernel=model[1], pooling_stride=model[2], pooling_padding=model[3],
                      fc_layers=model[4], fc_dropout=model[5], token_dim=token_dim)
            self.model_specs.append(cnn)
            _, _, _, channels = cnn.pass_final_dimensions()
            channel_cardinalities.append(channels)

        self.num_base_models = sum(channel_cardinalities)

        self.conv = nn.Conv1d(
            in_channels=self.num_base_models,
            out_channels=self.meta_channels,
            kernel_size=self.meta_kernel_size,
            dilation=self.meta_dilation,
            padding=0
        )

        # Conv1d attention:
        self.channel_attn_layer = nn.Conv1d(
            in_channels=self.meta_channels,
            out_channels=self.meta_channels,
            kernel_size=self.attention_kernel
        )

        self.conv_layer_length = (self.token_dim - self.meta_dilation * (self.meta_kernel_size - 1))
        self.flat_dim = self.meta_channels * self.conv_layer_length
        self.meta_fc_blocks = nn.ModuleList()
        meta_input_dim = self.flat_dim
        for i in range(len(meta_fc_layers) - 1):
            self.meta_fc_blocks.append(nn.Linear(meta_input_dim, meta_fc_layers[i]))
            self.meta_fc_blocks.append(nn.ReLU())
            if meta_dropout > 0:
                self.meta_fc_blocks.append(nn.Dropout(p=meta_dropout))
            meta_input_dim = meta_fc_layers[i]
        self.meta_fc_blocks.append(nn.Linear(meta_input_dim, meta_fc_layers[-1]))
        self.meta_block_sequence = nn.Sequential(*self.meta_fc_blocks)

    def forward(self, x):
        features = []
        for i, model in enumerate(self.model_specs):
            token = model(x)
            features.append(token)
        x_stack = torch.cat(features, dim=1)
        x_stack = self.conv(x_stack)
        x_stack = F.relu(x_stack)

        attention = self.channel_attn_layer(x_stack)
        attention = torch.sigmoid(attention)

        x_stack = x_stack * attention
        x_stack = x_stack.flatten(start_dim=1)
        x_stack = self.meta_block_sequence(x_stack)
        return x_stack
