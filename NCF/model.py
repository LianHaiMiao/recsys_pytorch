import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    def __init__(self, user_num, item_num, embed_size, num_layers, dropout, model):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embed_size: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF'
        """
        self.model = model
        self.dropout = dropout

        # GMF embedding
        self.GMF_user_embedding = nn.Embedding(user_num, embed_size)
        self.GMF_item_embedding = nn.Embedding(item_num, embed_size)

        # MLP embdding
        self.MLP_user_embedding = nn.Embedding(
            user_num, embed_size * (2 ** (num_layers - 1)))
        self.MLP_item_embedding = nn.Embedding(
            item_num, embed_size * (2 ** (num_layers - 1)))

        MLP_layers = []
        for i in range(num_layers):
            input_size = embed_size * (2 ** (num_layers - i))
            MLP_layers.append(nn.Dropout(p=self.dropout))
            MLP_layers.append(nn.Linear(input_size, input_size//2))
            MLP_layers.append(nn.ReLU())

        self.MLP_layers = nn.Sequential(*MLP_layers)

        if self.model == "GMF" or self.model == "MLP":
            self.predict_layer = nn.Linear(embed_size, 1)
        else:
            self.predict_layer = nn.Linear(embed_size * 2, 1)


        self._init_weight_()

    def _init_weight_(self):
        # embedding layers
        nn.init.normal_(self.GMF_user_embedding.weight, std=0.01)
        nn.init.normal_(self.GMF_item_embedding.weight, std=0.01)
        nn.init.normal_(self.MLP_user_embedding.weight, std=0.01)
        nn.init.normal_(self.MLP_item_embedding.weight, std=0.01)

        # MLP layers
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # predict layer
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                             a=1, nonlinearity='sigmoid')


    def forward(self, user, item):
        # MLP or NeuMF
        if self.model == "MLP" or self.model == "NeuMF":
            MLP_user_embedding = self.MLP_user_embedding(user)
            MLP_item_embedding = self.MLP_item_embedding(item)
            user_item_interaction = torch.cat((MLP_user_embedding, MLP_item_embedding), -1)
            MLP_output = self.MLP_layers(user_item_interaction)

        # GMF or NeuMF
        if self.model == "GMF" or self.model == "NeuMF":
            GMF_user_embedding = self.GMF_user_embedding(user)
            GMF_item_embedding = self.GMF_item_embedding(item)
            GMF_output = GMF_user_embedding * GMF_item_embedding

        if self.model == 'GMF':
            concat_data = GMF_output
        elif self.model == 'MLP':
            concat_data = MLP_output
            # if we choose NeuMF we concat both of GMF and MLP
        else:
            concat_data = torch.cat((GMF_output, MLP_output), -1)

        # prediction layer
        prediction = self.predict_layer(concat_data)

        return prediction.view(-1)



