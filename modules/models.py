import torch
from torch import nn
from torch.nn import functional as F

from modules.layers import SelfAttention, MLP


class ModelHelper:
    @staticmethod
    def plot_script_interaction(plot, script):
        _plot = plot.unsqueeze(1).repeat(1, script.size(0), 1)
        _script = script.unsqueeze(0).repeat(plot.size(0), 1, 1)

        product = _plot * _script
        diff = _plot - _script
        cos = F.cosine_similarity(_script, _plot, dim=2)
        out = torch.cat([_script, _plot, product, diff, cos.unsqueeze(2)], -1)

        return out

    @staticmethod
    def script_context_interaction(script, context):
        product = context * script
        diff = context - script
        cos = F.cosine_similarity(script, context, dim=1)
        out = torch.cat([script, context, product, diff, cos.unsqueeze(1)], -1)

        return out

    @staticmethod
    def scene_tp_interaction(script, context):
        product = context * script
        diff = context - script
        cos = F.cosine_similarity(script, context, -1)
        out = torch.cat([product, diff, cos.unsqueeze(1)], -1)
        return out


class TAM(nn.Module, ModelHelper):
    def __init__(self, config, window_length):
        super(TAM, self).__init__()

        self.window_length = window_length
        plot_size = config["plot_encoder_size"]
        if config["plot_encoder_bidirectional"]:
            plot_size *= 2

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2
        # --------------------------------------------------
        # Plot Encoder
        # --------------------------------------------------
        # Reads a sequence of plot sentences and produces
        # context-aware plot sentence representations
        # --------------------------------------------------
        self.plot_encoder = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["plot_encoder_size"],
            num_layers=config["plot_encoder_layers"],
            bidirectional=config["plot_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Scene Encoder
        # --------------------------------------------------
        # Reads a sequence of scene sentences and produces
        # context-aware scene sentence representations
        # --------------------------------------------------
        self.scene_encoder = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["scene_encoder_size"],
            num_layers=config["scene_encoder_layers"],
            bidirectional=config["scene_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Scene Attention
        # --------------------------------------------------
        # Self-attention over the sentences contained in a scene
        # --------------------------------------------------
        self.scene_attention = SelfAttention(scene_size)
        # --------------------------------------------------
        # Script Encoder
        # --------------------------------------------------
        # Reads a sequence of scenes and produces
        # context-aware scene representations
        # --------------------------------------------------
        self.script_encoder = nn.LSTM(
            input_size=scene_size,
            hidden_size=config["script_encoder_size"],
            num_layers=config["script_encoder_layers"],
            bidirectional=config["script_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Projection
        # --------------------------------------------------
        # project plot sentence representations to the same
        # dimension as the final scene representations
        # --------------------------------------------------
        self.projection = nn.Linear(plot_size, plot_size * 3)

        interaction_size = plot_size * 3 + script_size * 3 * 3 + 1

        self.prediction = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])

        # self.prediction = nn.Linear(interaction_size,1)

        print(self)

    def forward(self, plot, script, tp_ids, scene_lens, device):
        """

        Args:
            plot: plot synopsis; sentence-level encodings
            script: screenplay; scene-level vectors as sequence of sentence encodings
            tp_ids: gold-standard sentence-level TP annotations in the plot synopses
            scene_lens: true length of screenplay scenes as number of sentences

        Returns: y: scene-level logits per TP; shape: 5 x number_of_scenes

        """

        # ----------------------------------------------------------
        # 1 - encode the sentences of the plot
        # ----------------------------------------------------------
        plot_outs, _ = self.plot_encoder(plot.unsqueeze(0))
        plot_embeddings = plot_outs.squeeze()
        # keep only the sentences that are annotated as TPs
        tp_embeddings = plot_embeddings[tp_ids]
        # ----------------------------------------------------------
        # 2 - encode the sentences of the scenes
        # ----------------------------------------------------------
        script_outs, _ = self.scene_encoder(script)

        scene_embeddings, sa = self.scene_attention(script_outs, scene_lens)
        # ----------------------------------------------------------
        # 3 - contextualize the scene representations
        # ----------------------------------------------------------
        script_outs, _ = self.script_encoder(scene_embeddings.unsqueeze(0))
        script_embeddings = script_outs.squeeze()

        num_scenes = script_embeddings.size(0)
        left_context = []
        right_context = []

        for i in range(num_scenes):
            if i == 0:
                if device == torch.device("cuda"):
                    _left_context = torch.zeros((script_embeddings.size(1))). \
                                    cuda(script_embeddings.get_device())
                else:
                    _left_context = torch.zeros((script_embeddings.size(1)))
            elif i < int(self.window_length * num_scenes):
                _left_context = torch.mean(script_embeddings.narrow(0, 0, i), dim=0)
            else:
                _left_context = torch.mean(script_embeddings.
                                               narrow(0,
                                                      i - int(self.window_length * num_scenes),
                                                      int(self.window_length * num_scenes)),
                                               dim=0)

            if i == (num_scenes - 1):
                if device == torch.device("cuda"):
                    _right_context = torch.zeros((script_embeddings.size(1))). \
                                     cuda(script_embeddings.get_device())
                else:
                    _right_context = torch.zeros((script_embeddings.size(1)))
            elif i > (num_scenes - 1 - int(self.window_length * num_scenes)):
                _right_context = torch.mean(script_embeddings.
                                                narrow(0, i, (num_scenes - i)),
                                                dim=0)
            else:
                _right_context = torch.mean(script_embeddings.narrow(0,
                                                                     i,
                                                                     int(self.window_length * num_scenes)),
                                                dim=0)

            left_context.append(_left_context)
            right_context.append(_right_context)

        left_context = torch.stack(left_context, dim=0)
        right_context = torch.stack(right_context, dim=0)

        u = torch.cat([script_embeddings, left_context, right_context], -1)

        # ----------------------------------------------------------
        # 4 - compute the interaction between each TP plot sentence
        # and all scenes
        # ----------------------------------------------------------
        tp_embeddings = self.projection(tp_embeddings)

        u = self.plot_script_interaction(tp_embeddings, u)

        y = self.prediction(u).squeeze()

        return y


class TAM_Screenplays(nn.Module,ModelHelper):
    def __init__(self, config, window_length, temperature):
        super(TAM_Screenplays, self).__init__()

        self.window_length = window_length
        self.temperature = temperature

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2
        # --------------------------------------------------
        # Scene Encoder
        # --------------------------------------------------
        # Reads a sequence of scene sentences and produces
        # context-aware scene sentence representations
        # --------------------------------------------------
        self.scene_encoder = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["scene_encoder_size"],
            num_layers=config["scene_encoder_layers"],
            bidirectional=config["scene_encoder_bidirectional"],
            batch_first=True)
        # --------------------------------------------------
        # Scene Attention
        # --------------------------------------------------
        # Self-attention over the sentences contained in a scene
        # --------------------------------------------------
        self.scene_attention = SelfAttention(scene_size)
        # --------------------------------------------------
        # Script Encoder
        # --------------------------------------------------
        # Reads a sequence of scenes and produces
        # context-aware scene representations
        # --------------------------------------------------
        self.script_encoder = nn.LSTM(
            input_size=scene_size,
            hidden_size=config["script_encoder_size"],
            num_layers=config["script_encoder_layers"],
            bidirectional=config["script_encoder_bidirectional"],
            batch_first=True)

        interaction_size = (script_size * 4 + 1) * 2 + script_size
        # --------------------------------------------------
        # TP-specific prediction layers
        # --------------------------------------------------
        self.prediction_1 = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])
        self.prediction_2 = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])
        self.prediction_3 = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])
        self.prediction_4 = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])
        self.prediction_5 = MLP(interaction_size,
                               layers=config["interaction_layers"],
                               dropout=config["interaction_dropout"],
                               non_linearity=config["interaction_activation"])

        self.softmax = nn.Softmax(dim=-1)

        print(self)

    def forward(self, script, scene_lens, device):
        """

        Args:
            script: screenplay; scene-level vectors as sequence of sentence encodings
            scene_lens: true length of screenplay scenes as number of sentences

        Returns: y: scene-level logits per TP; shape: 5 x number_of_scenes

        """
        # ----------------------------------------------------------
        # 1 - encode the sentences of the scenes
        # ----------------------------------------------------------
        script_outs, _ = self.scene_encoder(script)

        scene_embeddings, sa = self.scene_attention(script_outs, scene_lens)
        # ----------------------------------------------------------
        # 2 - contextualize the scene representations
        # ----------------------------------------------------------
        script_outs1, _ = self.script_encoder(scene_embeddings.unsqueeze(0))
        script_embeddings = script_outs1.squeeze()

        num_scenes = script_embeddings.size(0)

        left_context = []
        right_context = []
        for i in range(num_scenes):

            if i == 0:
                if device == torch.device("cuda"):
                    _left_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _left_context = torch.zeros((script_embeddings.size(1)))
            elif i < int(self.window_length * num_scenes):
                _left_context = torch.mean(
                    script_embeddings.narrow(0, 0, i), dim=0)
            else:
                _left_context = torch.mean(script_embeddings.
                                           narrow(0,
                                                  i - int(self.window_length * num_scenes),
                                                  int(self.window_length * num_scenes)),
                                           dim=0)

            if i == (num_scenes - 1):
                if device == torch.device("cuda"):
                    _right_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _right_context = torch.zeros((script_embeddings.size(1)))
            elif i > (
                    num_scenes - 1 - int(self.window_length * num_scenes)):
                _right_context = torch.mean(script_embeddings.
                                            narrow(0, i, (num_scenes - i)),
                                            dim=0)
            else:
                _right_context = torch.mean(script_embeddings.narrow(0,
                                                                     i,
                                                                     int(self.window_length * num_scenes)),
                                            dim=0)

            left_context.append(_left_context)
            right_context.append(_right_context)

        left_context = torch.stack(left_context, dim=0)
        right_context = torch.stack(right_context, dim=0)

        left_interaction = self.script_context_interaction(
            script_embeddings, left_context)
        right_interaction = self.script_context_interaction(
            script_embeddings, right_context)

        u = torch.cat(
            [script_embeddings, left_interaction, right_interaction], -1)
        # ----------------------------------------------------------
        # 3 - probability of each scene to represent each TP
        # ----------------------------------------------------------
        y = []
        # number of output posterior distributions = number of TPs = 5
        for i in range(5):
            if i == 0:
                y_now = self.prediction_1(u)
            elif i == 1:
                y_now = self.prediction_2(u)
            elif i == 2:
                y_now = self.prediction_3(u)
            elif i == 3:
                y_now = self.prediction_4(u)
            else:
                y_now = self.prediction_5(u)
            y_now = torch.transpose(y_now, 0, 1)
            y_now = self.softmax(y_now/self.temperature).squeeze()
            y.append(y_now.squeeze())

        y = torch.stack(y, dim=0)

        return y


class SUMMER_unsupervised(nn.Module, ModelHelper):
    def __init__(self, config, window_length, pretrained_model, lambda_1, beta,
                 compression_rate, temperature):
        super(SUMMER_unsupervised, self).__init__()
        # --------------------------------------------------
        # Parameter settings
        # --------------------------------------------------
        self.window_length = window_length
        self.lambda_1 = lambda_1
        self.beta = beta
        self.compression_rate = compression_rate
        self.temperature = temperature

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2
        # --------------------------------------------------
        # Pretrained model on TP identification
        # --------------------------------------------------
        self.scene_encoder = pretrained_model.scene_encoder

        self.scene_attention = pretrained_model.scene_attention

        self.script_encoder = pretrained_model.script_encoder

        self.attention1 = pretrained_model.prediction_1

        self.attention2 = pretrained_model.prediction_2

        self.attention3 = pretrained_model.prediction_3

        self.attention4 = pretrained_model.prediction_4

        self.attention5 = pretrained_model.prediction_5

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, script, scene_lens, device):
        """

        Args:
            script: sequence of screenplay scenes as sequence of sentence
                    representations
            scene_lens: real length of scenes as actual number of sentences
            forward_lambda: weight attributed to forward pass, accepted values
                            in [1,10], default value=7
            compression_rate: % of total number of scenes to be included
                              in the summary, default=0.3

        Returns: y: centrality scores for the screenplay scenes, size num_scenes
                 extracted: scene indices that have been selected as summary
                            based on the given compression rate

        """
        # ----------------------------------------------------------
        # Scene embeddings for constructing the graph in TextRank
        # ----------------------------------------------------------
        scene_embeddings = script.mean(1)
        # ----------------------------------------------------------
        # Narrative structure identification - TAM
        # ----------------------------------------------------------
        script_outs, _ = self.scene_encoder(script)
        scene_embeddings_for_TPs, sa = self.scene_attention(
            script_outs, scene_lens)
        script_outs, _ = self.script_encoder(
            scene_embeddings_for_TPs.unsqueeze(0))
        script_embeddings = script_outs.squeeze()

        num_scenes = script_embeddings.size(0)

        left_context = []
        right_context = []
        for i in range(num_scenes):

            if i == 0:
                if device == torch.device("cuda"):
                    _left_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _left_context = torch.zeros((script_embeddings.size(1)))
            elif i < int(self.window_length * num_scenes):
                _left_context = torch.mean(
                    script_embeddings.narrow(0, 0, i), dim=0)
            else:
                _left_context = torch.mean(script_embeddings.
                                           narrow(0,
                                                  i - int(self.window_length * num_scenes),
                                                  int(self.window_length * num_scenes)),
                                           dim=0)

            if i == (num_scenes - 1):
                if device == torch.device("cuda"):
                    _right_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _right_context = torch.zeros((script_embeddings.size(1)))
            elif i > (
                            num_scenes - 1 - int(
                            self.window_length * num_scenes)):
                _right_context = torch.mean(script_embeddings.
                                            narrow(0, i, (num_scenes - i)),
                                            dim=0)
            else:
                _right_context = torch.mean(script_embeddings.narrow(0,
                                                                     i,
                                                                     int(self.window_length * num_scenes)),
                                            dim=0)

            left_context.append(_left_context)
            right_context.append(_right_context)

        left_context = torch.stack(left_context, dim=0)
        right_context = torch.stack(right_context, dim=0)

        left_interaction = self.script_context_interaction(
            script_embeddings, left_context)
        right_interaction = self.script_context_interaction(
            script_embeddings, right_context)

        u = torch.cat(
            [script_embeddings, left_interaction, right_interaction], -1)
        # ----------------------------------------------------------
        # TP-weight calculation for each screenplay scene
        # ----------------------------------------------------------
        attentions = []
        epsilon = 0.00000000001

        # number of output posterior distributions = number of TPs = 5
        for i in range(5):
            if i == 0:
                energies = self.attention1(u)
            elif i == 1:
                energies = self.attention2(u)
            elif i == 2:
                energies = self.attention3(u)
            elif i == 3:
                energies = self.attention4(u)
            else:
                energies = self.attention5(u)
            energies = torch.transpose(energies, 0, 1)
            energies = self.softmax(energies/self.temperature).squeeze()
            attentions.append(energies + epsilon)
        # ----------------------------------------------------------
        # Modified TextRank w/ narrative structure info
        # ----------------------------------------------------------
        attentions = torch.stack(attentions, dim=0)
        attentions = torch.max(attentions, dim=0)[0]

        similarity_matrix = torch.zeros(scene_embeddings.size(0),
                                        scene_embeddings.size(0)). \
            cuda(scene_embeddings.get_device())

        for i in range(scene_embeddings.size(0)):
            for j in range(scene_embeddings.size(0)):
                if i == j:
                    continue
                score = F.cosine_similarity(scene_embeddings[i],
                                            scene_embeddings[j], 0)
                similarity_matrix[i, j] = score

        min_score = similarity_matrix.min()
        max_score = similarity_matrix.max()
        edge_threshold = min_score + self.beta * (max_score - min_score)

        forward_scores = [0 for i in range(len(similarity_matrix))]
        backward_scores = [0 for i in range(len(similarity_matrix))]
        edges = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[i])):
                edge_score = similarity_matrix[i][j]
                if edge_score > edge_threshold:
                    forward_scores[j] += (edge_score + attentions[i])
                    backward_scores[i] += (edge_score + attentions[i])
                    edges.append((i, j, edge_score))

        lambda1 = self.lambda_1 / 10
        lambda2 = 1 - lambda1

        paired_scores = []
        y = []
        for node in range(len(forward_scores)):
            paired_scores.append(
                [node, lambda1 * (forward_scores[node]) + lambda2 * (
                    backward_scores[node])])
            y.append(lambda1 * (forward_scores[node]) + lambda2 * (
                backward_scores[node]))

        paired_scores.sort(key=lambda x: x[1], reverse=True)
        extracted = [item[0] for item in
                     paired_scores[:int(self.compression_rate *
                                        scene_embeddings.size(0))]]
        y = torch.stack(y, dim=0)

        return y, extracted


class SUMMER_supervised(nn.Module, ModelHelper):
    def __init__(self, config, window_length, pretrained_model, temperature):
        super(SUMMER_supervised, self).__init__()
        # --------------------------------------------------
        # Parameter settings
        # --------------------------------------------------
        self.window_length = window_length
        self.temperature = temperature

        scene_size = config["scene_encoder_size"]
        if config["scene_encoder_bidirectional"]:
            scene_size *= 2

        script_size = config["script_encoder_size"]
        if config["script_encoder_bidirectional"]:
            script_size *= 2
        # --------------------------------------------------
        # Pretrained model on TP identification
        # fine-tuned on CSI summarization
        # --------------------------------------------------
        self.scene_encoder = pretrained_model.scene_encoder

        self.scene_attention = pretrained_model.scene_attention

        self.script_encoder = pretrained_model.script_encoder

        self.attention1 = pretrained_model.prediction_1

        self.attention2 = pretrained_model.prediction_2

        self.attention3 = pretrained_model.prediction_3

        self.attention4 = pretrained_model.prediction_4

        self.attention5 = pretrained_model.prediction_5

        self.softmax = nn.Softmax(dim=-1)
        # --------------------------------------------------
        # Summarization specific layer
        # for calculating probability of each scene to be
        # part of the summary
        # --------------------------------------------------
        attention_size = (script_size * 4 + 1) * 2 + script_size

        final_size = script_size + (attention_size * 2 + 1)

        self.final_layer = nn.Linear(final_size, 1)

    def forward(self, script,  scene_lens, device):
        """

        Args:
            script: sequence of screenplay scenes as sequence of sentence
                    representations
            scene_lens: real length of scenes as actual number of sentences

        Returns: y: logits for each scene that represent whether the scene
                 is part of the summary, size num_scenes
                 attentions: probability distribution over the screenplay scenes
                 to act as a specific TP, size num_scenes

        """
        # ----------------------------------------------------------
        # 1 - encode the sentences of the scenes (part of TAM)
        # ----------------------------------------------------------
        script_outs, _ = self.scene_encoder(script)
        scene_embeddings, sa = self.scene_attention(script_outs, scene_lens)
        # ----------------------------------------------------------
        # 2 - contextualize the scene representations (part of TAM)
        # ----------------------------------------------------------
        script_outs, _ = self.script_encoder(
            scene_embeddings.unsqueeze(0))
        script_embeddings = script_outs.squeeze()
        num_scenes = script_embeddings.size(0)

        left_context = []
        right_context = []
        for i in range(num_scenes):

            if i == 0:
                if device == torch.device("cuda"):
                    _left_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _left_context = torch.zeros((script_embeddings.size(1)))
            elif i < int(self.window_length * num_scenes):
                _left_context = torch.mean(
                    script_embeddings.narrow(0, 0, i), dim=0)
            else:
                _left_context = torch.mean(script_embeddings.
                                           narrow(0,
                                                  i - int(self.window_length * num_scenes),
                                                  int(self.window_length * num_scenes)),
                                           dim=0)

            if i == (num_scenes - 1):
                if device == torch.device("cuda"):
                    _right_context = torch.zeros((script_embeddings.size(1))). \
                    cuda(script_embeddings.get_device())
                else:
                    _right_context = torch.zeros((script_embeddings.size(1)))
            elif i > (
                    num_scenes - 1 - int(self.window_length * num_scenes)):
                _right_context = torch.mean(script_embeddings.
                                            narrow(0, i, (num_scenes - i)),
                                            dim=0)
            else:
                _right_context = torch.mean(script_embeddings.narrow(0,
                                                                     i,
                                                                     int(self.window_length * num_scenes)),
                                            dim=0)

            left_context.append(_left_context)
            right_context.append(_right_context)

        left_context = torch.stack(left_context, dim=0)
        right_context = torch.stack(right_context, dim=0)

        left_interaction = self.script_context_interaction(
            script_embeddings, left_context)
        right_interaction = self.script_context_interaction(
            script_embeddings, right_context)

        u = torch.cat(
            [script_embeddings, left_interaction, right_interaction], -1)

        tps_embeddings = []
        attentions = []
        epsilon = 0.00000000001
        # ----------------------------------------------------------
        # 3 - probability of each scene to represent each TP (part of TAM)
        # ----------------------------------------------------------
        # number of output posterior distributions = number of TPs = 5
        for i in range(5):
            if i == 0:
                energies = self.attention1(u)
            elif i == 1:
                energies = self.attention2(u)
            elif i == 2:
                energies = self.attention3(u)
            elif i == 3:
                energies = self.attention4(u)
            else:
                energies = self.attention5(u)
            energies = torch.transpose(energies, 0, 1)
            energies = self.softmax(energies /self.temperature).squeeze()
            attentions.append(energies + epsilon)

            contexts = (u * energies.unsqueeze(-1))
            contexts = torch.mean(contexts, dim=0)
            tps_embeddings.append(contexts)

        tps_embeddings = torch.stack(tps_embeddings, dim=0)
        attentions = torch.stack(attentions, dim=0)
        # ----------------------------------------------------------
        # 4 - measure similarity of each scene with each tp representation
        # ----------------------------------------------------------
        similarities = []
        for i in range(tps_embeddings.size(0)):
            context = tps_embeddings[i].unsqueeze(0).repeat(script.size(0), 1)

            interaction = self.scene_tp_interaction(u, context)
            similarities.append(interaction)
        # ----------------------------------------------------------
        # 5 - max pooling; concat similarities and content representations
        # ----------------------------------------------------------
        similarities = torch.stack(similarities, dim=1)
        similarities = torch.max(similarities, dim=1)[0]
        y = torch.cat((similarities, script_embeddings), dim=-1)
        # ----------------------------------------------------------
        # 6 - probability of each scene to be in the summary
        # ----------------------------------------------------------
        y = self.final_layer(y).squeeze()

        return y, attentions