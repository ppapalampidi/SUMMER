import numpy

import torch
from modules.trainer import Trainer
from torch.nn import functional as F
import scipy.stats as stats


class BaselineTrainer(Trainer):

    def __init__(self, test_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_loader = test_loader

    def _process_batch(self, plot, script, tp_ids, scene_lengths, labels):
        y = self.model(plot, script, tp_ids, scene_lengths, self.device)
        loss = self.criterion(y.view(-1), labels.view(-1))
        loss = loss.mean()

        total_loss = loss

        return total_loss, y, labels


    def eval_epoch(self, mu, sigma):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []
        outputs = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs, _ = self._process_batch(*batch)
                m = torch.nn.Softmax(-1)
                for j in range(batch_outputs.size(0)):
                    batch_outputs[j] = m(batch_outputs[j])
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                outputs.append(batch_outputs)

        final_posteriors = []
        final_neighborhood = []
        for movie in outputs:
            final_posteriors.append([])
            final_neighborhood.append([])
            posteriors = movie.tolist()
            indices = list(range(len(posteriors[0])))
            for j, tp in enumerate(posteriors):
                final_posteriors[-1].append(tp)
                indices_of_w = [x for _, x in sorted(zip(tp, list(range(len(tp)))),
                                                     reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index-1):(top_post_index+2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[top_post_index:(top_post_index+3)]
                else:
                    top_neighborhood = indices[(top_post_index-2):(top_post_index+1)]
                final_neighborhood[-1].append(top_neighborhood)

        return numpy.array(losses).mean(axis=0), final_posteriors,\
               final_neighborhood

    def eval_epoch_test(self, mu, sigma):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []
        outputs = []

        if isinstance(self.test_loader, (tuple, list)):
            iterator = zip(*self.test_loader)
        else:
            iterator = self.test_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.test_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs, _ = self._process_batch(*batch)
                m = torch.nn.Softmax(-1)
                for j in range(batch_outputs.size(0)):
                    batch_outputs[j] = m(batch_outputs[j])
                # aggregate the losses into a single loss value
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                outputs.append(batch_outputs)

        final_posteriors = []
        final_neighborhood = []
        for movie in outputs:
            final_posteriors.append([])
            final_neighborhood.append([])
            posteriors = movie.tolist()
            indices = list(range(len(posteriors[0])))
            for j, tp in enumerate(posteriors):
                final_posteriors[-1].append(tp)
                indices_of_w = [x for _, x in sorted(zip(tp, list(range(len(tp)))), reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index-1):(top_post_index+2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[top_post_index:(top_post_index+3)]
                else:
                    top_neighborhood = indices[(top_post_index-2):(top_post_index+1)]
                final_neighborhood[-1].append(top_neighborhood)

        return numpy.array(losses).mean(axis=0), final_posteriors,\
               final_neighborhood



class BaselineTrainer_Screenplays(BaselineTrainer):

    def _process_batch(self, plot, script, tp_ids, scene_lengths, labels):
        y = self.model(script, scene_lengths, self.device)
        loss = self.criterion(y.view(-1), labels.view(-1))

        loss = loss.mean()
        total_loss = loss

        return total_loss, y, labels


class Trainer_SUMMER(Trainer):

    def __init__(self, test_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_loader = test_loader

    def _process_batch(self, script, scene_lengths, labels):
        y, sa = self.model(script, scene_lengths, self.device)
        loss = self.criterion(y, labels)
        loss = loss.mean()

        # focal regularization

        mu = [0.1139, 0.3186, 0.5065, 0.7415, 0.8943]
        sigma = [0.05, 0.05, 0.05, 0.05, 0.05]

        focal_loss = 0

        for i in range(len(mu)):
            dist = torch.from_numpy(
                stats.norm.pdf(list(range(sa[i].size(0))),
                               mu[i] * sa[i].size(0),
                               sigma[i] * sa[i].size(0))). \
                float().cuda(sa.get_device())
            dist = F.softmax(dist / 0.005, dim=0)
            loss_now = F.kl_div(torch.log(sa[i]), dist)
            focal_loss += loss_now

        epsilon = 0.0000001

        #orthogonal regularization

        orthogonal_loss = 0
        for i in range(sa.size(0)):
            for j in range(sa.size(0)):
                if i != j:
                    _loss = F.kl_div(torch.log(sa[i]), sa[j])
                    _loss = torch.log(torch.abs(1/(_loss + epsilon)))
                    orthogonal_loss += _loss

        total_loss = loss + 0.1*focal_loss + 0.15*orthogonal_loss

        return total_loss, y, labels

    def eval_epoch(self, mu, sigma, compression_rate):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader
        outputs = []
        pre = []
        re = []
        f1 = []

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))
                batch_losses, batch_outputs, labels = self._process_batch(*batch)
                # aggregate the losses into a single loss value
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                labels = labels.data.cpu().numpy()
                outputs.append(batch_outputs)
                gs_summary = []
                for m, label in enumerate(labels):
                    if label == 1:
                        gs_summary.append(m)
                m = torch.nn.Sigmoid()
                raw_batch_outputs = m(batch_outputs)
                raw_batch_outputs = raw_batch_outputs.data.cpu().numpy()
                indices_of_w = [x for _, x in
                                sorted(zip(raw_batch_outputs,
                                           list(range(len(raw_batch_outputs)))),
                                       reverse=True)]

                pr_summary = indices_of_w[:int(compression_rate * len(raw_batch_outputs))]

                right_now = len(list(set(gs_summary).intersection(pr_summary)))
                overall_now = len(gs_summary)
                predicted_now = len(pr_summary)
                try:
                    recall_now = right_now/overall_now
                except:
                    recall_now = 0
                re.append(recall_now)
                try:
                    precision_now = right_now/predicted_now
                except:
                    precision_now = 0
                pre.append(precision_now)
                try:
                    f1.append((2*precision_now*recall_now)/(precision_now + recall_now))
                except:
                    f1.append(0)

        return numpy.array(losses).mean(axis=0), numpy.asarray(f1).mean(), \
               numpy.asarray(pre).mean(), \
               numpy.asarray(re).mean()

    def eval_epoch_test(self, mu, sigma, compression_rate):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if isinstance(self.test_loader, (tuple, list)):
            iterator = zip(*self.test_loader)
        else:
            iterator = self.test_loader
        outputs = []
        pre = []
        re = []
        f1 = []

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.test_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))
                batch_losses, batch_outputs, labels = self._process_batch(
                    *batch)
                # aggregate the losses into a single loss value
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                labels = labels.data.cpu().numpy()
                outputs.append(batch_outputs)
                gs_summary = []
                for m, label in enumerate(labels):
                    if label == 1:
                        gs_summary.append(m)
                m = torch.nn.Sigmoid()
                raw_batch_outputs = m(batch_outputs)
                raw_batch_outputs = raw_batch_outputs.data.cpu().numpy()
                indices_of_w = [x for _, x in
                                sorted(zip(raw_batch_outputs,
                                           list(range(len(raw_batch_outputs)))),
                                       reverse=True)]

                pr_summary = indices_of_w[
                             :int(compression_rate * len(raw_batch_outputs))]

                right_now = len(list(set(gs_summary).intersection(pr_summary)))
                overall_now = len(gs_summary)
                predicted_now = len(pr_summary)
                try:
                    recall_now = right_now / overall_now
                except:
                    recall_now = 0
                re.append(recall_now)
                try:
                    precision_now = right_now / predicted_now
                except:
                    precision_now = 0
                pre.append(precision_now)
                try:
                    f1.append((2 * precision_now * recall_now) / (
                    precision_now + recall_now))
                except:
                    f1.append(0)

        return numpy.array(losses).mean(axis=0), numpy.asarray(f1).mean(), \
               numpy.asarray(pre).mean(), \
               numpy.asarray(re).mean()

