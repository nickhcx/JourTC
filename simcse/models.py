import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import sys


from transformers.activations import gelu
from transformers.file_utils import (add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward,
                                     replace_return_docstrings)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import (BertLMPredictionHead,
                                                    BertModel,
                                                    BertPreTrainedModel)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead, RobertaModel, RobertaPreTrainedModel)
import math



class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    do_mask=False,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)
    anc_input_ids = input_ids[:,0,:].clone().detach() #experimental stop-gradient
    anc_attention_mask = attention_mask[:,0,:]

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0)) #[bsz/2, 1, nh] * [1, bsz/2, nh] -> [bsz/2, bsz/2]



    #consistency loss
    # cst_loss_fct = nn.SmoothL1Loss(beta=cls.model_args.huber_delta)
    # cst_loss_fct = nn.KLDivLoss()
    fn_loss = None

    #kmeans clustering
    if cls.model_args.kmeans > 0:
        fn_loss = None
        normalized_cos = cos_sim * cls.model_args.temp
        avg_cos = normalized_cos.mean().item()
        # z12 = torch.cat([z1, z2], dim=0)
        if not cls.cluster.initialized:
            if avg_cos <= cls.model_args.kmean_cosine:
                # with torch.no_grad():
                #     all_cos_sim = cls.sim(z12.unsqueeze(1), z12.unsqueeze(0))
                cls.cluster.optimized_centroid_init(z1, cos_sim*cls.model_args.temp)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("RGMM start!")
        elif cls.cluster.initialized:
            # if cls.cluster.global_step % 100 == 0:
            #     if not dist.is_initialized() or dist.get_rank() == 0:
            #         print(cls.cluster.centroid.data[0][:4].tolist())
            cls.cluster(z1, normalized_cos)
            num_sent = 3 #to be fix
            z3, fn_loss, z3_p = cls.gmm.GMM_EM(z1, normalized_cos)





            # cos_sim_mask = cos_sim_mask==0
            # cos_sim = cos_sim + cos_sim_mask * -10000
            # cos_sim = cos_sim * cos_sim_mask.float()
        cls.cluster.global_step += 1

    # Hard negative
    if num_sent >= 3:

        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0)) +torch.log(z3_p)
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  #to be fix

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight

        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights


    loss = loss_fct(cos_sim, labels)
    if fn_loss is not None:
        loss = loss + cls.model_args.bml_weight * fn_loss

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)
            self.gmm = GMM(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        if self.model_args.dropout_prob is not None:
            config.attention_probs_dropout_prob = self.model_args.dropout_prob
            config.hidden_dropout_prob = self.model_args.dropout_prob
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        if self.model_args.kmeans > 0:
            self.cluster = kmeans_cluster(config, self.model_args)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )




class GMM:
    def __init__(self, config, model_args,weights = None,means = None,covars = None):

        super().__init__()
        self.model_args = model_args
        #self.K_GMM = model_args.kmeans
        self.K_GMM = 24
        self.sim = Similarity(temp=1)

        if weights is not None:
            self.weights = weights
        else:



            self.weights = torch.tensor([1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24]).cuda()


        self.dim = 192
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K_GMM):
                mean = torch.randn(self.dim).cuda()
                self.means.append(mean)

        if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K_GMM):

                cov = torch.eye(self.dim).cuda() * torch.rand(1).cuda()
                self.covars.append(cov)
        self.beta = model_args.bml_beta
        self.alpha = model_args.bml_alpha


    def Gaussian(self,x,mean,cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = cov.size()[0]

        min_value = cov.min()
        max_value = cov.max()

        cov = (cov - min_value) / (max_value - min_value)

        tensor_upper = torch.triu(cov)
        cov = tensor_upper.T + tensor_upper - torch.diag(cov.diagonal())

        if torch.min(torch.diag(cov)) <= 0.0:
            cov = cov - torch.eye(dim).cuda() * torch.min(torch.diag(cov)) + torch.eye(dim).cuda() * 0.0001

        if torch.min(torch.diag(cov)) == 0.0:
            cov = cov + torch.eye(dim).cuda() * 0.0001


        mvn = torch.distributions.MultivariateNormal(mean.float(), cov.float())


        log_probs = mvn.log_prob(x)

        has_nan = torch.isnan(log_probs).any().item()

        return log_probs


    def Gaussian1(self,x,mean,cov):

        #print("mean"+str(mean))

        dim = cov.size()[0]

        min_value = cov.min()
        max_value = cov.max()

        cov = (cov - min_value) / (max_value - min_value)

        tensor_upper = torch.triu(cov)
        cov = tensor_upper.T + tensor_upper - torch.diag(cov.diagonal())

        if torch.min(torch.diag(cov)) <= 0.0:
            cov = cov - torch.eye(dim).cuda() * torch.min(torch.diag(cov)) + torch.eye(dim).cuda() * 0.01

        if torch.min(torch.diag(cov)) == 0.0:
            cov = cov + torch.eye(dim).cuda() * 0.01



        mvn = torch.distributions.MultivariateNormal(mean.float(), cov.float())

        log_probs = mvn.log_prob(x)

        return log_probs



    def GMM_EM(self, datapoints, normalized_cos):



        if self.weights == []:

            self.weights = torch.tensor([1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24,1/24]).cuda()

        if self.means == []:
            for i in range(self.K_GMM):
                mean = torch.randn(self.dim).cuda()
                self.means.append(mean)

        if self.covars == []:
            for i in range(self.K_GMM):

                cov = torch.eye(self.dim).cuda() * torch.rand(1).cuda()
                self.covars.append(cov)






        train_data = datapoints

        train_tensor = train_data.data.reshape(-1, 768).float()  # 将图像数据展平为 (n, 784) 的张量

        pca = PCA(n_components=self.dim)
        data = torch.from_numpy(pca.fit_transform(train_tensor.cpu())).to(torch.float32).cuda()

        loglikelyhood = torch.tensor([0.0]).cuda()
        oldloglikelyhood = torch.tensor([2.1]).cuda()
        len_d,dim = data.shape

        loglike_gap = torch.tensor([0.0]).cuda()




        while torch.abs(loglikelyhood-oldloglikelyhood) > 2.0:
            gammas = []
            oldloglikelyhood = loglikelyhood

            # E-step
            for n in range(len_d):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(data[n], self.means[k], self.covars[k])
                                                    for k in range(self.K_GMM)]
                respons = torch.tensor(respons)

                # if n%10 == 0:
                #     print(str(n))
                has_nan = torch.isnan(respons).any().item()
                if has_nan:
                    if respons[~torch.isnan(respons)].nelement() != 0:
                        respons[torch.isnan(respons)] = torch.min(respons[~torch.isnan(respons)]) * 1.1
                    else:


                        respons = torch.tensor([-1.0,-1.0,-1.0,-1.0,-1.0])



                has_inf = torch.isinf(respons).any().item()
                if has_inf:
                    if respons[~torch.isinf(respons)].nelement() != 0:
                        respons[torch.isinf(respons)] = torch.max(respons[~torch.isinf(respons)]) * 2
                    else:
                        respons = torch.tensor([-1.0,-1.0,-1.0,-1.0,-1.0])


                sum_respons = torch.sum(respons)
                has_inf = torch.isinf(sum_respons).any().item()

                if has_inf:
                    if sum_respons[~torch.isinf(sum_respons)].nelement() != 0:
                        sum_respons[torch.isinf(sum_respons)] = torch.max(sum_respons[~torch.isinf(sum_respons)]) * 2.0
                    else:
                        print("sumrespons错了")
                        sum_respons = torch.tensor([-256.0,-256.0,-256.0,-256.0,-256.0])
                gammas.append(respons/sum_respons)

            for k in range(self.K_GMM):

                for n in range(len_d):
                    isLargest = True
                    for j in range(self.K_GMM):
                        if gammas[n][k] < gammas[n][j]:
                            isLargest = False
                    if isLargest:
                        G = self.Gaussian(data[n], self.means[k], self.covars[k])
                        H = H - G
                crossH.append(H)


            if self.K_GMM > 0:

                #计算最大值
                max_value = max(crossH)
                # 获取最大值的索引

                max_index = crossH.index(max_value)

                if  len([x for x in crossH if x != 0]) >=2:
                    filtered_crossH = [x for x in crossH if x != 0]

                    min_value = min(filtered_crossH)
                    # 获取最小值的索引
                    min_index = crossH.index(min_value)
                else:
                    min_value = min(crossH)
                    # 获取最小值的索引
                    min_index = crossH.index(min_value)








                if min_value < max_value :

                    for n in range(len_d):
                        isLargest = True
                        for k in range(self.K_GMM):
                            if gammas[n][min_index] < gammas[n][k]:
                                isLargest = False
                        if isLargest:

                            f = torch.tensor(0.0).cuda()
                            f = f + gammas[n][max_index]
                            gammas[n][max_index] = gammas[n][min_index]
                            gammas[n][min_index] = f



                # merge
                    for k in range(self.K_GMM):
                        nk = torch.sum(torch.tensor([gammas[n][k] for n in range(len_d)]))
                        self.weights[k] = nk / len_d
                        dk = 0.0
                        for n in range(len_d):
                            dk = dk + gammas[n][k] * data[n]
                        self.means[k] = (1.0 / nk) * dk
                        xdiffs = data - self.means[k]
                        ck = torch.zeros(dim, dim).cuda()
                        for n in range(len_d):
                            a = torch.matmul(xdiffs[n].reshape((dim, 1)), xdiffs[n].reshape((1, dim)))
                            b = gammas[n][k] * a

                            ck = ck + b
                        self.covars[k] = (1.0 / nk) * ck


                    diagonal = self.covars[max_index].diag()
                    _, max_index_dia = diagonal.max(dim=0)

                    MS = self.means[max_index][max_index_dia]
                    for n in range(len_d):
                        isLargest = True
                        for k in range(self.K_GMM):
                            if gammas[n][max_index] < gammas[n][k]:
                                isLargest = False
                        if isLargest:
                            if data[n][max_index_dia] < MS:
                                f = torch.tensor(0.0).cuda()
                                f = f + gammas[n][min_index]
                                gammas[n][min_index] = gammas[n][max_index]
                                gammas[n][max_index] = f


            # M-step
            for k in range(self.K_GMM):
                nk = torch.sum(torch.tensor([gammas[n][k] for n in range(len_d)]))

                self.weights[k] = nk / len_d

                dk = 0.0
                for n in range(len_d):
                    dk = dk + gammas[n][k] * data[n]
                self.means[k] = (1.0 / nk) * dk
                xdiffs = data - self.means[k]
                ck = torch.zeros(dim, dim).cuda()
                for n in range(len_d):
                    a = torch.matmul(xdiffs[n].reshape((dim, 1)), xdiffs[n].reshape((1, dim)))
                    #print("a"+str(a))
                    b = gammas[n][k] * a
                    ck = ck + b
                if nk != 0.0:
                    self.covars[k] = (1.0 / nk) * ck
                else:
                    self.covars[k] = ck
            loglikelyhood = []
            for n in range(len_d):
                tmp = torch.tensor([0.0]).cuda()
                for k in range(self.K_GMM):

                    tmp += self.weights[k]*self.Gaussian1(data[n],self.means[k],self.covars[k])

                loglikelyhood.append(tmp)
            loglikelyhood = torch.sum(torch.tensor(loglikelyhood))

            print(torch.abs(loglikelyhood-oldloglikelyhood))

            if loglike_gap != torch.tensor([0.0]).cuda() and torch.abs(loglikelyhood-oldloglikelyhood)>loglike_gap:
                break
            else:
                loglike_gap = torch.abs(loglikelyhood-oldloglikelyhood)


        posibility = torch.cat(gammas).reshape(-1,self.K_GMM).cuda() #(256,96)




        values1, indices1 = torch.topk(posibility, k=1, dim=1)


        # 求每一行的第二大值和索引
        values2, indices2 = torch.topk(posibility, k=2, dim=1)

        z3_mean = torch.zeros(1, 768).cuda()
        for i in range(datapoints.size()[0]):
            mean = torch.zeros(1, 768).cuda()
            k = 0
            for j in range(datapoints.size()[0]):
                if indices1[j] == indices2[:, 1:][i]:
                    mean += datapoints[j]
                    k += 1

            if torch.sum(mean) == 0.0:

                k = 0
                mean = torch.zeros(1, 768).cuda()
                for j in range(datapoints.size()[0]):
                    if indices2[:, 1:][j] == indices2[:, 1:][i]:
                        mean += datapoints[j]+ torch.randn(1).cuda()
                        k += 1

            z3_mean = torch.cat((z3_mean, mean / k), dim=0)

        z3 = z3_mean[1:]




        print(crossH)
        loss_fn = 0.0

        z3_weight = torch.zeros(1, 1).cuda()

        for i in range(data.size()[0]):
            v_i = torch.var(torch.masked_select(posibility[i], posibility[i] != torch.max(posibility[i])))
            con_i = torch.sigmoid(values1[i] - values2[:, 1:][i]) / torch.sigmoid(v_i)
            z3_weight = torch.cat((z3_weight,  values2[:, 1:][i].unsqueeze(-1)/con_i), dim=0)

            p_cos = 0.0
            k = 0

            delta = 0.0
            loss_cbml_i = 0.0
            list2_a = 0.0

            p_cos += normalized_cos[i][i] * (values1[i]/values2[:, 1:][i])

            for j in range(data.size()[0]):
                if indices1[j] == indices1[i]:
                    v_j = torch.var(torch.masked_select(posibility[j], posibility[j] != torch.max(posibility[j])))



                    fn_cos = self.sim(data[i], data[j])
                    delta = delta + fn_cos - p_cos
                    loss_cbml_i += F.relu(delta + self.alpha) + F.relu(-delta - self.beta)


                    list2_a += fn_cos


                    k += 1

            loss_cbml_i = loss_cbml_i / k
            loss_fn+=loss_cbml_i



        loss_fn = torch.squeeze(loss_fn)
        z3_weight = z3_weight[1:]
        z3_weight = z3_weight / torch.mean(z3_weight)



        self.weights = []
        self.means = []
        self.covars = []



        return z3 ,loss_fn, z3_weight








