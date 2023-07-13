import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cmpm_loss_weighted(image_embeddings, text_embeddings, labels, weight=None, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    i2t_loss = i2t_loss.sum(dim=1)
    t2i_loss = t2i_loss.sum(dim=1)

    if weight is None:
        weight = torch.ones_like(i2t_loss)

    cmpm_loss = (i2t_loss * weight).sum() / weight.sum() + (t2i_loss * weight).sum() / weight.sum()
    sim_cos = torch.matmul(image_norm, text_norm.t())

    pos_sim = torch.masked_select(sim_cos, labels_mask)
    neg_sim = torch.masked_select(sim_cos, labels_mask == 0)

    return cmpm_loss, pos_sim, neg_sim


def compute_cmpm_loss(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = i2t_loss.sum(dim=1).mean() + t2i_loss.sum(dim=1).mean()
    sim_cos = torch.matmul(image_norm, text_norm.t())

    pos_sim = torch.masked_select(sim_cos, labels_mask)
    neg_sim = torch.masked_select(sim_cos, labels_mask == 0)

    return cmpm_loss, pos_sim, neg_sim


def compute_cmpc_loss(W, image_embeddings, text_embeddings, labels):
    """
    Cross-Modal Projection Classfication loss(CMPC)
    :param W: Tensor with dtype torch.float32
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
    """
    criterion = nn.CrossEntropyLoss(reduction='mean')
    W_norm = W / W.norm(dim=0)

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
    text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

    image_logits = torch.matmul(image_proj_text, self.W_norm)
    text_logits = torch.matmul(text_proj_image, self.W_norm)

    cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    # classification accuracy for observation
    image_pred = torch.argmax(image_logits, dim=1)
    text_pred = torch.argmax(text_logits, dim=1)

    image_precision = torch.mean((image_pred == labels).float())
    text_precision = torch.mean((text_pred == labels).float())

    return cmpc_loss, image_precision, text_precision


class CMPMLoss(nn.Module):
    def __init__(self, feature_size, num_classes, CMPM=True, CMPC=False, epsilon=1e-8):
        super().__init__()
        self.CMPM = CMPM
        self.CMPC = CMPC
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.feature_size = feature_size

        self.W = nn.Parameter(torch.randn(feature_size, num_classes))
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def forward(self, image_embeddings, text_embeddings, labels):
        result = {
            m: 0.0
            for m in [
                'loss',
                'cmpm',
                'cmpc',
                'image_precision',
                'text_precision',
                'negative_similarity',
                'position_similarity',
            ]
        }

        if self.CMPM:
            result['cmpm'], result['positive_similarity'], result['negative_similarity'] = compute_cmpm_loss(
                image_embeddings, text_embeddings, labels
            )

        if self.CMPC:
            result['cmpc'], result['image_precision'], result['text_precision'] = compute_cmpc_loss(
                self.W, image_embeddings, text_embeddings, labels
            )

        result['loss'] = result['cmpm'] + result['cmpc']
        return result
