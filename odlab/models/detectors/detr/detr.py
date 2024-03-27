import math
import torch
import torch.nn as nn

from ...backbone          import build_backbone
from ...basic.mlp         import MLP
from ...basic.conv        import BasicConv, UpSampleWrapper
from ...basic.transformer import TransformerEncoder, PlainDETRTransformer, get_clones

from utils.misc import multiclass_nms


# DETR
class DETR(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes = 80,
                 conf_thresh = 0.1,
                 nms_thresh  = 0.5,
                 topk        = 300,
                 use_nms     = False,
                 ca_nms      = False,
                 ):
        super().__init__()
        # ---------------- Basic setting ----------------
        self.stride = cfg['out_stride']
        self.upsample_factor = cfg['max_stride'] // cfg['out_stride']
        self.num_classes = num_classes
        ## Transformer parameters
        self.num_queries_one2one = cfg['num_queries_one2one']
        self.num_queries_one2many = cfg['num_queries_one2many']
        self.num_queries = self.num_queries_one2one + self.num_queries_one2many
        ## Post-process parameters
        self.ca_nms = ca_nms
        self.use_nms = use_nms
        self.num_topk = topk
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

        # ---------------- Network setting ----------------
        ## Backbone Network
        self.backbone, feat_dims = build_backbone(cfg)

        ## Input projection
        self.input_proj = BasicConv(feat_dims[-1], cfg['hidden_dim'], kernel_size=1, act_type=None, norm_type='GN')

        ## Transformer Encoder
        self.transformer_encoder = TransformerEncoder(d_model    = cfg['hidden_dim'],
                                                      num_heads  = cfg['en_num_heads'],
                                                      num_layers = cfg['en_num_layers'],
                                                      ffn_dim    = cfg['en_ffn_dim'],
                                                      dropout    = cfg['en_dropout'],
                                                      act_type   = cfg['en_act'],
                                                      pre_norm   = cfg['en_pre_norm'],
                                                      )

        ## Upsample layer
        self.upsample = UpSampleWrapper(cfg['hidden_dim'], self.upsample_factor)
        
        ## Output projection
        self.output_proj = BasicConv(cfg['hidden_dim'], cfg['hidden_dim'], kernel_size=3, padding=1, act_type='silu', norm_type='BN')
        
        ## Transformer
        self.query_embed = nn.Embedding(self.num_queries, cfg['hidden_dim'])
        self.transformer = PlainDETRTransformer(d_model             = cfg['hidden_dim'],
                                                num_heads           = cfg['de_num_heads'],
                                                ffn_dim             = cfg['de_ffn_dim'],
                                                dropout             = cfg['de_dropout'],
                                                act_type            = cfg['de_act'],
                                                pre_norm            = cfg['de_pre_norm'],
                                                rpe_hidden_dim      = cfg['rpe_hidden_dim'],
                                                feature_stride      = cfg['out_stride'],
                                                num_layers          = cfg['de_num_layers'],
                                                use_checkpoint      = cfg['use_checkpoint'],
                                                num_queries_one2one = cfg['num_queries_one2one'],
                                                num_queries_one2many    = cfg['num_queries_one2many'],
                                                proposal_feature_levels = cfg['proposal_feature_levels'],
                                                proposal_in_stride      = cfg['out_stride'],
                                                proposal_tgt_strides    = cfg['proposal_tgt_strides'],
                                                return_intermediate = True,
                                                )
    
        ## Detect Head
        class_embed = nn.Linear(cfg['hidden_dim'], num_classes)
        bbox_embed = MLP(cfg['hidden_dim'], cfg['hidden_dim'], 4, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = get_clones(class_embed, cfg['de_num_layers'] + 1)
        self.bbox_embed  = get_clones(bbox_embed, cfg['de_num_layers'] + 1)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

    def get_posembed(self, d_model, mask, temperature=10000, normalize=False):
        not_mask = ~mask
        scale = 2 * torch.pi
        num_pos_feats = d_model // 2

        # -------------- Generate XY coords --------------
        ## [B, H, W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        ## Normalize coords
        if normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + 1e-6)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + 1e-6)
        else:
            y_embed = y_embed - 0.5
            x_embed = x_embed - 0.5
        # [H, W] -> [B, H, W, 2]
        pos = torch.stack([x_embed, y_embed], dim=-1)

        # -------------- Sine-PosEmbedding --------------
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        x_embed = pos[..., 0] * scale
        y_embed = pos[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_embed = torch.cat((pos_y, pos_x), dim=-1)
        
        # [B, H, W, C] -> [B, C, H, W]
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        
        return pos_embed

    def post_process(self, box_pred, cls_pred):
        # Top-k select
        cls_pred = cls_pred[0].flatten().sigmoid_()
        box_pred = box_pred[0]

        # Keep top k top scoring indices only.
        num_topk = min(self.num_topk, box_pred.size(0))

        # Topk candidates
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:self.num_topk]

        # Filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        topk_scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]
        topk_box_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')

        ## Top-k results
        topk_labels = topk_idxs % self.num_classes
        topk_bboxes = box_pred[topk_box_idxs]

        topk_scores = topk_scores.cpu().numpy()
        topk_labels = topk_labels.cpu().numpy()
        topk_bboxes = topk_bboxes.cpu().numpy()

        # nms
        if self.use_nms:
            topk_scores, topk_labels, topk_bboxes = multiclass_nms(
                topk_scores, topk_labels, topk_bboxes, self.nms_thresh, self.num_classes, self.nms_class_agnostic)

        return topk_bboxes, topk_scores, topk_labels

    def resize_mask(self, src, mask=None):
        bs, c, h, w = src.shape
        if mask is not None:
            # [B, H, W]
            mask = nn.functional.interpolate(mask[None].float(), size=[h, w]).bool()[0]
        else:
            mask = torch.zeros([bs, h, w], device=src.device, dtype=torch.bool)

        return mask
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_old, outputs_deltas):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_boxes_old": c, "pred_deltas": d, }
            for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_coord_old[:-1], outputs_deltas[:-1])
        ]

    def inference_single_image(self, x):
        # ----------- Image Encoder -----------
        pyramid_feats = self.backbone(x)
        src = self.input_proj(pyramid_feats[-1])
        src = self.transformer_encoder(src)
        src = self.upsample(src)
        src = self.output_proj(src)

        # ----------- Prepare inputs for Transformer -----------
        mask = self.resize_mask(src)
        pos_embed = self.get_posembed(src.shape[1], mask, normalize=False)
        query_embeds = self.query_embed.weight[:self.num_queries_one2one]
        self_attn_mask = None

        # -----------Transformer -----------
        (
            hs,
            init_reference,
            inter_references,
            _,
            _,
            _,
            _,
            max_shape
        ) = self.transformer(src, mask, pos_embed, query_embeds, self_attn_mask)

        # ----------- Process outputs -----------
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_deltas_one2one = []

        for lid in range(hs.shape[0]):
            if lid == 0:
                reference = init_reference
            else:
                reference = inter_references[lid - 1]
            outputs_class = self.class_embed[lid](hs[lid])
            tmp = self.bbox_embed[lid](hs[lid])
            outputs_coord = self.transformer.decoder.delta2bbox(reference, tmp, max_shape)  # xyxy

            outputs_classes_one2one.append(outputs_class[:, :self.num_queries_one2one])
            outputs_coords_one2one.append(outputs_coord[:, :self.num_queries_one2one])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        # ------------ Post process ------------
        cls_pred = outputs_classes_one2one[-1]
        box_pred = outputs_coords_one2one[-1]
        
        # post-process
        bboxes, scores, labels = self.post_process(box_pred, cls_pred)
        # normalize bbox
        bboxes[..., 0::2] /= x.shape[-1]
        bboxes[..., 1::2] /= x.shape[-2]
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels
        
    def forward(self, x, src_mask=None, targets=None):
        if not self.training:
            return self.inference_single_image(x)

        # ----------- Image Encoder -----------
        pyramid_feats = self.backbone(x)
        src = self.input_proj(pyramid_feats[-1])
        src = self.transformer_encoder(src)
        src = self.upsample(src)
        src = self.output_proj(src)

        # ----------- Prepare inputs for Transformer -----------
        mask = self.resize_mask(src, src_mask)
        pos_embed = self.get_posembed(src.shape[1], mask, normalize=False)
        query_embeds = self.query_embed.weight
        self_attn_mask = torch.zeros(
            [self.num_queries, self.num_queries, ]).bool().to(src.device)
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True

        # -----------Transformer -----------
        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape
        ) = self.transformer(src, mask, pos_embed, query_embeds, self_attn_mask)

        # ----------- Process outputs -----------
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lid in range(hs.shape[0]):
            if lid == 0:
                reference = init_reference
            else:
                reference = inter_references[lid - 1]
            outputs_class = self.class_embed[lid](hs[lid])
            tmp = self.bbox_embed[lid](hs[lid])
            outputs_coord = self.transformer.decoder.box_xyxy_to_cxcywh(
                self.transformer.decoder.delta2bbox(reference, tmp, max_shape))

            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])

            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

            outputs_coords_old_one2one.append(reference[:, :self.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, self.num_queries_one2one:])
            outputs_deltas_one2one.append(tmp[:, :self.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],

            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        out["aux_outputs"] = self._set_aux_loss(
            outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
        )
        out["aux_outputs_one2many"] = self._set_aux_loss(
            outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
        )

        out["enc_outputs"] = {
            "pred_logits": enc_outputs_class,
            "pred_boxes": enc_outputs_coord_unact,
            "pred_boxes_old": output_proposals,
            "pred_deltas": enc_outputs_delta,
        }

        return out
