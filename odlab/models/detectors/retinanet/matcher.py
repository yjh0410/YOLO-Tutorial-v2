# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.
import torch
from utils.box_ops import box_iou


class RetinaNetMatcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self,
                 num_classes, 
                 iou_threshold, 
                 iou_labels, 
                 allow_low_quality_matches=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        self.num_classes = num_classes
        # Add -inf and +inf to first and last position in iou_thresholdhreshold
        iou_threshold = iou_threshold[:]
        assert iou_threshold[0] > 0
        iou_threshold.insert(0, -float("inf"))
        iou_threshold.append(float("inf"))
        assert all(low <= high for (low, high) in zip(iou_threshold[:-1], iou_threshold[1:]))
        assert all(label in [-1, 0, 1] for label in iou_labels)
        assert len(iou_labels) == len(iou_threshold) - 1
        self.iou_threshold = iou_threshold
        self.iou_labels = iou_labels
        self.allow_low_quality_matches = allow_low_quality_matches

    @torch.no_grad()
    def __call__(self, anchors, targets):
        """
            anchors: (Tensor) [B, M, 4] (x1, y1, x2, y2)
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        # list[Tensor(R, 4)], one for each image
        gt_classes = []
        gt_boxes = []
        device = anchors.device

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # [N,]
            tgt_labels = targets_per_image['labels'].to(device)
            # [N, 4]
            tgt_boxes = targets_per_image['boxes'].to(device)
            # [N, M], N is the number of targets, M is the number of anchors
            match_quality_matrix, _ = box_iou(tgt_boxes, anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matching(match_quality_matrix)
            has_gt = len(tgt_labels) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = tgt_boxes[gt_matched_idxs]

                gt_classes_i = tgt_labels[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                matched_gt_boxes = torch.zeros_like(anchors_per_image)

            gt_classes.append(gt_classes_i)
            gt_boxes.append(matched_gt_boxes)

        return torch.stack(gt_classes), torch.stack(gt_boxes)

    def matching(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an N x M tensor, containing the
                pairwise quality between N ground-truth elements and M predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length M, where matches[i] is a matched
                ground-truth index in [0, N)
            match_labels (Tensor[int8]): a vector of length M, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.iou_labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is N (gt) x M (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

        for (l, low, high) in zip(self.iou_labels, self.iou_threshold[:-1], self.iou_threshold[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None],
            as_tuple=False
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        match_labels[pred_inds_to_update] = 1
