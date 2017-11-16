def flow_loss_block(
    gt_flow2, 
    gt_flow5, 
    gt_flow2_sig, 
    pr_flow2, 
    pr_flow5, 
    pr_conf2, 
    pr_conf5, 
    flow_weight, 
    conf_weight, 
    flow_sig_weight, 
    conf_sig_weight, 
    conf_diff_scale=1,
    level5_factor=0.5,
    loss_prefix='',
    ):
    """Adds loss operations to the flow outputs

    gt_flow2: ground truth flow at resolution level 2
    gt_flow5: ground truth flow at resolution level 5
    gt_flow2_sig: the scale invariant gradient of the ground truth flow at resolution level 2
    pr_flow2: predicted flow at resolution level 2
    pr_flow5: predicted flow at resolution level 5
    pr_conf2: predicted confidence at resolution level 2
    pr_conf5: predicted confidence at resolution level 5
    flow_weight: the weight for the 'absolute' loss on the flows
    conf_weight: the weight for the 'absolute' loss on the flow confidence
    flow_sig_weight: the weight for the loss on the scale invariant gradient images of the flow
    conf_sig_weight: the weight for the loss on the scale invariant gradient images of the confidence
    conf_diff_scale: scale factor for the absolute differences in the conf map computation
    level5_factor: factor for the losses at the smaller resolution level 5. affects losses on pr_flow5 and pr_conf5.
    loss_prefix: prefix name for the loss in the returned dict e.g. 'netFlow1_'

    Returns a dictionary with the losses
    """
    losses = {}
    epsilon = 0.00001

    loss_flow5 = (level5_factor*flow_weight) * pointwise_l2_loss(pr_flow5, gt_flow5, epsilon=epsilon)
    losses['loss_flow5'] = loss_flow5
    loss_flow2 = (flow_weight) * pointwise_l2_loss(pr_flow2, gt_flow2, epsilon=epsilon)
    losses['loss_flow2'] = loss_flow2

    loss_flow5_unscaled = pointwise_l2_loss(pr_flow5, gt_flow5, epsilon=0)
    losses['loss_flow5_unscaled'] = loss_flow5_unscaled
    loss_flow2_unscaled = pointwise_l2_loss(pr_flow2, gt_flow2, epsilon=0)
    losses['loss_flow2_unscaled'] = loss_flow2_unscaled

    # ground truth confidence maps
    conf2 = compute_confidence_map(pr_flow2, gt_flow2, conf_diff_scale)
    conf5 = compute_confidence_map(pr_flow5, gt_flow5, conf_diff_scale)

    if not pr_conf5 is None: 
        loss_conf5 = (level5_factor*conf_weight) * pointwise_l2_loss(pr_conf5, conf5, epsilon=epsilon)
        losses['loss_conf5'] = loss_conf5  
        loss_conf5_unscaled = pointwise_l2_loss(pr_conf5, conf5, epsilon=0)
        losses['loss_conf5_unscaled'] = loss_conf5_unscaled  
    if not pr_conf2 is None:
        loss_conf2 = conf_weight * pointwise_l2_loss(pr_conf2, conf2, epsilon=epsilon)
        losses['loss_conf2'] = loss_conf2  
        loss_conf2_unscaled = pointwise_l2_loss(pr_conf2, conf2, epsilon=0)
        losses['loss_conf2_unscaled'] = loss_conf2_unscaled  


    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}

    if not flow_sig_weight is None:
        pr_flow2_sig = scale_invariant_gradient(pr_flow2, **sig_params)
        loss_flow2_sig = flow_sig_weight * pointwise_l2_loss(pr_flow2_sig, gt_flow2_sig, epsilon=epsilon)
        losses['loss_flow2_sig'] = loss_flow2_sig  
        loss_flow2_sig_unscaled = pointwise_l2_loss(pr_flow2_sig, gt_flow2_sig, epsilon=0)
        losses['loss_flow2_sig_unscaled'] = loss_flow2_sig_unscaled  

    if not conf_sig_weight is None and not pr_conf2 is None:
        pr_conf2_sig = scale_invariant_gradient(pr_conf2, **sig_params)
        conf2_sig = scale_invariant_gradient(conf2, **sig_params)
        loss_conf2_sig = conf_sig_weight * pointwise_l2_loss(pr_conf2_sig, conf2_sig, epsilon=epsilon)
        losses['loss_conf2_sig'] = loss_conf2_sig  
        loss_conf2_sig_unscaled = pointwise_l2_loss(pr_conf2_sig, conf2_sig, epsilon=0)
        losses['loss_conf2_sig_unscaled'] = loss_conf2_sig_unscaled  

    # add prefix and return
    return { loss_prefix+k: losses[k] for k in losses }
