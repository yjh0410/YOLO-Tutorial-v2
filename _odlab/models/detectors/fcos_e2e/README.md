# Empirical research on End-to-End FCOS
Inspired by the YOLOv10, I recently make the empirical research on FCOS to evaluate the **End-to-End detection** paradigm.

## Experiments

- COCO

Incredibly, the FPS of the three FCOS are almost the same!

| Model                | Sclae      | FPS<sup>FP32<br>RTX 4060 | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs |
|----------------------|------------|--------------------------|------------------------|-------------------|--------|------|
| FCOS_RT_R18_3x       |  512,736   |           56             |          35.8          |        53.3       | [ckpt](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/fcos_rt_r18_3x_coco.pth) | [log](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/FCOS-RT-R18-3x.txt) |
| FCOS_RT_R18_3x (O2O) |  512,736   |           56             |          30.9          |        48.8       | [ckpt](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/fcos_rt_r18_3x_top1_coco.pth) | [log](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/FCOS-RT-R18-3x-COCO-top1.txt) |
| FCOS_E2E_R18_3x      |  512,736   |           56             |          34.1          |        50.6       | [ckpt](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/fcos_e2e_r18_3x_coco.pth) | [log](https://github.com/yjh0410/E2E_FCOS/releases/download/fcos_weight/FCOS-E2E-R18-3x-COCO.txt) |

For **FCOS_RT_R18_3x**, we only use one-to-many assinger to train `FCOS-RT-R18-3x` and evaluate it with NMS.

For **FCOS_RT_R18_3x (O2O)**, we only use one-to-one assinger to train `FCOS-RT-R18-3x` and evaluate it without NMS.

For **FCOS_E2E_R18_3x**, we deploy two parallel detection head, one using one-to-many assinger (o2m head) and the other using one-to-one assinger (o2o head). To avoid conflicts between the gradients returned by o2o head and o2m head, we truncate the gradients returned from o2o head to the backbone and neck, and only allow the gradients returned from o2m head to update the backbone and neck. This operation is consistent with the practice of YOLOv10. For evaluation, we remove the o2m head and only use o2o head without NMS.
