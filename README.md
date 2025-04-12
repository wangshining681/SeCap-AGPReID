# SeCap_AGPReID
SeCap: Self-Calibrating and Adaptive Prompts for Cross-view Person Re-Identification in Aerial-Ground Networks (CVPR 2025 Highlight)

### Updates

We will update more detailed result (including dataset, training, verification) in the future

- [x] **2025.2.28**: Build project page
- [x] **2025.3.2**: Add code
- [x] :Add G2APS-ReID reconstructed code
- [x] :Add detailed process description
- [ ] :Add the LAGPeR and the usage license(LAGPR is undergoing systematic collation and revision)

### News
20250405 - Our paper was selected as a CVPR 2025 Highlight！

20250310 - Our paper is available on [arxiv](https://arxiv.org/abs/2503.06965)

20250227 - Our paper has been accepted by CVPR'25!


## Dataset：LAGPeR and G2APS-ReID

We propose two large-scale aerial-ground cross-view person re-identification datasets to extend the AGPReID task framework. As depicted in Figure 1, the AGPReID task configuration involves matching pedestrian identities across heterogeneous aerial and ground surveillance perspectives. The proposed datasets are formally characterized as follows:

**Figure 1: Illustrative example of AGPReID task**<img src=".\assets\LAGPeR.png"  />

### LAGPeR

We constructed the LAGPeR dataset, a large-scale AGPReID benchmark, by collecting multi-scenario surveillance data across seven distinct real-world environments. (We sincerely apologize that, due to copyright and privacy protection regulations, the LAGPeR dataset is currently undergoing systematic collation and revision. The usage license will be opened subsequently upon completion of compliance reviews.)

<img src=".\assets\scene.png"  />

#### Hightlight

- **Real World**: The LAGPeR dataset is constructed from seven different real-world environments through autonomous data collection and manual annotations.
- **Multi View**: Features a multi-camera framework comprising 21 surveillance nodes distributed across 7 scenes, with synchronized capture from three observation planes: aerial view, ground oblique view, and frontal ground view.
- **Large Scale**: Contains $63,841$ images of $4,231$ identities, establishing LAGPeR as one of the largest real-world aerial-ground ReID benchmark to date.

### G2APS-ReID

We reconstructed the AGPReID dataset G2APS-ReID from a large-scale pedestrian search dataset [G2APS](https://github.com/yqc123456/HKD_for_person_search). Its scene instance is as follows. Under copyright constraints, the G2APS-ReID dataset cannot be publicly released. However, we have made available the complete codebase for reconstructing this dataset from G2APS, which can be accessed at [here](https://github.com/wangshining681/G2APS-ReID).

<img src=".\assets\scene_overview.jpg"/>

### Settings

- For the LAGPeR dataset, we selected 12 cameras (including 8 ground cameras and 4 drone cameras) from the first four scenes as the training set, while images from 9 cameras in the remaining three scenes were used for evaluation. 
- For the G2APS-ReID datasets, we randomly selected 60% of the IDs as the training set and the remaining as the test set. And then, we manually adjusted the IDs in the test set by reallocating IDs with too few or too many images to the training set.
- We calculated the gradient histogram features of images with the same ID and view and used K-nearest neighbor clustering to divide the images into K groups, randomly selecting one image from each group as a query image, thus selecting K representative images as queries (as shown in Tab 1). 
- We added $G \rightarrow A+G$ setting, which includes images from both ground and aerial perspectives in the gallery.

<table>
<caption>Table 1: Experimental setup and data division of the LAGPeR and G2APS-ReID datasets.</caption>
<thead>
<tr>
<th rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center;">Setting</th>
<th rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center;">Subset</th>
<th rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center;">#View.</th>
<th colspan="3" style="border: 1px solid #000; padding: 8px; text-align: center;">LAGPeR</th>
<th colspan="3" style="border: 1px solid #000; padding: 8px; text-align: center;">G2APS-ReID</th>
</tr>
<tr>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#Cam</th>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#IDs</th>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#Images</th>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#Cam</th>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#IDs</th>
<th style="border: 1px solid #000; padding: 8px; text-align: center;">#Images</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Train</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Aerial+Ground</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">12</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">2,708</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">40,770</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">2</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,569</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">100,871</td>
</tr>
<tr>
<td rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center; vertical-align: middle;">A → G</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Query</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Aerial</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">3</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">3,046</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,219</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">4,876</td>
</tr>
<tr>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Gallery</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Ground</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">6</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">15,533</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,219</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">37,202</td>
</tr>
 <tr>
<td rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center; vertical-align: middle;">G → A</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Query</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Ground</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">6</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">3,046</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,219</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">4,876</td>
</tr>
<tr>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Gallery</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Aerial</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">3</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">7,717</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,219</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">62,791</td>
</tr>
 <tr>
<td rowspan="2" style="border: 1px solid #000; padding: 8px; text-align: center; vertical-align: middle;">A → G</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Query</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Ground</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">6</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">3,046</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
</tr>
<tr>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Query</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">Aerial+Ground</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">9</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">1,523</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">20,204</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
<td style="border: 1px solid #000; padding: 8px; text-align: center;">-</td>
</tr>
</tbody>
</table>



## Method: SeCap

<img src=".\assets\Secap.png"/>

### Requirements

#### Step1: Prepare environments

Please refer to [INSTALL.md](./INSTALL.md).

#### Step2: Prepare datasets

Download the LAGPeR and G2APS-ReID datasets and modify the dataset path.  

#### Step3: Prepare ViT Pre-trained Models

Download the ViT-base Pre-trained model and modify the path. Line 13 in [configs](./configs/LAGPeR/secap.yml):

> PRETRAIN_PATH: xxx

### Training & Testing

Training SeCap on the LAGPeR dataset with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/LAGPeR/secap.yml MODEL.DEVICE "cuda:0" SOLVER.IMS_PER_BATCH 64  
```

Training SeCap on the LAGPeR dataset with 4 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/LAGPeR/secap.yml --num-gpus 4 SOLVER.IMS_PER_BATCH 256
```

Testing SeCap on the LAGPeR dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --config-file ./configs/LAGPeR/secap.yml --eval-only MODEL.WEIGHTS xxx 
```

