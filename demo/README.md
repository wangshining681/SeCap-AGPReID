# FastReID Demo

We provide a command line tool to run a simple demo of builtin models.

You can run this command to get cosine similarites between different images

```bash
CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml \
--parallel --vis-label --dataset-name DukeMTMC --output logs/mgn_duke_vis \
--opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth


CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2A --output VDT-AGPRID/logs/GAReID/G2A_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2A --output VDT-AGPRID/logs/GAReID/G2A_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_A2GG --output VDT-AGPRID/logs/GAReID/A2GG_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_A2GG --output VDT-AGPRID/logs/GAReID/A2GG_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2AA --output VDT-AGPRID/logs/GAReID/G2AA_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2AA --output VDT-AGPRID/logs/GAReID/G2AA_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/GAReID/Vit_base_new/config.yaml --parallel --vis-label --dataset-name GAReID_G2AG --output VDT-AGPRID/logs/GAReID/G2AG_baseline --opts MODEL.WEIGHTS logs/GAReID/Vit_base_new/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/GAReID/VQD_128/config.yaml --parallel --vis-label --dataset-name GAReID_G2AG --output VDT-AGPRID/logs/GAReID/G2AG_ours --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_final.pth


CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/AGReID/sbs_vit_base/config.yaml --parallel --vis-label --dataset-name AG_ReID --output VDT-AGPRID/logs/AG_ReID/AG_baseline --opts MODEL.WEIGHTS logs/AGReID/sbs_vit_base/model_final.pth

CUDA_VISIBLE_DEVICES=5 python demo/visualize_result.py --config-file logs/AGReID/VQD_128_10view/config.yaml --parallel --vis-label --dataset-name AG_ReID --output VDT-AGPRID/logs/AG_ReID/AG_ours --opts MODEL.WEIGHTS logs/AGReID/VQD_128_10view/model_final.pth

CUDA_VISIBLE_DEVICES=4 python demo/visualize_result.py --config-file logs/AGReID/VQD_128_10view/config.yaml --parallel --vis-label --dataset-name AG_ReID_G2A --output VDT-AGPRID/logs/AGReID/GA_ours --opts MODEL.WEIGHTS logs/AGReID/VQD_128_10view/model_final.pth

CUDA_VISIBLE_DEVICES=6 python demo/visualize_result.py --config-file logs/AGReID/sbs_vit_base/config.yaml --parallel --vis-label --dataset-name AG_ReID_G2A --output VDT-AGPRID/logs/AGReID/GA_baseline --opts MODEL.WEIGHTS logs/AGReID/sbs_vit_base/model_final.pth


CUDA_VISIBLE_DEVICES=0 python demo/visualize_result.py --config-file logs/CARGO/secap_64/config.yaml --parallel --vis-label --dataset-name CARGO_AA --output VDT-AGPRID/logs/CARGO/secap_64 --opts MODEL.WEIGHTS logs/CARGO/secap_64/model_best.pth

CUDA_VISIBLE_DEVICES=0 python demo/visualize_tSNE.py --config-file logs/GAReID/VQD_128/config.yaml --vis-label --dataset-name GAReID --output VDT-AGPRID/logs/GAReID/tSNE --opts MODEL.WEIGHTS logs/GAReID/VQD_128/model_best.pth DATALOADER.NUM_INSTANCE 32 SOLVER.IMS_PER_BATCH 64

CUDA_VISIBLE_DEVICES=1 python demo/visualize_tSNE.py --config-file logs/GAReID/Vit_base/config.yaml --vis-label --dataset-name GAReID --output VDT-AGPRID/logs/GAReID/tSNE/vit --opts MODEL.WEIGHTS logs/GAReID/Vit_base/model_best.pth DATALOADER.NUM_INSTANCE 32 SOLVER.IMS_PER_BATCH 64

```
