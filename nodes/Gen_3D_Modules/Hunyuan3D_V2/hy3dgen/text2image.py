# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import random

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


class HunyuanDiTPipeline:
    def __init__(
        self,
        model_path="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
        device='cuda'
    ):
        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            enable_pag=True,
            pag_applied_layers=["blocks.(16|17|18|19)"]
        ).to(device)
        self.pos_txt = ",\u767d\u8272\u80cc\u666f,3D\u98ce\u683c,\u6700\u4f73\u8d28\u91cf"
        self.neg_txt = "\u6587\u672c,\u7279\u5199,\u88c1\u526a,\u51fa\u6846,\u6700\u5dee\u8d28\u91cf,\u4f4e\u8d28\u91cf,JPEG\u4f2a\u5f71,PGLY,\u91cd\u590d,\u75c5\u6001," \
                       "\u6b8b\u7f3a,\u591a\u4f59\u7684\u624b\u6307,\u53d8\u5f02\u7684\u624b,\u753b\u5f97\u4e0d\u597d\u7684\u624b,\u753b\u5f97\u4e0d\u597d\u7684\u8138,\u53d8\u5f02,\u7578\u5f62,\u6a21\u7cca,\u8131\u6c34,\u7cdf\u7cd5\u7684\u89e3\u5256\u5b66," \
                       "\u7cdf\u7cd5\u7684\u6bd4\u4f8b,\u591a\u4f59\u7684\u80a2\u4f53,\u514b\u9686\u7684\u8138,\u6bc1\u5bb9,\u6076\u5fc3\u7684\u6bd4\u4f8b,\u7578\u5f62\u7684\u80a2\u4f53,\u7f3a\u5931\u7684\u624b\u81c2,\u7f3a\u5931\u7684\u817f," \
                       "\u989d\u5916\u7684\u624b\u81c2,\u989d\u5916\u7684\u817f,\u878d\u5408\u7684\u624b\u6307,\u624b\u6307\u592a\u591a,\u957f\u8116\u5b50"

    def compile(self):
        # accelarate hunyuan-dit transformer,first inference will cost long time
        torch.set_float32_matmul_precision('high')
        self.pipe.transformer = torch.compile(self.pipe.transformer, fullgraph=True)
        # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=True)
        generator = torch.Generator(device=self.pipe.device)  # infer once for hot-start
        out_img = self.pipe(
            prompt='\u7f8e\u5c11\u5973\u6218\u58eb',
            negative_prompt='\u6a21\u7cca',
            num_inference_steps=25,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]

    @torch.no_grad()
    def __call__(self, prompt, seed=0):
        seed_everything(seed)
        generator = torch.Generator(device=self.pipe.device)
        generator = generator.manual_seed(int(seed))
        out_img = self.pipe(
            prompt=prompt[:60] + self.pos_txt,
            negative_prompt=self.neg_txt,
            num_inference_steps=25,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=generator,
            return_dict=False
        )[0][0]
        return out_img
