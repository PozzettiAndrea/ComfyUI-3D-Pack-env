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

class imageSuperNet:
    """Pass-through. Was a 4x Real-ESRGAN enhance pass over the multiview
    albedo/mr maps.

    Dropped because realesrgan pulls in basicsr 1.4.2, whose setup.py reads its
    version via `exec()` + `locals()` -- which PEP 667 broke in Python 3.13, so
    it cannot build at all. basicsr's last release was 2022 and the
    `basicsr-fixed` fork carries the identical bug.

    Little is lost: textureGenPipeline resized the 4x output straight back down
    to render_size before baking, so this only ever acted as a sharpening pass,
    never as added resolution. Upstream reached the same conclusion for the
    newer pipeline -- the equivalent calls in Hunyuan3D_V2's texgen/pipelines.py
    (lines 94 and 215) are commented out.

    Kept as a class rather than deleted so textureGenPipeline's construction
    (line 91) and call sites (lines 164-165) stay untouched.
    """

    def __init__(self, config) -> None:
        pass

    def __call__(self, image):
        return image
