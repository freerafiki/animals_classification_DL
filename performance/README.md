# Performance

Here we report the performances of the different models

| Model | Framework | Type | # Animals | Training Acc. | Test Acc. | Image Res | # of Pars | Epochs | Who |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ff_tf_32 | Tensorflow | MLP | 12 | 28.9% | 32.88% | 32x32 | 197956 | 1500 | lucap |
| ff_tf_32_5cl | Tensorflow | MLP | 4 | 60.48% | 58.92% | 32x32 | 197884 | 500 | lucap |
| pt_ff_basic | Pytorch | MLP | 15 | 14.18% | 12.46% | 32x32 | 404175 | 5 | lucap |
