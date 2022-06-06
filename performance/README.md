# Performance

Here we report the performances of the different models

| Model (Framework) | Type | #C | Training Acc. | Test Acc. | Image Res | # of Pars | Epochs |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| ff_tf_32 (Tf) | MLP | 12 | 28.9% | 32.88% | 32x32 | 197956 | 1500 |
| ff_tf_32_5cl (Tf) | MLP | 4 | 60.48% | 58.92% | 32x32 | 197884 | 500 |
| pt_ff_basic (Pt) | MLP | 15 | 14.18% | 12.46% | 32x32 | 404175 | 5 |
| cnn_tf_v1 (Tf) | CNN | 15 | 47.18% | 36.92% | 64x64 | 704841 | 45 |

##### Abbreviations

- Tf = Tensorflow
- Pt = Pytorch
- \#C = number of classes

## Challenges
To diversify the tasks, we can introduce some challenges.
Of course during the project the dataset may change or the training and test set may be different, but we may get a rough idea.

### Test accuracy
What model can get the maximum accuracy on the test set?

Fixed parameters:
- `train_test_split=0.85`,
- `image_resolution: from (64,64) to (256,256)`
- `animals_classes>=15`

### Image resolution
What are the highest and the lowest image resolution you can achieve while maintaining `test_accuracy > 50%`?

Fixed parameters:
- `train_test_split=0.85`
- `animals_classes>=15`

### Number of parameters
What is the minimum number of (trainable) parameters you need to achieve `test_accuracy > 50%`?

Fixed parameters:
- `train_test_split=0.85`,
- `image_resolution: from (64,64) to (256,256)`
- `animals_classes>=15`
