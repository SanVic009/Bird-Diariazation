# It should have:

## Pre processing
- Normalization on frequency axis
- Gaussian noise injection
- Frequency and time masking
- MixUp augmentation

## Model
- use PANNs
- use mobilenet if possible
- try effecient net
- Sound Event Detection (SED) framework with multi-label classification head (potential)

## Training
- Multi-label loss (BCE)
- Class/instance balancing
- 5-fold CV
- Mixed precision
