# MNIST
simple GAN to recognize and reproduce hand written digits
this was trained on 2 H100 gpus, i strongly suggest you check that your pytorch version matches your CUDA drivers version, this would typically not be an issue (as long as the difference is not too big) but if you want to use DDP which significantly accelerates training you will need similar versions.
