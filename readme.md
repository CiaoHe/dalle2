# Origanize
## Class DALLE2()



# Tokenzie / Embed

`text(str)` -tokenize-> `text-tokens`

# Tensor Shape
1. In dalle2_pytorch.py for `DiffusionPrior()`, its `p_sample()`/`p_sample_loop()`, the input x'shape should be `(batch_size, image_embed_dim`)