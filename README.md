# Recurrent Attention Model

<p align="center">
 <img src="./images/raw.png">
 <img src="./images/patch_1_scale_1.png">
 <img src="./images/patch_2_scale_2.png">
 <img src="./images/patch_3_scale_4.png">
</p>

<p align="center">
 <img src="./images/MNIST-ng6-ps8-np1-sc2-sd0.15-se1.gif">
</p>

<p align="center">
 <img src="./images/MNIST-ts60-ng8-ps12-np3-sc2-sd0.25-se1.gif">
</p>

<p align="center">
 <img src="./images/MNIST-ts60-nc4-cs8-ng8-ps12-np3-sc2-sd0.3-se1.gif">
</p>


## Results

<center>
| Experiment | Paper | This Repo. |
|------------|-------|------------|
| 28x28 MNIST w/ 6 Glimpses, 8x8, 1 scale | 1.29% | 1.08% w/ std = 0.15 |
| 60x60 Translated MNIST w/ 8 Glimpses, 12x12, 3 scales | 1.84% | 1.87% w/ std = 0.25|
| 60x60 Translated MNIST w/ 8 Glimpses, 12x12, 3 scales | 5.23% | 2.13% w/ std = 0.25 |
</center>

## References

- https://github.com/kevinzakka/recurrent-visual-attention
- https://github.com/ipod825/recurrent-visual-attention/
- http://torch.ch/blog/2015/09/21/rmva.html
- https://robromijnders.github.io/RAM/
- https://github.com/hehefan/Recurrent-Attention-Model
- https://github.com/amasky/ram