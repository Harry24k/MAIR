
<p align="center">
  <img src="images/logo_wide.png?raw=true" width="467" title="ART logo">
</p>

<h2 align="center">Improve and Evaluate Robustness</h2>

<p>
  <a href="https://github.com/Harry24k/MAIR/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/Harry24k/MAIR?&color=brightgreen" /></a>
  <a href="https://github.com/Harry24k/MAIR/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/Harry24k/MAIR.svg?&color=blue" /></a>
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FHarry24k%2FMAIR&count_bg=%23FFC107&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


> "Make your AI Robust."

_MAIR_  is a PyTorch-based adversarial training framework. The goal of MAIR is to (1) _provide an easy implementation of adversarial training methods_ and (2) _make it easier to evaluate the adversarial robustness of deep learning models_.

Adversarial training has become the de-facto standard method for improving the robustness of models against adversarial examples. However, during the writing of [our paper](https://openreview.net/forum?id=AGVBqJuL0T), we realized that there is no framework integrating adversarial training methods. Therefore, to promote reproducibility and transparency in the field of deep learning, we integrated the algorithms, tools, and pre-trained models. 

_Citation:_

```
@inproceedings{
    kim2023fantastic,
    title={Fantastic Robustness Measures: The Secrets of Robust Generalization},
    author={Hoki Kim and Jinseong Park and Yujin Choi and Jaewook Lee},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=AGVBqJuL0T}
}
```

_Benchmarks on several adversarially trained models are available at [our notion](https://harry24k.notion.site/harry24k/958ba2d81d194c1fa86accf65c1f6b9e?v=e02792dc2e7e47c697ff6b4a2dfe1a54)._ 



## Installation and usage

### Installation

`pip install git+https://github.com/Harry24k/MAIR.git`


### Usage

```python
import mair
```

#### How to train a model?
**Step1.** Load model as follows:

```python
model = ...
rmodel = mair.RobModel(model, n_classes=10).cuda()
```

**Step2.** Set trainer as follows:

```python
from mair.defenses import AT
# Set adversarial training method: [Strandard, AT, TRADES, MART].
trainer = AT(rmodel, eps=EPS, alpha=ALPHA, steps=STEPS)
# Set recording information.
trainer.record_rob(train_loader, val_loader, eps=EPS, alpha=2/255, steps=10, std=0.1)
# Set detail training methods.
trainer.setup(optimizer="SGD(lr=0.1, momentum=0.9)",
              scheduler="Step(milestones=[100, 150], gamma=0.1)",
              scheduler_type="Epoch",
              minimizer=None, # or "AWP(rho=5e-3)",
              n_epochs=200
             )
```

**Step3.** Fit model as follows:

```python
trainer.fit(train_loader=train_loader,
            n_epochs=200,
            save_path='./models/', 
            save_best={"Clean(Val)":"HBO", "PGD(Val)":"HB"},
            # 'save_best': model with high PGD are chosen, 
            # while in similar cases, model with high Clean are selected.
            save_type="Epoch", 
            save_overwrite=False, 
            record_type="Epoch"
           )
```

#### How to evaluate a model?
**Step1.** Transform model as follows:

```python
model = ...
rmodel = mair.RobModel(model, n_classes=10).cuda()
```

**Step2.** Evaluate model as follows:

```python
rmodel.eval_accuracy(test_loader)  # clean accuracy
rmodel.eval_rob_accuracy_gn(test_loader)  # gaussian noise accuracy
rmodel.eval_rob_accuracy_fgsm(test_loader, eps)  # FGSM accuracy
rmodel.eval_rob_accuracy_pgd(test_loader, eps, alpha, steps)  # PGD accuracy
```



#### Please refer to [demo](https://github.com/Harry24k/MAIR/blob/main/demos/Train%20and%20Evaluation.ipynb) for details.



## Pre-trained models

Through [our notion](https://harry24k.notion.site/harry24k/958ba2d81d194c1fa86accf65c1f6b9e?v=e02792dc2e7e47c697ff6b4a2dfe1a54), you can directly download and use our pretrained models.
```python
from mair.hub import load_pretrained
rmodel = load_pretrained("CIFAR10_ResNet18_AT(eps=8, alpha=2, steps=10)", flag='Best', save_dir="./")
```

Please refer to [demo](https://github.com/Harry24k/MAIR/blob/main/demos/Load%20Pretrained%20Model.ipynb) for details.



Or you can use [Google-drive](https://drive.google.com/drive/folders/1JoMWfAqXuvROyBbPX1KxPTro7i8LCNLo).

In each folder, we upload four different files:

* `log.txt`: training log during training.
* `last.pth`: model at the end of epoch.
* `init.pth`: model at the start of epoch.
* `best.pth`: best model selected by the argment `save_best` in `trainer.fit`.

To load model, 

```python
rmodel.load_dict('./models/.../best.pth')
```

We are excited to share modes with the community, but we've run into a storage limitation on Google Drive. Any help would be greatly appreciated!



## Contribution

We welcome all contribution to MAIR in many forms :smiley:.
Especially, we are looking for diverse adversarial training methods beyond AT, TRADES, and MART.




## Future work
- [ ] Merge measures.
- [ ] Generalize attacks gathered from [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch).
- [ ] ...

