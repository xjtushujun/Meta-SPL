# Meta-SPL (self-paced learning)

Self-paced learning (SPL) is a learning regime inspired by the learning process of humans and animals that gradually incorporates easy to more complex samples into training. Many progresses have been made recently. Current SPL algorithms, however, still exist critical limitations, such as how to determine the age hyper-parameters (especially the age parameters). Some heuristic strategies have been designed through cross-validation, or manually setting these parameters empirically. Such strategies, however, are with low efficiency, short of theoretical evidences, and very hard to be generally applied in practice. To address the above issues, we propose a meta-learning regime for adaptively learning age parameters involved in SPL. Three kinds of typical SPL algorithms are integrated into this regime, and their accuracy and generalization capability are substantiated through regression and classiffication experiments, as compared with conventional SPL paradigms without this adaptive age parameter tuning strategy.

========================================================================================================================================================


This is the code for the paper:
[Meta self-paced learning](http://engine.scichina.com/publisher/scp/journal/SSI/50/6/10.1360/SSI-2020-0005?slug=fulltext)  
Jun Shu, Deyu Meng*, Zongben Xu. 

If you find this code useful in your research then please cite  
```bash
@article{shu2020meta,
  title={Meta self-paced learning},
  author={Shu, Jun and Meng, Deyu and Xu, Zongben},
  journal={SCIENTIA SINICA Informationis},
  year={2020},
  publisher={Science China Press}
}
``` 


## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 0.4.0 
- Torchvision 0.2.0


## Running Meta-SPL on benchmark datasets (CIFAR-10 and CIFAR-100).
```bash
python main.py --dataset cifar10 --corruption_type unif(flip2) --corruption_prob 0.4
```



Contact: Jun Shu (xjtushujun@gmail.com); Deyu Meng(dymeng@mail.xjtu.edu.cn).
