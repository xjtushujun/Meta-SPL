# Meta-SPL (self-paced learning)


================================================================================================================================================================


This is the code for the paper:
[Meta self-paced learning](http://engine.scichina.com/publisher/scp/journal/SSI/50/6/10.1360/SSI-2020-0005?slug=fulltext)  
Jun Shu, Deyu Meng*, Zongben Xu. 

If you find this code useful in your research then please cite  
```bash
@article{shu2020meta,
  title={Meta self-paced learning},
  author={SHU, Jun and MENG, Deyu and XU, Zongben},
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
