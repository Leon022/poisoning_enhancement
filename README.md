# poisoning_enhancement
In this repository, code is for our paper [PoE: Poisoning Enhancement Through Label Smoothing in Federated Learning]

## Installation
Install Python>=3.7.0 and Pytorch>=1.8.0

## Usage
### Prepare the dataset:
#### MNIST and CIFAR-10 dataset:

- MNIST and CIFAR will be automatically download 

## Code structures
- `main.py, update,py, test.py, Fed_aggregation.py, enhancement.py`: our FL framework
- `main.py`: entry point of the command line tool
- `options.py`: a parser of the FL configs, also assigns possible attackers given the configs
- `poison_data.py`: implement FL backdoor attack
- `enhancement.py`: code for label smoothing




## Running Federated Learning tasks

### Command line tools
```
python main.py --dataset mnist --model lenet5 --num_users 100 --rho 0.2 --epoch 150 --iid False --attack_start 4 --attack_methods CBA --attacker_list 2 5 7 9 --aggregation_methods EVE --detection_size 50 --gpu 0
```
Check out `parser.py` for the use of the arguments, most of them are self-explanatory. 
- If you choose the backdoor method `--attack_methods` to be `DBA`, then you will also need to set the number of attackers in `--attacker_list` to 4 or a multiple of 4.
- If you choose the aggregation rule `--aggregation_methods` to be `RLR`, then you will also need to set the threshold in `--robustLR_threshold`.
- If `--save_results` is specified, the training results will be saved under `./results` directory. 
