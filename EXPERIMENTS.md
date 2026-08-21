
# CIFAR10

```
CUDA_VISIBLE_DEVICES=5,6 python acat/runner.py with \
h.exp_name=cifar10/madry \
h.model=resnet18 \
h.batch_size=128 \
h.epochs=200 \
h.max_epochs=200 \
h.model_opt.scheduler_milestones='[0.5,0.75]' \
h.model_opt.lr=0.1 \
h.val_split=0.2 \
h.gpus=2 \
h.val_rate=1 \
h.attack_opt.type=defaultpgd \
h.attack_opt.num_steps=7 \
h.attack_opt.eps=0.031 \
h.attack_opt.lr=0.00797 \
h.test_attack_opt.eps=0.031 \
h.test_attack_opt.num_steps=20 \
h.test_attack_opt.type=defaultpgd \
h.danskinattack=False \
h.num_of_attacks=10 \
h.log_verbose=False

CUDA_VISIBLE_DEVICES=5,6 python acat/runner.py with \
h.exp_name=cifar10/danskin \
h.model=resnet18 \
h.batch_size=128 \
h.epochs=200 \
h.max_epochs=200 \
h.model_opt.scheduler_milestones='[0.5,0.75]' \
h.model_opt.lr=0.1 \
h.val_split=0.2 \
h.gpus=2 \
h.val_rate=1 \
h.attack_opt.type=defaultpgd \
h.attack_opt.num_steps=7 \
h.attack_opt.eps=0.031 \
h.attack_opt.lr=0.00797 \
h.test_attack_opt.eps=0.031 \
h.test_attack_opt.num_steps=20 \
h.test_attack_opt.type=defaultpgd \
h.danskinattack=True \
h.num_of_attacks=10 \
h.log_verbose=False
```


# Debugging

```
python acat/runner.py with \
h.exp_name=cifar10/madry/linear \
h.model=linear \
h.batch_size=128 \
h.epochs=200 \
h.max_epochs=200 \
h.model_opt.scheduler_milestones='[0.5,0.75]' \
h.model_opt.lr=0.1 \
h.val_split=0.2 \
h.gpus=0 \
h.val_rate=1 \
h.attack_opt.type=defaultpgd \
h.attack_opt.num_steps=7 \
h.attack_opt.eps=0.031 \
h.attack_opt.lr=0.00797 \
h.test_attack_opt.eps=0.031 \
h.test_attack_opt.num_steps=20 \
h.test_attack_opt.type=defaultpgd \
h.danskinattack=False \
h.max_ensemble=False \
h.num_of_attacks=10 \
h.log_verbose=False \
h.limit_train_batches=1 \
h.limit_val_batches=1 \
h.limit_test_batches=1
```
