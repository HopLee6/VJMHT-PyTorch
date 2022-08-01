srun -p gpgpu --gres=gpu:p100:1 --qos=gpgpumse python -u main.py -c $1 --eval

#srun -p deeplearn --gres=gpu:v100:1 --qos=gpgpudeeplearn --time=0-3:00:00 python -u main.py -c $1 --eval