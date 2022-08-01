partition=$1
cfg=$2

if [ "$partition" = "gpgpu" ]; then
    qos="gpgpumse"
    gres="gpu:p100:1"
else
    qos="gpgpudeeplearn"
    gres="gpu:v100:1"
fi

srun -p gpgpu --gres=gpu:p100:1 --qos=gpgpumse --time=0-3:00:00 python -u main.py -c $cfg