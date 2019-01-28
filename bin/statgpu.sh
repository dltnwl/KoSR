#!/bin/bash

#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [ -n "$SLURM_JOB_NAME" ]; then
    export JOB_NAME=$SLURM_JOB_NAME
fi

if [ -n "$SLURM_JOB_ID" ]; then
    source $HOME/.mail.env
    notify () {
        echo "$@"
        curl -s --user "api:$API_KEY" \
             "https://api.mailgun.net/v3/$DOMAIN/messages" \
             -F from="$MAIL_FROM" -F to="$MAIL_TO" \
             -F subject="$@" -F text="Sent via mailgun."
    }
else
    notify () {
        echo "$@"
    }
fi

notify "##### BEGIN $JOB_NAME ($SLURM_JOB_ID) AT $(date) #####"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "##### JOB INFOS #####"
    scontrol show jobid -dd $SLURM_JOB_ID
    module purge
    source activate DLKoASR
fi

mkdir -p target
LMS=( nolm fsm )
if [[ $JOB_NAME = *"bayesrnn" ]]; then
    CONFIGS=( 1x1 8x1 32x1 1x100 8x100 32x100 )
else
    CONFIGS=( 1x1 8x1 32x1 )
fi
for LM in "${LMS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        python -Ou main.py \
               --sess-dir "target/${JOB_NAME}" \
               --config "configs/${JOB_NAME}_${LM}_${CONFIG}.yml" > \
               "target/${JOB_NAME}_${LM}_${CONFIG}.csv"
    done
done

notify "#### FINISH $JOB_NAME ($SLURM_JOB_ID) AT $(date) #####"
