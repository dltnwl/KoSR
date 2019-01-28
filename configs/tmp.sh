#!/bin/bash

MODELS=( CTC Att CTCAtt )
USE_BAYES_RNN=( True False )
DATAS=( nangdok opendict zeroth )
LMS=( nolm fsm )
CONFIGS=( 1x1 8x1 32x1 1x100 8x100 32x100 )
for MODEL in "${MODELS[@]}"; do
    for USE_BAYES in "${USE_BAYES_RNN[@]}"; do
        for DATA in "${DATAS[@]}"; do
            for LM in "${LMS[@]}"; do
                for CONFIG in "${CONFIGS[@]}"; do
                    if [ "$USE_BAYES" = "True" ]; then
                        FILE="${MODEL,,}_${DATA}_bayesrnn_${LM}_${CONFIG}.yml"
                    else
                        FILE="${MODEL,,}_${DATA}_${LM}_${CONFIG}.yml"
                    fi
                    echo "BATCH_SIZE: 128" >> $FILE
                    echo "N_EPOCHS: 100" >> $FILE
                    echo "DATA_SEED: 21920" >> $FILE
                    echo "DATA_TYPE: '$DATA'" >> $FILE
                    echo "DATA_DIR: '$HOME/_data/$DATA'" >> $FILE
                    echo "ENCODER_ARGS:" >> $FILE
                    echo "  USE_BAYES_RNN: $USE_BAYES" >> $FILE
                    echo "DECODER: '$MODEL'" >> $FILE
                    echo "DECODER_ARGS:" >> $FILE
                    if [ "$LM" = "fsm" ]; then
                        echo "  USE_JAMO_FSM: True" >> $FILE
                    else
                        echo "  USE_JAMO_FSM: False" >> $FILE
                    fi
                    CONFIG=(`echo $CONFIG | tr "x" " "`)
                    beam_width=${CONFIG[0]}
                    n_samples=${CONFIG[1]}
                    echo "  BEAM_WIDTH: $beam_width" >> $FILE
                    echo "  N_SAMPLES: $n_samples" >> $FILE
                    echo "OPTIM_ARGS:" >> $FILE
                    if [ "$MODEL" = "Att" ]; then
                        echo "  learning_rate: 0.0001" >> $FILE
                    else
                        echo "  learning_rate: 0.00025" >> $FILE
                    fi
                done
            done
        done
    done
done
