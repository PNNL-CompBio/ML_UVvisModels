OMP_NUM_THREADS=8 THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32,openmp=True python deep_tensor_refactored/predict.py --clen 40 --batch_size 50 --num_neu_1 100 --num_neu_2 200 --model_name model_neu1_neu2_with_noise $data_path/new.xyz $saved_model_path/results/model_epoch9998.pkl.gz $saved_model_path/Y_vals.npz