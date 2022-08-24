python /storage/colinconwell/workspace/deepgaze/run_magnif_mod.py -data /storage/colinconwell/workspace/deepgaze/Dataset -dataset-name stl10 \
    --workers 16 --log_root /storage/colinconwell/workspace/deepgaze/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
    --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
    --run_label test_temp_5  --magnif \
    --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
    --gridfunc_form radial_quad  --sample_temperature 5  --sampling_bdr 16 \
    --K 20  --temperature 0.07  --gpu_number 1

python /storage/colinconwell/workspace/deepgaze/run_magnif_mod.py -data /storage/colinconwell/workspace/deepgaze/Dataset -dataset-name stl10 \
    --workers 16 --log_root /storage/colinconwell/workspace/deepgaze/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
    --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
    --run_label test_temp_10  --magnif \
    --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
    --gridfunc_form radial_quad  --sample_temperature 10  --sampling_bdr 16 \
    --K 20  --temperature 0.07  --gpu_number 2
	
python /storage/colinconwell/workspace/deepgaze/run_magnif_mod.py -data /storage/colinconwell/workspace/deepgaze/Dataset -dataset-name stl10 \
    --workers 16 --log_root /storage/colinconwell/workspace/deepgaze/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
    --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
    --run_label test_temp_25  --magnif \
    --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
    --gridfunc_form radial_quad  --sample_temperature 25  --sampling_bdr 16 \
    --K 20  --temperature 0.07  --gpu_number 3
	
python /storage/colinconwell/workspace/deepgaze/run_magnif_mod.py -data /storage/colinconwell/workspace/deepgaze/Dataset -dataset-name stl10 \
    --workers 16 --log_root /storage/colinconwell/workspace/deepgaze/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
    --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
    --run_label test_temp_50  --magnif \
    --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
    --gridfunc_form radial_quad  --sample_temperature 50  --sampling_bdr 16 \
    --K 20  --temperature 0.07  --gpu_number 4
	
python /storage/colinconwell/workspace/deepgaze/run_magnif_mod.py -data /storage/colinconwell/workspace/deepgaze/Dataset -dataset-name stl10 \
    --workers 16 --log_root /storage/colinconwell/workspace/deepgaze/Foveated_Saccade_SimCLR-dev/runs  --randomize_seed \
    --ckpt_every_n_epocs 5 --epochs 100  --batch-size 256  --out_dim 256  \
    --run_label test_temp_100  --magnif \
    --disable_blur  --cover_ratio 0.05 0.35  --fov_size 20 \
    --gridfunc_form radial_quad  --sample_temperature 100  --sampling_bdr 16 \
    --K 20  --temperature 0.07  --gpu_number 5