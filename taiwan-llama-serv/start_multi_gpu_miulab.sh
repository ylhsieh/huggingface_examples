CUDA_VISIBLE_DEVICES=0,1,2,3 python api_server.py \
--model "../miulab_llama2/" \
--dtype "half" \
--seed 1331 \
--gpu-memory-utilization 0.85 \
--port 8090 \
--tensor-parallel-size 4
