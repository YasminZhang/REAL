conda create -n real python=3.12 -y
source activate real
pip install -e .
pip3 install -e .[vllm]
# pip install flash-attn
pip install flash-attn==2.8.1 --no-build-isolation
pip install "transformers<4.54.0"
pip install "ray[default]"  
pip install ray==2.38
conda install -c conda-forge rdma-core