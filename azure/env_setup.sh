conda create -n jepo python=3.12 -y
source activate jepo
pip install -e .
pip3 install -e .[vllm]
# pip install flash-attn
pip install flash-attn==2.8.1 --no-build-isolation
pip install "transformers<4.54.0"
pip install "ray[default]"  
pip install ray==2.38