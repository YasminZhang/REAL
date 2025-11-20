# conda create -n jepo python=3.12 -y
# source activate jepo
pip install -e .
pip3 install -e .[vllm]
pip install flash-attn
pip install "transformers<4.54.0"
pip install "ray[default]" debugpy