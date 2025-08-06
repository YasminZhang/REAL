conda create -n jepo python=3.12 -y
suorce activate jepo
pip install -e .
pip3 install -e .[vllm]
