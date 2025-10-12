from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token/GRPO-BASE-TRACT1/global_step_820/actor")
# tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

model = AutoModelForCausalLM.from_pretrained("/blob/v-tianyuchen/Projects/jepo/ckpts/JEPO_token/GRPO-BASE-TRACT1/global_step_820/actor")