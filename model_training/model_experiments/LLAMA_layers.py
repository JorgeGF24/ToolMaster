
LLAMA_LAYERS = {
    f"Layer {i}":[
        f"model.layers.{i}.self_attn.q_proj.weight",
        f"model.layers.{i}.self_attn.k_proj.weight",
        f"model.layers.{i}.self_attn.v_proj.weight",
        f"model.layers.{i}.self_attn.o_proj.weight",
        f"model.layers.{i}.mlp.gate_proj.weight",
        f"model.layers.{i}.mlp.up_proj.weight",
        f"model.layers.{i}.mlp.down_proj.weight",
        f"model.layers.{i}.input_layernorm.weight",
        f"model.layers.{i}.post_attention_layernorm.weight",
    ] for i in range(32)
}

LLAMA_LAYERS["Embedding Layer"] = ["model.embed_tokens.weight"]
LLAMA_LAYERS["LM Head"] = [
    "model.norm.weight",
    "lm_head.weight"
]
