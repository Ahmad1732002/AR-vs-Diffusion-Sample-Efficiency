from d3pm_absorbing import D3PMAbsorbing


def d3pm_shakespeare():
    model_args = dict(
        vocab_size=50257,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.0,
        T=1000,
        lambda_ce=0.05,
    )
    training_args = dict(
        batch_size=8,
        seq_len=512,
        learning_rate=6e-4,
        min_lr=1e-5,
        gradient_accumulation_steps=16,
        warmup_iters=2_500,
        max_iters=100000,
        eval_iters=10,
        weight_decay=0.1,
        training_seed=1,
    )
    return D3PMAbsorbing, model_args, training_args


def d3pm_text8_4gpu():
    model, model_args, training_args = d3pm_text8()
    training_args["gradient_accumulation_steps"] = 1
    training_args["eval_iters"] = 250
    return model, model_args, training_args


def d3pm_openwebtext_8gpu():
    model_args = dict(
        vocab_size=50257,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.0,
        T=1000,
        lambda_ce=0.05,
    )
    training_args = dict(
        dataset="openwebtext",
        batch_size=16,
        seq_len=1024,
        learning_rate=6e-4,
        min_lr=1e-5,
        gradient_accumulation_steps=8,
        warmup_iters=2_500,
        max_iters=500_000,
        eval_iters=400,
        weight_decay=0.1,
        training_seed=9,
    )
    return D3PMAbsorbing, model_args, training_args


def d3pm_openwebtext_32gpu():
    model, model_args, training_args = d3pm_openwebtext_8gpu()
    training_args["gradient_accumulation_steps"] = 2
    training_args["eval_iters"] = 50
    return model, model_args, training_args
