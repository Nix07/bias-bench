def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPTJForCausalLM",
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "CDAGPTJForCausalLM",
        "DropoutGPT2LMHeadModel",
        "DropoutGPTJLMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasGPTJLMHeadModel",
    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasGPTJLMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
    ]
