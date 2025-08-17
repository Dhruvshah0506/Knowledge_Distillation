model_id = CONFIG['models']['fitb_model_id']
self.max_new_tokens = CONFIG['models'].get('fitb_max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
self.temperature = CONFIG['models'].get('fitb_temperature', 0.7)

self.tokenizer = T5Tokenizer.from_pretrained(model_id)
self.model = T5ForConditionalGeneration.from_pretrained(model_id)
self.rephrase_pipe = pipeline(
    "text2text-generation",
    model=self.model,
    tokenizer=self.tokenizer
)
