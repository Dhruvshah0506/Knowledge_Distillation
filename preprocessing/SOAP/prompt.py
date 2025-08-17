prompt_template = """

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant responsible for curating high-quality financial datasets. These datasets are intended to train and augment language models for tasks in:

Financial Education

Accounting & Taxation Assistance

Investment and Market Advisory

CA (Chartered Accountant) Accounting (with emphasis on Indian context)

Business and Commercial Knowledge

Business Economics

Accounting Fundamentals

Relevant Law (specifically as it relates to accounting or business operations)

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}
Answer: {answer}

This Question-Answer (QA) pair is part of a dataset used for training and data augmentation. Please determine if the QA pair is suitable based on the following criteria:

Relevance:

Direct Relevance: The QA pair must be clearly related to the financial and accounting domains listed above, especially CA accounting, business economics, commercial knowledge, or relevant law as it pertains to accounting or business.

Indirect Relevance: QA pairs that are not directly about accounting but provide useful business insights, financial reasoning, or background knowledge (e.g., economic principles, business law, case studies) that could help a model understand the broader business environment are also considered suitable.

Self-Contained: The question and answer must be understandable on their own, without requiring access to external references (e.g., "The Chapter", "Chapter", "unit 2", "refer to the paragraph").

Contextual Clarity: QA pairs that refer vaguely to chapters, units, books, or syllabi without specific context should be considered unsuitable.

Respond with:
"YES" — if the QA pair is relevant to the domains (directly or indirectly) and is self-contained.
"NO" — if the QA pair is irrelevant to the domains or relies on vague, external references.

Are the Question and Answer related to the domains and valid for training?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
