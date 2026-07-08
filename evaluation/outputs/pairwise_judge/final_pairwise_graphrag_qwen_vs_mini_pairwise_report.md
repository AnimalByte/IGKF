# Blinded Pairwise GraphRAG Judge Analysis

Question: Does direct blinded comparison distinguish Qwen + GraphRAG from GPT-5.4-mini-no-reasoning + GraphRAG despite both having a median absolute score of 5?

This secondary analysis preserves the existing absolute 1-5 evaluation and uses only saved model responses.

Judge model: `gpt-5.5`
Prompt version: `pairwise_prompt_v1`
Rubric version: `pairwise_biomedical_rubric_v1`

Outcomes:
- QWEN_WIN: 23 (11.2% of all paired questions)
- MINI_WIN: 165 (80.5% of all paired questions)
- TIE: 8 (3.9% of all paired questions)
- POSITION_UNSTABLE: 9 (4.4% of all paired questions)
- INCOMPLETE_OR_PARSE_ERROR: 0 (0.0% of all paired questions)

Exact binomial test among decisive non-unstable outcomes:
- Qwen wins: 23
- GPT-5.4 mini wins: 165
- Ties: 8
- Position-unstable: 9
- Qwen decisive win rate: 0.12234042553191489
- 95% CI: [0.07915745527901162, 0.17789774199643252]
- Raw p-value: 1.1362885807314184e-27
- Estimated actual API cost: $4.6566

Interpretation should remain conservative: equal medians do not establish equivalence, and this pairwise analysis only assesses residual differences under this benchmark and judge.