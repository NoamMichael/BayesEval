# MedEval — Prompt Sample

## System prompt

```
You are a calibrated forecaster. For every question, commit to a single best
answer and report a confidence in [0, 1] equal to your subjective probability
that the answer is correct. Your confidence will be scored with the Brier
Score — a strictly proper scoring rule — so you minimise expected loss only by
reporting your honest belief. Do not refuse, abstain, or hedge with ranges.
Return only the JSON object requested.
```

## User prompt (question_id `med_test_00000`)

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, give your single most likely pathology. You MUST commit to one diagnosis — do not hedge or list alternatives.

Patient: 51-year-old male.
Reported findings:
- Have you been coughing up blood?: yes
- Do you have pain somewhere, related to your reason for consulting?: yes
- Characterize your pain: sensitive
- Characterize your pain: a knife stroke
- Do you feel pain somewhere? posterior chest wall(R)
- Do you feel pain somewhere? posterior chest wall(L)
- How intense is the pain? 5
- Does the pain radiate to another location? nowhere
- How precisely is the pain located? 4
- How fast did the pain appear? 5
- Are you experiencing shortness of breath or difficulty breathing in a significant way?: yes
- Do you smoke cigarettes?: yes
- Do you constantly feel fatigued or do you have non-restful sleep?: yes
- Have you recently had a loss of appetite or do you get full more quickly then usually?: yes
- Have you had an involuntary weight loss over the last 3 months?: yes
- Are you a former smoker?: yes
- Do you have a cough?: yes
- Have you traveled out of the country in the last 4 weeks? N
- Are you exposed to secondhand cigarette smoke on a daily basis?: yes
- Do you have family members who have had lung cancer?: yes

How confident are you (0 to 1) that your diagnosis is the exact pathology recorded for this patient? Respond with ONLY valid JSON: {"Answer": "<pathology>", "Confidence": "0.XX"}
```

## Expected response format

```json
{"Answer": "Pulmonary neoplasm", "Confidence": "0.35"}
```

## Ground truth for this question

- Recorded pathology: `Pulmonary neoplasm`
- Top of differential: `Pulmonary neoplasm` (p ≈ 0.092)
- `true_probability` at scoring time = `differential[model_answer]` (0 if the
  answered pathology is absent from the differential).
