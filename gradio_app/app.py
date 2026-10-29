import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"

# (api field, label, default, integer?)
FIELDS = [
    ("call_failure", "Call failures", 8, True),
    ("complains", "Complains (0 = no, 1 = yes)", 1, True),
    ("subscription_length", "Subscription length (months)", 20, True),
    ("charge_amount", "Charge amount", 3, True),
    ("seconds_of_use", "Seconds of use", 2500, True),
    ("frequency_of_use", "Frequency of use", 55, True),
    ("frequency_of_sms", "Frequency of SMS", 12, True),
    ("distinct_called_numbers", "Distinct called numbers", 25, True),
    ("age_group", "Age group", 3, True),
    ("tariff_plan", "Tariff plan", 1, True),
    ("status", "Status", 1, True),
    ("age", "Age", 42, True),
    ("customer_value", "Customer value", 1200.5, False),
]


def predict_churn(*values):
    payload = {field[0]: value for field, value in zip(FIELDS, values, strict=True)}

    try:
        response = requests.post(f"{API_URL}/predictions", json=payload, timeout=10)
    except requests.RequestException:
        return f"Cannot reach the API at {API_URL}. Is it running?", None, ""

    if response.status_code == 422:
        errors = response.json().get("detail", [])
        bad_fields = ", ".join(str(e["loc"][-1]) for e in errors)
        return f"Invalid input: {bad_fields}", None, ""

    if response.status_code != 200:
        return f"API error ({response.status_code}).", None, ""

    body = response.json()
    probability = body["churn_probability"]
    verdict = "Likely to churn" if body["prediction"] == 1 else "Likely to stay"
    details = (
        f"Model: {body['model_version']}\nRequest ID: {body.get('request_id', 'n/a')}"
    )
    return (
        verdict,
        {"Churn": probability, "Stay": round(1 - probability, 4)},
        details,
    )


with gr.Blocks() as app:
    gr.Markdown("# Churn Prediction Demo")

    with gr.Tab("Predict Churn"):
        gr.Markdown("Enter customer details to get a churn prediction from the API.")

        inputs = []
        for start in range(0, len(FIELDS), 4):
            with gr.Row():
                for _, label, default, is_int in FIELDS[start : start + 4]:
                    inputs.append(
                        gr.Number(
                            label=label,
                            value=default,
                            precision=0 if is_int else None,
                        )
                    )

        predict_button = gr.Button("Predict", variant="primary")

        with gr.Row():
            verdict_output = gr.Textbox(label="Result", interactive=False)
            probability_output = gr.Label(label="Churn probability")
        details_output = gr.Textbox(label="Details", interactive=False, lines=2)

        predict_button.click(
            fn=predict_churn,
            inputs=inputs,
            outputs=[verdict_output, probability_output, details_output],
        )

    with gr.Tab("Hello World"):
        gr.Markdown("# Test")

app.launch(server_name="127.0.0.1", server_port=7860, share=False)
