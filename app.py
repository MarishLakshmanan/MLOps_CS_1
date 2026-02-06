import gradio as gr
from huggingface_hub import InferenceClient
import os

pipe = None
stop_inference = False

# UI refresh: cleaner, modern, no logic changes
fancy_css = """
:root{
  --bg: #0b1220;
  --panel: rgba(255,255,255,0.06);
  --panel-2: rgba(255,255,255,0.08);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.72);
  --border: rgba(255,255,255,0.12);
  --accent: #7c5cff;
  --accent-2:#22c55e;
  --shadow: 0 18px 50px rgba(0,0,0,0.35);
}

.gradio-container{
  max-width: 980px !important;
  margin: 24px auto !important;
  padding: 0 !important;
  background: transparent !important;
}

/* Page background */
body{
  background:
    radial-gradient(1200px 600px at 15% 10%, rgba(124,92,255,0.28), transparent 60%),
    radial-gradient(900px 520px at 85% 20%, rgba(34,197,94,0.18), transparent 55%),
    radial-gradient(900px 520px at 50% 95%, rgba(56,189,248,0.12), transparent 60%),
    var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

/* App shell */
#app-shell{
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  background: var(--panel);
  box-shadow: var(--shadow);
}

/* Header */
#app-header{
  padding: 18px 18px 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border-bottom: 1px solid var(--border);
}

#app-title{
  margin: 0;
  font-size: 1.35rem;
  letter-spacing: 0.2px;
  color: var(--text);
}

#app-subtitle{
  margin: 6px 0 0;
  font-size: 0.95rem;
  color: var(--muted);
}

/* Login button aligns nicely */
#login-wrap{
  display:flex;
  justify-content:flex-end;
  align-items:center;
  gap: 10px;
}

/* Main layout */
#app-main{
  padding: 16px;
}

/* Buttons */
.gr-button{
  background: linear-gradient(180deg, rgba(124,92,255,1), rgba(124,92,255,0.86)) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  transition: transform 0.08s ease, filter 0.2s ease;
}
.gr-button:hover{
  filter: brightness(1.05);
}
.gr-button:active{
  transform: translateY(1px);
}

/* Make secondary buttons look subtle if present */
.gr-button.gr-button-secondary{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
}

/* Inputs & panels */
.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown,
.gr-number input,
.gr-slider{
  border-radius: 12px !important;
}
.gr-box, .gr-panel{
  background: var(--panel-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

/* Chat styling */
.gr-chatbot{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  background: rgba(0,0,0,0.18) !important;
}
.gr-chat-message{
  font-size: 15px;
  line-height: 1.45;
}
.gr-chat-message .message{
  border-radius: 14px !important;
}
.gr-chat-message.user .message{
  background: rgba(124,92,255,0.18) !important;
  border: 1px solid rgba(124,92,255,0.22) !important;
}
.gr-chat-message.bot .message{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--border) !important;
}

/* Sliders accent */
input[type="range"]::-webkit-slider-thumb{
  background: var(--accent) !important;
}
"""


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool,
):
    global pipe

    # Build messages from history
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local_model:
        print("[MODE] local")
        from transformers import pipeline
        import torch

        if pipe is None:
            pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

        # Build prompt as plain text
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt) :]
        yield response.strip()

    else:
        print("[MODE] api")

        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response


chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
)

with gr.Blocks(css=fancy_css) as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=8):
                gr.Markdown(
                    """
                    <h1 id="app-title">✨ Fancy AI Chatbot</h1>
                    <p id="app-subtitle">Switch between Hugging Face API and a local Phi-3 model. Same brain, nicer outfit.</p>
                    """
                )
            with gr.Column(scale=4, elem_id="login-wrap"):
                gr.LoginButton()

        with gr.Column(elem_id="app-main"):
            chatbot.render()

if __name__ == "__main__":
    demo.launch()
