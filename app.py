import os
import gradio as gr
import time
from ipcc_colab_agent import IPCCLLMAgent, create_interface

# Initialize the agent
agent = IPCCLLMAgent()

# Create Gradio interface
interface = create_interface()

if __name__ == "__main__":
    # Mobile-friendly launch settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        favicon_path="favicon.ico",
        inbrowser=False
    )
    # Keep the app running
    while True:
        time.sleep(600)
