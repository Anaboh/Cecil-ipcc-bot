import os
import gradio as gr
from dotenv import load_dotenv
from ipcc_colab_agent import IPCCLLMAgent, create_interface

# Load environment variables first
load_dotenv()

print("🌍 Starting IPCC Climate Bot...")
print("📚 Loading IPCC knowledge base...")
print("🔧 Setting up model configurations...")

# Initialize the agent
agent = IPCCLLMAgent()

# Create and launch interface
interface = create_interface()

print("🚀 Launching interface...")
print("💡 Tip: Use 'Mock AI' model for demo mode, or configure API keys for full functionality")

# Launch the interface
interface.launch(
    server_name="0.0.0.0",
    server_port=8080,
    share=False,
    show_error=True,
    quiet=False
)
