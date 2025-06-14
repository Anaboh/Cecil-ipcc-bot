import os
import json
import datetime
import time
from typing import List, Dict, Tuple, Optional
import pandas as pd
import random
import subprocess
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Install required packages
def install_packages():
    packages = [
        'gradio>=4.0.0',
        'openai>=1.0.0',
        'anthropic>=0.18.0',
        'google-generativeai>=0.3.0',
        'requests>=2.31.0',
        'python-dotenv>=1.0.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check API availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class IPCCLLMAgent:
    """IPCC Climate Reports LLM Agent"""
    
    def __init__(self):
        self.conversation_history = []
        self.ipcc_reports = {
            'all': {'name': 'All IPCC Reports', 'color': '🌍'},
            'ar6': {'name': 'AR6 (2021-2023)', 'color': '🌱'},
            'ar5': {'name': 'AR5 (2013-2014)', 'color': '📊'},
            'special': {'name': 'Special Reports', 'color': '⚠️'},
            'ar4': {'name': 'AR4 (2007)', 'color': '📄'}
        }
        
        self.llm_models = {
            'gpt-4': {'name': 'GPT-4 Turbo', 'provider': 'OpenAI'},
            'gpt-3.5': {'name': 'GPT-3.5 Turbo', 'provider': 'OpenAI'},
            'claude-3': {'name': 'Claude 3 Sonnet', 'provider': 'Anthropic'},
            'gemini': {'name': 'Gemini Pro', 'provider': 'Google'},
            'mock': {'name': 'Mock AI (Demo)', 'provider': 'Local'}
        }
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None
        self.setup_api_clients()
        
        # Print API status
        print("\n🌐 API Client Status:")
        print(f"  OpenAI: {'✅ Available' if self.openai_client else '❌ Not available'}")
        print(f"  Anthropic: {'✅ Available' if self.anthropic_client else '❌ Not available'}")
        print(f"  Gemini: {'✅ Available' if self.gemini_client else '❌ Not available'}")
        
        # IPCC Knowledge Base
        self.ipcc_knowledge = self.load_ipcc_knowledge()
    
    def setup_api_clients(self):
        """Setup API clients with keys from environment"""
        # Get API keys from environment
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        # Initialize OpenAI
        if openai_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
            except Exception as e:
                print(f"⚠️ OpenAI initialization error: {str(e)}")
        
        # Initialize Anthropic
        if anthropic_key and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except Exception as e:
                print(f"⚠️ Anthropic initialization error: {str(e)}")
        
        # Initialize Gemini
        if gemini_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"⚠️ Gemini initialization error: {str(e)}")
    
    def load_ipcc_knowledge(self) -> Dict:
        """Load IPCC knowledge base with comprehensive climate information"""
        # ... (Keep your existing knowledge base content unchanged) ...
        # Return your IPCC knowledge dictionary
        return {
            'ar6_summary': { ... },
            'urgent_actions_2030': { ... },
            'carbon_budgets': { ... },
            'climate_impacts_risks': { ... }
        }
    
    def format_response(self, content: str, sources: List[str] = None) -> str:
        """Format response with sources"""
        formatted = content
        if sources:
            formatted += f"\n\n**📚 Sources**: {', '.join(sources)}"
        return formatted
    
    def get_mock_response(self, message: str, report_focus: str) -> Tuple[str, List[str]]:
        """Generate mock responses for demonstration"""
        # ... (Keep your existing mock response logic unchanged) ...
        # Return your mock response content and sources
        pass
    
    def call_llm_api(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, List[str]]:
        """Call appropriate LLM API based on model selection"""
        if model == 'mock':
            time.sleep(1)
            user_message = messages[-1]['content'] if messages else ""
            return self.get_mock_response(user_message, 'all')
        
        # Prepare system prompt
        system_prompt = """You are an expert IPCC climate reports analyst. Provide accurate, science-based responses using information from IPCC Assessment Reports (AR4, AR5, AR6) and Special Reports. 

Key guidelines:
- Base responses on IPCC findings and scientific consensus
- Include specific data, figures, and projections when relevant  
- Distinguish between different confidence levels and scenarios
- Cite specific IPCC reports and sections when possible
- Explain complex concepts clearly for policymakers and general audiences
- Highlight policy implications and actionable insights
- Note uncertainties and ranges in projections"""
        
        print(f"📡 Calling LLM API: {model} with {len(messages)} messages")
        
        try:
            if model.startswith('gpt') and self.openai_client:
                model_name = "gpt-4-turbo" if model == 'gpt-4' else "gpt-3.5-turbo"
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                return content, [f"{self.llm_models[model]['name']} API Response"]
            
            elif model == 'claude-3' and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.content[0].text
                return content, [f"{self.llm_models[model]['name']} API Response"]
            
            elif model == 'gemini' and self.gemini_client:
                # Convert messages to Gemini format
                conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                full_prompt = f"{system_prompt}\n\nConversation:\n{conversation_text}"
                
                response = self.gemini_client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text, [f"{self.llm_models[model]['name']} API Response"]
            
            else:
                print(f"⚠️ Falling back to mock: Model {model} not available")
                user_message = messages[-1]['content'] if messages else ""
                content, sources = self.get_mock_response(user_message, 'all')
                return f"⚠️ Selected model not available. Using demo mode.\n\n{content}", sources
                
        except Exception as e:
            error_msg = f"API Error: {str(e)[:200]}\n\nFalling back to demo mode with IPCC knowledge base."
            print(error_msg)
            user_message = messages[-1]['content'] if messages else ""
            content, sources = self.get_mock_response(user_message, 'all')
            return f"{error_msg}\n\n{content}", sources
    
    def process_message(self, message: str, history: List, model: str, temperature: float, max_tokens: int, report_focus: str) -> Tuple[List, str]:
        """Process user message and return updated history"""
        if not message.strip():
            return history, ""
        
        # Add user message to history
        history.append([message, None])
        
        # Prepare messages for LLM
        messages = []
        for human, assistant in history[:-1]:
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})
        
        # Get LLM response
        try:
            content, sources = self.call_llm_api(messages, model, temperature, max_tokens)
            
            # Format response with sources
            formatted_response = self.format_response(content, sources)
            
            # Update history with response
            history[-1][1] = formatted_response
            
        except Exception as e:
            error_response = f"⚠️ Error processing request: {str(e)}\n\nPlease try again or check your API configuration."
            history[-1][1] = error_response
            print(f"🔥 Critical error: {str(e)}")
        
        return history, ""

# Quick prompts for easy access
quick_prompts = [
    "Summarize AR6 key findings for policymakers",
    "What urgent climate actions are needed by 2030?", 
    "Explain carbon budgets in simple terms",
    "What are the main climate risks and impacts?",
    "Compare climate projections between AR5 and AR6",
    "What are the most effective mitigation strategies?",
    "How is climate change affecting different regions?",
    "What adaptation measures are recommended?"
]

def create_interface():
    """Create Gradio interface for the IPCC LLM Agent"""
    
    with gr.Blocks(
        title="🌍 IPCC Climate Reports LLM Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🌍 IPCC Climate Reports LLM Agent</h1>
            <p style='font-size: 18px; margin: 10px 0;'>AI-Powered Analysis of Climate Science Reports</p>
            <p style='font-size: 14px; opacity: 0.9;'>Multiple LLM Support • IPCC AR4/AR5/AR6 Knowledge Base</p>
        </div>
        """)
        
        with gr.Row():
            # Main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=600,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about IPCC climate reports, projections, impacts, or policies...",
                        label="Your Question",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                
                # Quick prompts
                gr.HTML("<h3>💡 Quick Prompts</h3>")
                with gr.Row():
                    for i in range(0, len(quick_prompts), 2):
                        with gr.Column():
                            if i < len(quick_prompts):
                                gr.Button(
                                    quick_prompts[i], 
                                    size="sm"
                                ).click(
                                    lambda x=quick_prompts[i]: x,
                                    outputs=msg
                                )
                            if i+1 < len(quick_prompts):
                                gr.Button(
                                    quick_prompts[i+1], 
                                    size="sm"
                                ).click(
                                    lambda x=quick_prompts[i+1]: x,
                                    outputs=msg
                                )
            
            # Settings sidebar
            with gr.Column(scale=1):
                gr.HTML("<h3>🔧 Configuration</h3>")
                
                model_choice = gr.Dropdown(
                    choices=list(IPCCLLMAgent().llm_models.keys()),
                    value="mock",
                    label="AI Model",
                    info="Select your preferred LLM"
                )
                
                report_focus = gr.Dropdown(
                    choices=list(IPCCLLMAgent().ipcc_reports.keys()),
                    value="all",
                    label="Report Focus",
                    info="Focus on specific IPCC reports"
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Response creativity (0=focused, 1=creative)"
                )
                
                max_tokens = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Max Tokens",
                    info="Response length limit"
                )
                
                gr.HTML("<h3>🔑 API Keys</h3>")
                gr.HTML("""
                <p style='font-size: 12px; color: #666;'>
                Set API keys as environment variables:<br>
                • OPENAI_API_KEY<br>
                • ANTHROPIC_API_KEY<br>
                • GEMINI_API_KEY<br><br>
                Or use 'Mock AI' for demo mode.
                </p>
                """)
                
                # Model status
                gr.HTML("<h3>📊 Model Status</h3>")
                status_html = f"""
                <div style='font-size: 12px;'>
                {'✅' if OPENAI_AVAILABLE else '❌'} OpenAI: {'Available' if os.getenv('OPENAI_API_KEY') else 'No API Key'}<br>
                {'✅' if ANTHROPIC_AVAILABLE else '❌'} Anthropic: {'Available' if os.getenv('ANTHROPIC_API_KEY') else 'No API Key'}<br>
                {'✅' if GEMINI_AVAILABLE else '❌'} Gemini: {'Available' if os.getenv('GEMINI_API_KEY') else 'No API Key'}<br>
                ✅ Mock AI: Always Available
                </div>
                """
                gr.HTML(status_html)
        
        # Event handlers
        def respond(message, history, model, temp, tokens, report):
            agent = IPCCLLMAgent()
            return agent.process_message(message, history, model, temp, tokens, report)
        
        # Send button click
        send_btn.click(
            respond,
            inputs=[msg, chatbot, model_choice, temperature, max_tokens, report_focus],
            outputs=[chatbot, msg]
        )
        
        # Enter key press
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_choice, temperature, max_tokens, report_focus],
            outputs=[chatbot, msg]
        )
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd; color: #666;'>
            <p>🚀 <strong>IPCC Climate Reports LLM Agent</strong> - Powered by AI for Climate Action</p>
            <p style='font-size: 12px;'>This tool provides AI-generated summaries of IPCC reports for educational and research purposes.</p>
        </div>
        """)
    
    return interface
