import os
import json
import datetime
import time
import gradio as gr  # Added at top level
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
        return {
            'ar6_summary': {
                'content': """# 🌡️ AR6 Synthesis Report: Key Findings for Policymakers

## **Physical Science Basis (Working Group I)**
- **Global Temperature Rise**: 1.1°C above 1850-1900 levels
- **Human Influence**: Unequivocally the dominant cause of warming
- **Rate of Change**: Faster than any period in over 2,000 years
- **Regional Impacts**: Every inhabited region experiencing climate change
- **Irreversible Changes**: Many changes locked in for centuries to millennia

## **Impacts, Adaptation & Vulnerability (Working Group II)**
- **Population at Risk**: 3.3-3.6 billion people highly vulnerable
- **Current Impacts**: Widespread losses and damages already occurring
- **Food Security**: 828 million people undernourished (2021)
- **Water Stress**: Up to 3 billion people experience water scarcity
- **Ecosystem Degradation**: Widespread species shifts and ecosystem changes

## **Mitigation of Climate Change (Working Group III)**
- **Emission Trends**: Global GHG emissions continued to rise
- **Peak Requirement**: Emissions must peak before 2025 for 1.5°C
- **2030 Target**: 43% reduction needed by 2030 (vs 2019 levels)
- **2050 Target**: Net zero CO₂ emissions required
- **Investment Gap**: $4 trillion annually needed in clean energy

## **Integrated Solutions**
- **Rapid Transformation**: Deep, immediate cuts across all sectors
- **Technology Readiness**: Many solutions available and cost-effective
- **Co-benefits**: Climate action improves health, economy, equity
- **Just Transitions**: Equitable pathways essential for success""",
                'sources': ['AR6 Synthesis Report SPM', 'AR6 WG1-3 Reports']
            },
            
            'urgent_actions_2030': {
                'content': """# 🚨 Critical Climate Actions Needed by 2030

## **Energy System Transformation**
### Renewable Energy Scale-up
- **Target**: 60% renewable electricity globally (vs ~30% today)
- **Solar**: Increase capacity 4x from 2020 levels
- **Wind**: Triple offshore wind capacity
- **Storage**: Deploy 120 GW of battery storage annually

### Fossil Fuel Phase-out
- **Coal**: Retire 2,400+ coal plants globally
- **Oil & Gas**: Reduce production 75% by 2050
- **Subsidies**: End $5.9 trillion in fossil fuel subsidies

## **Transport Decarbonization**
### Electric Vehicle Revolution
- **Target**: 50% of new car sales electric by 2030
- **Infrastructure**: 40 million public charging points needed
- **Heavy Transport**: 30% of trucks electric/hydrogen by 2030

### Sustainable Aviation & Shipping
- **Aviation**: 10% sustainable fuels by 2030
- **Shipping**: 5% zero-emission fuels by 2030

## **Buildings & Cities**
### Zero-Carbon Buildings
- **New Buildings**: All new buildings zero-carbon by 2030
- **Retrofits**: Deep renovation of 3% of building stock annually
- **Heat Pumps**: 600 million heat pumps by 2030

### Urban Planning
- **15-Minute Cities**: Reduce transport demand 20%
- **Green Infrastructure**: 30% urban tree canopy coverage

## **Natural Climate Solutions**
### Forest Protection
- **Deforestation**: End deforestation by 2030
- **Restoration**: 350 million hectares by 2030
- **Carbon Storage**: 5.8 GtCO₂ annually from forests

### Sustainable Agriculture
- **Regenerative Practices**: 30% of farmland by 2030
- **Food Waste**: Reduce food waste 50%
- **Diets**: 20% shift toward plant-based diets

## **Financial Requirements**
- **Total Investment**: $4-6 trillion annually
- **Clean Energy**: $1.6-3.8 trillion annually
- **Nature**: $350 billion annually
- **Adaptation**: $140-300 billion annually by 2030""",
                'sources': ['AR6 WG3 Ch5', '1.5°C Special Report', 'AR6 Synthesis']
            },
            
            'carbon_budgets': {
                'content': """# 🎯 Carbon Budgets: The Climate Math Explained

## **What is a Carbon Budget?**
A carbon budget is the maximum amount of CO₂ that can be emitted to limit global warming to a specific temperature target with a given probability.

## **Current Carbon Budget Status (AR6 Update)**
### For 1.5°C Target (50% probability)
- **Remaining Budget**: ~400 GtCO₂ from 2020
- **At Current Rate**: ~10 years remaining (40 GtCO₂/year)
- **Updated from AR5**: Previously ~1,000 GtCO₂ (methodology improved)

### For 2°C Target (67% probability)  
- **Remaining Budget**: ~1,150 GtCO₂ from 2020
- **At Current Rate**: ~29 years remaining

## **Key Insights**
### Why Budgets Matter
- **Linear Relationship**: Cumulative CO₂ determines peak warming
- **Location Independent**: Doesn't matter where emissions occur
- **Timing Flexible**: When matters less than total amount

### Budget Uncertainties
- **Non-CO₂ GHGs**: Methane, N₂O add ~0.4°C warming
- **Climate Feedbacks**: Could reduce budget by 100-200 GtCO₂
- **Temperature Response**: ±0.2°C uncertainty in climate sensitivity

## **Sharing the Carbon Budget**
### Equity Principles
- **Historical Responsibility**: Developed countries 79% of budget since 1850
- **Capability**: High-income countries have greater resources
- **Development Needs**: Developing countries need emissions for basic needs

### Fair Share Approaches
- **Per Capita**: Equal emissions rights per person
- **Capability**: Based on ability to pay
- **Grandfathering**: Continue current emission shares
- **Hybrid**: Combination of principles

## **Policy Implications**
### Carbon Pricing
- **Social Cost**: $50-100/tCO₂ by 2030
- **Price Signal**: Guide investment decisions
- **Revenue Use**: Support just transitions

### National Budgets
- **NDC Alignment**: Check if commitments fit global budget
- **Sectoral Allocation**: Distribute budgets across sectors
- **Monitoring**: Track progress against allocations""",
                'sources': ['AR6 WG1 Ch5', 'AR6 WG3 Ch2', 'Carbon Budget Studies']
            },
            
            'climate_impacts_risks': {
                'content': """# 🌊 Climate Impacts and Risks: What We Face

## **Observed Impacts (Already Happening)**
### Temperature Extremes
- **Heat Waves**: 3-5x more frequent deadly heat
- **Record Temperatures**: New records set regularly
- **Urban Heat**: Cities 2-5°C warmer than surroundings

### Water Cycle Changes
- **Droughts**: 1.5 billion people affected by severe drought
- **Floods**: 1.65 billion people exposed to flood risk
- **Glaciers**: Lost 6,000 km³ of ice since 2000
- **Sea Ice**: Arctic sea ice declining 13% per decade

### Ecosystem Impacts
- **Coral Bleaching**: 50% of shallow coral reefs bleached since 2009
- **Species Shifts**: 1,000+ species moving poleward
- **Forest Fires**: 2x increase in large fires since 1980s
- **Permafrost**: Thawing releases methane and CO₂

## **Near-term Risks (2021-2040)**
### Human Systems
- **Food Security**: 8-80 million more at hunger risk by 2050
- **Water Stress**: 1-4 billion additional people affected
- **Health**: 250,000 additional deaths annually (2030-2050)
- **Migration**: 200 million-1 billion climate migrants by 2050

### Natural Systems
- **Biodiversity**: 1 million species threatened with extinction
- **Ocean**: pH decrease of 0.3-0.4 units by 2100
- **Forests**: Amazon approaching tipping point
- **Wetlands**: 50% of coastal wetlands lost to sea level rise

## **Long-term Risks (Post-2040)**
### Irreversible Changes
- **Sea Level Rise**: 0.43-2.84m by 2100 (scenario dependent)
- **Ice Sheets**: West Antarctic ice sheet potentially unstable
- **Permafrost**: 30-70% thaw by 2100
- **Ocean Circulation**: AMOC weakening, possible collapse

### Tipping Points
- **Arctic Sea Ice**: Summer ice-free by 2050
- **Amazon Rainforest**: Shift to savanna possible
- **Coral Reefs**: 99% loss at 2°C warming
- **Mountain Glaciers**: Many will disappear this century

## **Compound and Cascading Risks**
### System Interactions
- **Food-Water-Energy**: Nexus stressed simultaneously
- **Supply Chains**: Climate disrupts global trade
- **Financial Systems**: Stranded assets, credit risks
- **Social Stability**: Resource conflicts, governance stress

### Regional Risk Clusters
- **Small Islands**: Sea level + storms + ecosystem loss
- **Arctic**: Ice loss + permafrost + Indigenous cultures
- **Mediterranean**: Heat + drought + fire + migration
- **Monsoon Asia**: Extreme weather + sea level + population

## **Adaptation Limits**
### Hard Limits
- **Ecosystem Tolerance**: Species can't adapt fast enough
- **Infrastructure**: Some areas become uninhabitable
- **Agriculture**: Crop failures beyond adaptation capacity

### Soft Limits
- **Economic**: Costs exceed benefits of adaptation
- **Social**: Lack of knowledge, technology, or institutions
- **Cultural**: Conflicts with values or ways of life""",
                'sources': ['AR6 WG2 SPM', 'AR6 WG1 Ch11-12', 'Special Report on Extremes']
            }
        }
    
    def format_response(self, content: str, sources: List[str] = None) -> str:
        """Format response with sources"""
        formatted = content
        if sources:
            formatted += f"\n\n**📚 Sources**: {', '.join(sources)}"
        return formatted
    
    def get_mock_response(self, message: str, report_focus: str) -> Tuple[str, List[str]]:
        """Generate mock responses for demonstration"""
        message_lower = message.lower()
        
        # Keyword matching for appropriate responses
        if any(word in message_lower for word in ['ar6', 'synthesis', 'key findings', 'summary']):
            knowledge = self.ipcc_knowledge['ar6_summary']
            return knowledge['content'], knowledge['sources']
        
        elif any(word in message_lower for word in ['urgent', '2030', 'actions', 'immediate']):
            knowledge = self.ipcc_knowledge['urgent_actions_2030']
            return knowledge['content'], knowledge['sources']
        
        elif any(word in message_lower for word in ['carbon budget', 'budget', 'emissions budget']):
            knowledge = self.ipcc_knowledge['carbon_budgets']
            return knowledge['content'], knowledge['sources']
        
        elif any(word in message_lower for word in ['impacts', 'risks', 'effects', 'consequences']):
            knowledge = self.ipcc_knowledge['climate_impacts_risks']
            return knowledge['content'], knowledge['sources']
        
        else:
            # General response with suggestions
            return """I can help you explore IPCC climate reports! Here are some key areas:

🎯 **Popular Topics**:
- "Summarize AR6 key findings" - Latest IPCC conclusions
- "What urgent actions are needed by 2030?" - Critical next steps  
- "Explain carbon budgets" - The climate math
- "What are the main climate risks?" - Impacts and consequences

📊 **Report Coverage**:
- **AR6 (2021-2023)**: Latest comprehensive assessment
- **Special Reports**: 1.5°C, Land, Ocean focus
- **AR5 (2013-2014)**: Previous major assessment
- **Cross-report analysis**: Evolution of science

🔍 **Analysis Types**:
- Policy summaries for decision-makers
- Technical deep-dives
- Regional impact assessments  
- Sectoral transformation pathways

What specific aspect of climate science would you like to explore?""", ['IPCC Knowledge Base']
    
    def call_llm_api(self, messages: List[Dict], model: str, temperature: float, max_tokens: int) -> Tuple[str, List[str]]:
        """Call appropriate LLM API based on model selection"""
        if model == 'mock':
            # Simulate API delay
            time.sleep(1)
            user_message = messages[-1]['content'] if messages else ""
            return self.get_mock_response(user_message, 'all')
        
        # Prepare system prompt for IPCC focus
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
    # Import Gradio again to ensure availability in this scope
    import gradio as gr
    
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
