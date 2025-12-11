# ValsAI Finance Agent Benchmark Codebase

Finance Agent is a tool for financial research and analysis that leverages large language models and specialized financial tools to answer complex queries about companies, financial statements, and SEC filings.

This repo contains the codebase to run the agent that was used to create the benchmark [Finance Agent](https://www.vals.ai/benchmarks/finance_agent). It makes it easy to test the harness with any question or model of your choices.

## Overview

This agent connects to various data sources including:

- SEC EDGAR database
- Google web search
- HTML page parsing capabilities
- Information retrieval and analysis

It uses a configurable LLM backend (OpenAI, Anthropic, Google, etc.) to orchestrate these tools and generate comprehensive financial analysis.
For all models except Anthropic's, we use OpenAI SDK for the agent.

## Installation

Run the following command to install the enviromnent:

```
pip install -r requirements.txt
```

We recommend creating and using a Conda environment for this. For detailed instructions on managing Conda environments, see the [official Conda documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Running both green and white agents

```bash

# Start green agent
python start_green_agent.py --mode white_agent --port 9001

# Start white agent
python start_white_agent.py --port 9002

# Test both agents
python test_white_agent_mode.py
```

## Railway Deployment for AgentBeats Integration

This repository is configured for **separate deployments** of green and white agents on Railway to integrate with AgentBeats. The deployment follows the structure from the [CS194_Ecom_GreenAgent](https://github.com/LupoSun/CS194_Ecom_GreenAgent/tree/main) reference repository.

### Separate Deployments

You can deploy:
- **Green Agent**: The evaluation agent that tests other agents (uses `Procfile.green`)
- **White Agent**: The agent being tested (uses `Procfile.white`)
- **Both**: As separate Railway services in the same project

### Quick Deployment

#### Deploy Green Agent

1. **Connect Repository to Railway**:
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure for Green Agent**:
   - Rename `Procfile.green` to `Procfile`, OR
   - Set start command: `python start_green_agent.py --host 0.0.0.0 --port $PORT --mode white_agent`

3. **Set Environment Variables** (optional for green agent):
   ```
   OPENAI_API_KEY=your_openai_key  # Only needed for direct mode
   ANTHROPIC_API_KEY=your_anthropic_key  # Only needed for direct mode
   ```

4. **Deploy**: Railway will automatically deploy

#### Deploy White Agent

1. **Create New Service** in Railway project (or separate project)

2. **Configure for White Agent**:
   - Rename `Procfile.white` to `Procfile`, OR
   - Set start command: `python start_white_agent.py --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables** (required for white agent):
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   SERP_API_KEY=your_serpapi_key
   SEC_API_KEY=your_sec_api_key
   ```

4. **Deploy**: Railway will automatically deploy

### Configuration Files

The following files are configured for Railway deployment:

**Green Agent:**
- **`Procfile.green`**: Web process for green agent
- **`railway-green.json`**: Railway config for green agent

**White Agent:**
- **`Procfile.white`**: Web process for white agent
- **`railway-white.json`**: Railway config for white agent

**Shared:**
- **`Procfile`**: Default (currently set for green agent)
- **`railway.json`**: Default (currently set for green agent)
- **`runtime.txt`**: Python version specification
- **`.railwayignore`**: Files to exclude from deployment

### AgentBeats Integration

Once deployed on Railway:

- **Green Agent URL**: `https://your-green-agent.railway.app`
- **White Agent URL**: `https://your-white-agent.railway.app`

Both agents:
1. Automatically use Railway's `PORT` environment variable
2. Construct URLs from `RAILWAY_PUBLIC_DOMAIN` if available
3. Are accessible via A2A protocol at `/.well-known/agent-card`

### Testing Deployed Agents

```python
import asyncio
from my_util.my_a2a import send_message

async def test():
    green_url = "https://your-green-agent.railway.app"
    white_url = "https://your-white-agent.railway.app"
    
    # Test green agent (it will test the white agent)
    response = await send_message(
        green_url,
        f"""
        Your task is to test the finance agent located at:
        <white_agent_url>
        {white_url}
        </white_agent_url>
        You should use the following configuration:
        <config>
        {{
          "mode": "white_agent",
          "max_iterations": 10,
          "question": "What was Apple's revenue in 2023?"
        }}
        </config>
        """
    )
    print(response)

asyncio.run(test())
```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Environment Setup

Create a `.env` file with the necessary API keys:

```
# LLM API Keys
# Note: It's only necessary to set the API keys for the models you plan on using
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
TOGETHER_API_KEY=your_together_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
GROK_API_KEY=your_grok_api_key
COHERE_API_KEY=cohere_api_key

# Tool API Keys
SERP_API_KEY=your_serpapi_key
SEC_API_KEY=your_sec_api_key
```

You can create a SERP API key [here](https://serpapi.com/), and an SEC API key [here](https://sec-api.io/).

## Running the Agent

To experiment with the agent, you can run the following command:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?"
```

You can specify multiple questions at once:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?" "What was NFLX's revenue in 2024?"
```

To specify a specific model, use the `--model` flag:

```bash
python run_agent.py --questions "What was Apple's revenue in 2023?" --model openai/gpt-4o
```

You can also specify a list of questions in a text file, one question per line, with the following command:

```bash
python run_agent.py --question-file my_questions.txt
```

For a full list of parameters, please run:

```bash
python run_agent.py --help
```

The default configuration is the one we used to run the benchmark.

## Available Tools

- `google_web_search`: Search the web for information
- `edgar_search`: Search the SEC's EDGAR database for filings
- `parse_html_page`: Parse and extract content from web pages
- `retrieve_information`: Access stored information from previous steps

## Available Models

The following models are supported by this repo:

```
- openai/gpt-4o-2024-08-06
- openai/gpt-4o-mini-2024-07-18
- openai/o3-mini-2025-01-31
- openai/o1-2024-12-17
- anthropic/claude-3-5-haiku-20241022
- anthropic/claude-3-7-sonnet-20250219
- anthropic/claude-3-7-sonnet-20250219-thinking
- google/gemini-2.0-flash-001
- google/gemini-2.5-pro-preview-03-25
- together/meta-llama/Llama-3.3-70B-Instruct-Turbo
- together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- together/meta-llama/Llama-4-Scout-17B-16E-Instruct
- mistralai/mistral-small-2503
- grok/grok-3-beta
- grok/grok-3-mini-fast-beta-high-reasoning
- grok/grok-3-mini-fast-beta-low-reasoning
- cohere/command-a-03-2025
- openai/gpt-4.1-mini-2025-04-14
- openai/gpt-4.1-nano-2025-04-14
- openai/gpt-4.1-2025-04-14
- openai/o4-mini-2025-04-16
- openai/o3-2025-04-16
```

## Logs and Output

The agent writes detailed logs to the `logs` directory, including:

- Session-specific logs with timestamps
- Tool usage statistics
- Token usage
- Error tracking
