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

```
conda create -n myenv python=3.13
```

## Running both green and white agents locally 

```bash

# Start green agent
python start_green_agent.py --mode white_agent --port 9001

# Start white agent
python start_white_agent.py --port 9002

# Test both agents
python test_white_agent_mode.py
```

## Environment Setup for Local Deployment

Create a `.env` file with the necessary API keys:

```
# LLM API Keys
# Note: It's only necessary to set the API keys for the models you plan on using
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
# Tool API Keys

SERP_API_KEY=your_serpapi_key
SEC_API_KEY=your_sec_api_key
```

You can create a SERP API key [here](https://serpapi.com/), and an SEC API key [here](https://sec-api.io/).


## Railway Deployment for AgentBeats Integration

This repository is configured for **separate deployments** of green and white agents on Railway to integrate with AgentBeats.

### Quick Deployment

1. **Connect Repository to Railway**:

   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure for Green Agent using green branch**:

   - Set environment variable `CLOUDRUN_HOST=finance-green-production.up.railway.app`

3. **Configure for White Agent using WHITE branch**:

   - Set environment variable `CLOUDRUN_HOST=finance-white-production.up.railway.app`

4. **Set Environment Variables for both**:

   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   SERP_API_KEY=your_serp_key
   SEC_API_KEY=your_sec_key
   HTTPS_ENABLED=true
   ```

5. **Deploy**: Railway will automatically deploy, remember to update the deployment if changes are made!

### Configuration Files

The following files are configured for Railway deployment:

**Green Agent (in green branch):**

- **`Procfile`**: Web process for green agent
- **`railway-green.json`**: Railway config for green agent

**White Agent (in white branch):**

- **`Procfile`**: Web process for white agent
- **`railway-white.json`**: Railway config for white agent

**Shared:**

- **`runtime.txt`**: Python version specification
- **`.railwayignore`**: Files to exclude from deployment

### AgentBeats Integration

Once deployed on Railway:

- **Green Agent URL**: `https://finance-green-production.up.railway.app`
- **White Agent URL**: `https://finance-white-production.up.railway.app`