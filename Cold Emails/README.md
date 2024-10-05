# Cold-Emails-with-AI-agents

# ColdEmailAI

## Overview

ColdEmailAI is a Python-based application designed to assist users in crafting personalized cold emails by leveraging advanced AI models. The application utilizes the Groq API and integrates with OpenAI's API to generate market research queries and formulate concise email content based on the user's target industry or company.

## Features

- **Market Research Queries**: Automatically generates four targeted search queries based on the user's input regarding a specific industry or company.
- **Web Search Summarization**: Conducts web searches for the generated queries and provides concise summaries of the results.
- **Cold Email Generation**: Utilizes the summarized research to create personalized cold emails, focusing on the recipient's pain points, potential clients, and online resources.

## Getting Started

### Prerequisites

- Python 3.x
- Groq API Key
- OpenAI API Key (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ColdEmailAI.git
   ```
   
2. Navigate into the project directory:
   ```bash
   cd ColdEmailAI
   ```

3. Install required packages (if any):
   ```bash
   pip install -r requirements.txt
   ```
.

### Usage

1. Run the application:
   ```bash
   python ColdEmailAI_v1_basic.py
   ```

2. Enter the industry or company you wish to target when prompted.

3. The application will generate market research queries, perform web searches, and output a personalized cold email based on the gathered information.

## Code Structure

- `get_user_input()`: Prompts the user for input regarding the target industry or company.
- `query_agent(target)`: Generates market research queries based on the user's input.
- `web_search_agent(query)`: Performs web searches for the generated queries and summarizes the results.
- `cold_email_agent(target, search_results)`: Creates a personalized cold email using the market research findings.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.
