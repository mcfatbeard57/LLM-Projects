# LLM-research-assistant

<img width="852" alt="Screenshot 2024-08-21 at 2 54 00 PM" src="https://github.com/user-attachments/assets/34acbc62-914e-4e2f-b2c7-5dd7dd0b3071">




**Description:**

This Streamlit application leverages Large Language Models (LLMs) to assist with interview preparation. It gathers information about a target individual from various sources like Twitter, YouTube, and webpages, and then uses LLMs to generate interview questions or a one-page summary based on the collected data.

**Tools Used:**

* Streamlit: A Python library for creating interactive web applications.
* LangChain: A framework for building text processing pipelines.
* OpenAI API: Access to powerful LLMs for text generation.
* Tweepy: A library for interacting with the Twitter API.
* BeautifulSoup: A library for parsing HTML and XML documents.
* markdownify: A library for converting HTML to Markdown.
* Youtube Data API (unspecified in code, but likely used)

**Use Cases:**

* Researchers and recruiters can leverage this tool to gain insights about potential candidates and prepare effective interview questions.
* Individuals can use it to gain a comprehensive understanding of someone they are interviewing for better preparation.

**Benefits:**

* Saves time by automatically gathering information from various sources.
* Provides a data-driven approach to interview preparation.
* Generates insightful interview questions tailored to the individual's background and expertise.
* Creates a concise summary of the person, allowing for a quick grasp of their key points.

**Drawbacks:**

* Relies on the accuracy of the information available online.
* Requires an OpenAI API key, which has associated costs.
* May not always generate human-quality outputs, requiring careful review.

**Getting Started:**

1. **Install Dependencies:**
   ```bash
   pip install streamlit langchain tweepy beautifulsoup4 markdownify
   # Install Youtube Data API library if not already installed
   ```

2. **Set Up Environment Variables:**
   Create a `.env` file in your project directory and add the following lines, replacing the placeholders with your actual API keys:

   ```
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET=your_twitter_api_secret
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

**Additional Notes:**

* This project demonstrates a basic implementation. Further development could involve:
    * Error handling for invalid URLs or API keys.
    * More advanced filtering and processing of gathered information.
    * User interface enhancements for a more interactive experience.
* Remember to replace `your_placeholders` with your actual API keys before running the application.
