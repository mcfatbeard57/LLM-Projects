from groq import Groq
# import OpenAI

ngroq_API_Key = "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"
OpenAI_API_Key = ""
groq_client = Groq(api_key=ngroq_API_Key)

# chat_completion = groq_client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ], model="llama3-8b-8192",
# )
# print(chat_completion.choices[0].message.content)


# Write cold emails using llama and preplexity API
# client = OpenAI(api_key=OpenAI_API_Key,base_url="https://api perplexity.ai")


def get_user_input():
  user_input = input("Please enter the industry or company you want to target: ")
  return user_input

# openai_client = OpenAI(
#     api_key="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",
#     base_url="https://api.perplexity.ai"
# )

def query_agent(target):
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an AI assistant that writes concise search queries for market research.
                Create 4 short search queries based on the given target industry or company.
                Query 01: biggest pain points faced by this avatar
                Query 02: biggest companies in this industry
                Query 03: how companies in this industry get clients
                Query 04: where to find companeis in this industry online
                IMPORTANT: Respond with only the queries, one per line."""
            },
            {
                "role": "user",
                "content": f"Here's the industry / company to perform market research on: #### {target} ####"
            }
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=1,
        # top_k=50,
        # repetition_penalty=1,
        stop=["<|eot_id|>"],
        stream = False
    )
    queries = response.choices[0].message.content.split('\n')
    return [query.strip() for query in queries if query.strip()]



def web_search_agent(query):
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a web search assistant. Provide a concise summary of the search results."},
            {"role": "user", "content": f"Search the web for: {query}"}
        ],
    )
    return response.choices[0].message.content


def cold_email_agent(target, search_results):
    # Combine all search results into a single string
    combined_results = "\n".join(search_results)

    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": """You are an expert cold email writer.
                Your task is to write concise and personalized cold emails based on the Market Research given to you.
                Make sure to utilize all 4 areas of the research I(pain points, companies, clients, and online sources)
                Focus on describing what the target avatar will get, add an appealing guarantee.
                Keep the email concise and use plain English.
                DO NOT OUTPUT ANY OTHER TEXT !!! ONLY THE COLD EMAIL ITSELF!.
                """
            },
            {
                "role": "user",
                "content": f"Here is the target avatar: {target} \n Here is the market research: #### {combined_results} #### ONLY OUTPUT THE EMAIL ITSELF. NO OTHER TEXT!!"
            }
        ],
        max_tokens=500,
        temperature=0.1,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"]
    )

    return response.choices[0].message.content


# Example usage
user_input = get_user_input()
generated_queries = query_agent(user_input)
print(generated_queries)

# Example usage
search_results = []
for query in generated_queries:
    print("Searching for...", query)
    result = web_search_agent(query)
    search_results.append(result)
    
print(search_results)

# Example usage
cold_email = cold_email_agent(user_input, search_results)
print(cold_email)