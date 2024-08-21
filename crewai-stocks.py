# IMPORT DAS LIBS
import json
import os
from datetime import datetime, timedelta

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


# Função para obter preços de ações usando o Yahoo Finance API
def fetch_stock_price(ticket):
    # Calcula a data de um ano atrás
    one_year_ago = datetime.now() - timedelta(days=365)

    # Converte para o formato de string que o yf.download espera
    start_date = one_year_ago.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Baixa os preços históricos das ações para o período especificado
    stock = yf.download(ticket, start=start_date, end=end_date)
    return stock


# Cria uma ferramenta do Yahoo Finance para buscar preços de ações
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket),
)


# Configura o modelo OpenAI GPT para análise de preços de ações
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Cria um agente para análise de preços de ações
stockPriceAnalyst = Agent(
    role="Senior Stock Price Analyst",
    goal="Find the {ticket} stock price and analyze trends",
    backstory="""You're highly experienced in analyzing the price of a specific stock
    and making predictions about it's future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False,
)

# Define uma tarefa para analisar o histórico de preços das ações e criar uma análise de tendência
getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways",
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock = 'APPL, price'
    """,
    agent=stockPriceAnalyst,
)


# Cria uma ferramenta de busca para obter notícias relacionadas ao mercado
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

# Cria um agente para análise de notícias do mercado
newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with
    the news context. For each requested stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years.
    
    You're also a master level analyst in the traditional markets and have deep understanding of human psychology.
    
    You understand news, their titles and information, but you look at those with a healthy dose of skepticism. 
    You consider also the source of the news articles.
    """,
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False,
)


# Define uma tarefa para compilar notícias do mercado e análises de tendências
get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now()}.
    
    Compose the results into a helpfull report""",
    expected_output="""A summary of the overall market and one sentence summary for each requested asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst,
)


# Cria um agente para escrever uma análise detalhada sobre a ação
stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends price and news and write an insightfull compelling and informative 3 paragraphs long newsletter based on the stock report and price trend.""",
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences.
    
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analysis.
    You're able to hold multiple opinions when analyzing anything.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
)


# Define uma tarefa para escrever uma análise detalhada e criar um boletim informativo
writeAnalysis = Task(
    description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analysis of stock trend and news summary.
    """,
    expected_output="""An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
    
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed score
    - summary - key facts and concrete future trend prediction - up, down or sideways.
    """,
    agent=stockAnalystWriter,
    context=[getStockPrice, get_news],
)


# Cria uma equipe de agentes e define o processo de execução
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, get_news, writeAnalysis],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15,
)


# Interface de usuário com Streamlit para entrada e execução da pesquisa
with st.sidebar:
    st.header("Enter the Stock to Research")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        # Executa o processo da equipe com o ticket fornecido
        results = crew.kickoff(inputs={"ticket": topic})

        st.subheader("Results of your research:")
        st.write(results["final_output"])
