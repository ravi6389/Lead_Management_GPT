import joblib
import streamlit as st
import pandas as pd

import pandas as pd
from io import StringIO
import io

import os
import ssl
import certifi
import uuid
import sys

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import pandas as pd
import os
import json
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import Tool

from langchain.chains import LLMChain
# from langchain_ollama import ChatOllama
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI

import re
from datetime import datetime
from typing import Tuple
from openai import OpenAI  # or AzureOpenAI, or use LangChain LLM interface
import pandas as pd

from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent,AgentType
from langchain_groq import ChatGroq


distinct_values = {}

flag_column = 0


df_activities2 = pd.DataFrame()
df_final = pd.DataFrame()
df_contact = pd.DataFrame()
df_all_oppts = pd.DataFrame()
df = pd.DataFrame()
st.set_page_config(page_title="FM AI Agent", layout="centered")



if 'flag' not in st.session_state:
    st.session_state['flag'] = 0

if 'df_final' not in st.session_state:
    st.session_state['df_final'] = pd.DataFrame()
else:
    
    df_final = st.session_state['df_final'].copy(deep = True)


if 'account_selected' not in st.session_state:
    st.session_state['account_selected'] = 0

if 'df_activity_date_latest' not in st.session_state:
    st.session_state['df_activity_date_latest'] = 0

if 'df_all_contacts' not in st.session_state:
    st.session_state['df_all_contacts'] = 0

# if(st.session_state['flag'] ==0):




# Create a function to execute and cache the queries


# schema = "SFDC"
df_grouped = ''
df_activities = pd.DataFrame()
df_oppt_details = pd.DataFrame()
# os.environ['SSL_CERT_FILE'] = 'C:\\Users\\RSPRASAD\\AppData\\Local\\.certifi\\cacert.pem'

GITHUB_API = "https://api.github.com"

# GROQ_API_KEY = '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'
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
llm = ChatGroq(temperature=0.8, groq_api_key=GROQ_API_KEY,
            #    model_name="llama-3.1-8b-instant", streaming=True)
            model_name="llama-3.3-70b-versatile", streaming=True)
            # model_name="meta-llama/llama-guard-4-12b", streaming = True)
            # model_name = 'groq/compound', streaming = True)
            # model_name = 'openai/gpt-oss-20b')
def load_model():
    return joblib.load('lead_scoring_model.pkl')


model = load_model()

categorical = [ 'opportunity_owner', 'company_name', 'customer_name',
       'customer_company',  'currency', 
       'lead_source', 'region',
       'product_line', 'industry', 
      'sales_channel',
       'campaign_name']

categorical = [col.lower() for col in categorical]
numerical = ['funnel',  'age_of_opportunity_days']

# numerical = ['Amount (converted)',  'Age', 'numOpptsAccount',\
#             'wonOpptsAccount', 'lostOpptsAccount']


# numerical = ['Amount (converted)', 'numOpptsAccount',\
#             'wonOpptsAccount', 'lostOpptsAccount']


numerical = [col.lower() for col in numerical]

feature_cols = categorical + numerical


prompt_template2= """
You are given the global DataFrame, 'df':


The user has asked to filter this DataFrame based on the following query: "{query}".

Please write Python code that filters the DataFrame 'df_oppts' based on the user's request.
E.g. which optps are stuck for 'United States'
Put filter on the column 'country' having 'United States'.
"""

# def generate_filter_code(query: str):
#     # Convert the DataFrame to string for LLM input
    

#     # Generate Python code to filter DataFrame based on the user's query
#     prompt = PromptTemplate(input_variables=["query"], template=prompt_template2)
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     generated_code = llm_chain.invoke(query=query)
#     st.code(generated_code)
#     return generated_code

def execute_filter_code(query):
    # Prepare the environment for exec to safely execute the code
    query = st.session_state['prompt']
    global df_activities2
    global df_activities
    st.write(f' I am in execute filter code and query is.. {query}')
    prompt = PromptTemplate(input_variables=["query"], template=prompt_template2)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # generated_code = llm_chain.invoke(query=query)
    # st.code(generated_code)
    # return generated_code
    try:
                
        # Run the LLMChain to generate the Python script based on the question and CSV columns
        python_script = llm_chain.invoke({
            "query": query,
            
            
        })

        # Display the generated Python code
        # st.write(df.head())
        # st.write("### 📝 Generated Python Code:")
        python_script['text'] = python_script['text'].strip('`').replace('python', '')
        st.code(python_script['text'], language='python')
        # st.write('Executing the code to give the answers')
        # Option to execute the generated Python code
        # if st.button("▶️ Run Code"):
        try:
        
            import matplotlib.pyplot as plt
            

            exec_locals = {}
            exec_globals = {"df": df_activities2, "pd": pd}
            # st.write(python_script)
            # st.write(python_script['text'].strip('`').replace('python', ''))

            python_script['text'] = python_script['text'].strip('`').replace('python', '')
            exec_globals = globals().copy()  # Start with all existing globals

            # Append additional required objects
            exec_globals.update({
                "plt": plt
                
            })
            
            exec(python_script['text'], exec_globals, exec_locals)

            # st.write(exec(python_script['text'], {'df':df, 'pd':pd}))
        
            
            # If a result variable is present, display it
            if 'result' in exec_locals:
                st.write("### 📊 Result:")
                # st.write('After filtering for timeline given by user,  activities table data is as below')
            
                df_activities = exec_locals['result'].copy(deep = True)
                st.session_state['df_activities'] = df_activities.copy(deep = True)
                # st.write(exec_locals['result'])
                # st.write(df_activities())
                return((exec_locals['result']))
               
            
                
            else:
                st.warning("⚠️ The code did not produce a 'result' variable.")
                st.stop()
        
        except Exception as e:
            st.error(f"🚫 Error running the code 1: {e}")
            st.stop()
            # attempt += 1
            # return ('Error')
    except Exception as e:
        st.error(f"🚫 Error generating the code 2: {e}")
        st.stop()
        # attempt += 1
        # return('Error')





# def find_matching_columns(keyword, df):
#     """Find columns where the keyword appears (case-insensitive)."""
#     keyword= keyword.lower()
#     pattern = rf'\b{keyword}\b'
    
    
#     # df = st.session_state['df_oppts']
#     accepted_cols = ['2024 legal entity 2', '2024 legal entity 3','2024 legal entity 4','obi group'\
#                      ,'obi sub group','secondary sub-group', 'product line code', 'opportunity owner email']
    
#     for col in df.columns:

#         # if col in df.columns:
 
#         # First check: keyword in column name
#         if (col in accepted_cols):
#             col_values = df[col].dropna().astype(str)
#             # if col in df.columns:
#             if col_values.str.contains(pattern, case=False, na=False, regex=True).any():
#             # if col_values.str.lower().eq(keyword.lower()).any():
                
#                 st.write(f'Found the match and match is..{col}')
#                 return col  # Return only the first match based on hierarchy

#     return None  # No match found

column_hierarchy = [
 
  "opportunity_owner",
"company_name",
"customer_name",
"customer_company",
"funnel",
"currency",
"close_date",
"age_of_opportunity_days",
"stage",
"lead_source",
"region",
"product_line",
"industry",
"probability_to_close",
"created_date",
"last_contact_date",
"next_followup_date",
"priority",
"sales_channel",
"campaign_name"

]

list_matching_cols = [] 
def find_matching_columns(keyword, keywords):
    """Find columns where the keyword appears (case-insensitive)."""

    global df  # if df is defined globally
    global list_matching_cols
    global cols_to_group
    keyword_lower = keyword.lower()
    global flag_column 
    global df
    global user_specific_value 

    for col in df.columns: 
        
        if keyword_lower == col.lower():
            for keyword in keywords:
                if df[col].astype(str).str.contains(keyword, case=False, na=False).any():
                    st.write(f'Applying the user given filter of {keyword} in column of {col}')
                    mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
                    df = df[mask].copy(deep = True)
                    user_specific_value = 1
                    flag_column = 0
                    # return col
            
            st.write('Exact match for column has been found')
            st.write(f'Col is {col}')
            st.write('Type of col is', str(type(col)))
            cols_to_group = [col]
            st.write(f'Cols to group is {cols_to_group}')
            list_matching_cols = [col]
            st.write(f'list_matching_cols is {list_matching_cols}')
            flag_column = 1
            return col

    for col in column_hierarchy:

        # if col in df.columns:
 
        # First check: keyword in column name
    
        flag_column = 0
        if keyword_lower in col.lower():
            cols_to_group = list(col)
            flag_column = 1
            return col
            
        # if col in df.columns:
        if df[col].astype(str).str.contains(keyword, case=False, na=False).any():
            return col  # Return only the first match based on hierarchy

    return None  # No match found

def extract_keywords(query):
    """Extract words in single or double quotes."""
    raw_matches = re.findall(r"'(.*?)'|\"(.*?)\"", query)
    # Flatten the list of tuples and remove empty strings
    return [kw.strip().lower() for tup in raw_matches for kw in tup if kw]

# def apply_filters(query, df):
#     query = st.session_state['prompt']
#     st.write(f'query is {query}' )
#     keywords = extract_keywords(query)
#     # keywords = eval(identify_KWs(query))
#     # df = st.session_state['df_oppts']
    
#     st.write('I am in apply filters and KWs are..')
    
#     st.write(keywords)
#     list_matching_cols = []
#     st.write('data')
#     for keyword in keywords:
#         st.write(f'Keyword is..{keyword}')
#         matching_col = find_matching_columns(keyword, df)
#         list_matching_cols.append(matching_col)
#         # st.write(list_matching_cols)
#         # st.write(matching_col)
#         if not matching_col:
#             print(f"❌ No matching columns found for keyword: '{keyword}'")
#             return 0  # Empty DF if any keyword is unmatched

        
#         # mask = st.session_state['df_oppts'][matching_col].astype(str).str.contains(keyword, case=False, na=False)
#         # st.write('Applying filter of ', matching_col, 'in df_oppt_details of st.session_state')
#         # st.session_state['df_oppts'] = st.session_state['df_oppts'][mask]
        
#         # st.write(st.session_state['df_oppts'])

#         mask = df[matching_col].astype(str).str.contains(keyword, case=False, na=False)
#         st.write('Applying filter of ', matching_col, 'in df')
#         df = df[mask]
#         st.write('After applying filter')
#         st.write(df)
    
#     return df
        
def apply_filters(query, df):
    
    keywords = extract_keywords(query)
    # st.write('df in apply filters is..', df)
    
    
    st.write('I am in apply filters..')
    st.write(keywords)
    for keyword in keywords:
        global list_matching_cols
        matching_col = find_matching_columns(keyword, keywords)
        list_matching_cols.append(matching_col)
        st.write(list_matching_cols)
        st.write(matching_col)
        if not matching_col:
            print(f"❌ No matching columns found for keyword: '{keyword}'")
            return 0  # Empty DF if any keyword is unmatched

        if flag_column ==0:
            mask = df[matching_col].astype(str).str.contains(keyword, case=False, na=False)
            st.write('Applying filter of ', matching_col, 'in df')
            df = df[mask].copy(deep = True)
            # st.write(df)
        # st.write(df.head())
        # st.write('flag_column is..', flag_column)
    return df

# Start year wise performance
@st.cache_data
def load_data():
    df2 = pd.read_excel('abc_leads_dataset.xlsx')
    df2 = df2.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

    st.write(df2.head())
    return df2

def FM_Misc_fn(query: str) -> str:
    


    """
    Perform analysis on the global dataframe `df` 
    supplied and writes and executes Python code to give answers for monthwise, yearwise and leadsource wise performance of an opportunity owner or an email in terms of funnel
    creation.

    Args:
        query (str): The search query.
        

    Returns:
        str: Result of the code execution (without extra commentary)
    """
    global df
    df = load_data()
    st.write('I am in FM_Misc tool')
    keywords = extract_keywords(st.session_state['prompt'])
   
    df = load_data()
    df['created_year'] = pd.to_datetime(df['created_date']).dt.year
    df['close_year'] = pd.to_datetime(df['close_date']).dt.year

    df['created_month'] = pd.to_datetime(df['created_date']).dt.month
    df['close_month'] = pd.to_datetime(df['close_date']).dt.month

    df['created_year'] = df['created_year'].astype(str)
    df['close_year'] = df['close_year'].astype(str)
    
    st.write(keywords)





    global distinct_values

    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        # Keep only short, meaningful values (optional)
        unique_vals = [v for v in unique_vals]
        distinct_values[col] = unique_vals[:5]

    
    prompt_template = """
You are an expert in writing Python code and executing it.

Question is: "{question}".
You have a global dataframe 'df' already. 
Dont ever load any data in 'df' and dont ever write something like
df = pd.read_csv('...')

STRICT REQUIREMENTS:
*** 
Never ever change the case of any column in global dataframe 'df'. 
All the column names are in lower case already — keep them as is.
Just return the code with no descrption or comments.
Always include import streamlit as st in your code.
***

ADDITIONAL RULES FOR DATA TYPES:
- For integer or float columns (like 'funnel'), NEVER compare them to quoted strings.
- For varchar columns (like 'creaated_year'),  compare them to quoted strings.

  Example: use (df['created_year'] == '2025')
- For string columns, always compare in lowercase using `.str.lower()`.
  Example: (df['region'].str.lower() == 'north america').
- Always respect the data types listed below. Do not assume or change them.
- If you are grouping by a single column, .unstack() should not be used. Only use .unstack() when there’s a multi-level groupby.

Below are the columns with their meaning and possible distinct values 
(only use a column if the keyword or phrase from the {question}
matches one of its values. Convert keyword in lower letters):

{distinct_values}
Column Name	Description /Data type/  Meaning
id": string - Unique identifier for each opportunity (e.g., OPP-1023).

"opportunity_owner": string - Name of the sales representative responsible for this opportunity.

"company_name": string - Division or branch of abc.com managing the lead (e.g., "XYZ Technologies (abc.com Europe)").

"customer_name": string - Name of the individual contact at the customer’s organization.

"customer_company": string - Name of the customer’s company or organization.

"amount_of_lead": integer - Potential deal value or revenue amount associated with this opportunity.

"currency": string - Currency of the deal amount (e.g., USD, EUR, INR, GBP, JPY).

"close_date": date - Expected or actual date when the opportunity is closed (won or lost).

"age_of_opportunity_days": integer - Number of days the opportunity has been active since creation.

"stage": string - Current stage in the sales funnel (e.g., Prospecting, Proposal, Negotiation, Closed Won, Closed Lost).

"lead_source": string - Origin of the lead (e.g., Web, Referral, Partner, Advertisement, Event, Email Campaign).

"region": string - Geographic region of the customer or opportunity (e.g., North America, Europe, Asia Pacific, Middle East, Latin America).

"product_line": string - Product category or offering involved (e.g., Software, Hardware, Services, Cloud, AI Solutions, Consulting).

"industry": string - Industry sector of the customer’s company (e.g., Manufacturing, Retail, Finance, Healthcare, Technology).

"probability_to_close": integer - Estimated likelihood (in percentage) that the deal will close successfully.

"created_date": date - Date the opportunity was created in the CRM system.

"last_contact_date": date - Most recent date when the sales rep contacted the customer.

"next_followup_date": date - Planned next contact or follow-up date for the opportunity.

"priority": string - Priority level assigned to the lead (High, Medium, Low).

"sales_channel": string - Channel through which the lead is being pursued (e.g., Direct, Partner, Online, Distributor).

"campaign_name": string - Marketing campaign or initiative that generated the lead (e.g., Q1 Drive, Summer Promo, Referral Boost).

"created_year": varchar- Year derived from created_date.

"created_month": integer - Month (1–12) derived from created_date.

"close_year": varchar - year derived from close_date.


INTERPRETATION RULES:
- If the question includes the words “sort”, “rank”, “order”, “highest”, or “top”, 
  then you should:
    1. Identify which numeric column is being referenced (e.g., funnel, amount_of_lead, probability_to_close).
    2. Group by the relevant category (e.g., opportunity_owner, region, industry, etc.).
    3. Aggregate the numeric column using sum() or mean(), as appropriate.
    4. Sort the result in descending order unless otherwise stated (e.g., "lowest" or "ascending").
    5. Assign the sorted dataframe to `result`.
    6. Always draw a bar chart showing the sorted results:
        fig, ax = plt.subplots()
        ax.bar(result.index, result['funnel'])
        ax.set_xlabel('opportunity_owner')
        ax.set_ylabel('funnel')
        ax.set_title('Sorted funnel by opportunity_owner (2024)')
        st.pyplot(fig)
        fig.savefig('chart.png', bbox_inches='tight')
        chart_path = 'chart.png'

ANALYSIS RULES:
- Ignore case differences in both dataframes and in user questions.
- Never add any new filters — all necessary filters are already applied in 'df_oppts_yearwise'.
- When performing a comparison, also compute and include the subtraction value in `result`.
- - Sort the months (if applicable) as '1','2', '3', '4','5','6','7','8','9','10','11',12'
- If you want a specific column (say, "year") to become the columns after .unstack(),
make sure "year" is the second (inner) key in your groupby:
e.g. df.groupby(['region', 'year'])['funnel'].sum().unstack()

or be explicit:


df.groupby(['created_year', 'region'])['funnel'].sum().unstack(level='created_year')
STRICT REQUIREMENTS:
***
- If the question only asks for a simple total, sum, count, or single numeric result 
  (e.g., "total funnel created", "how many leads", "sum of amount_of_lead", 
  "number of opportunities"), then:
    - Do NOT draw any chart.
    - Just compute the numeric result and assign it to `result`.
- Whenever the question includes words like "compare", "show", "graph", "trend", "difference", "split", "breakdown", "distribution", "vs", "over time", "by", or "across", ALWAYS draw a bar graph or line graph using matplotlib.
- Decide whether to draw horizontal bar graph or stacked bar graph.
- Use clear labels and titles and tilt lables in x axis by 90 degrees.
- Always use Streamlit syntax for plotting:
    fig, ax = plt.subplots()
    ... # plotting code
    st.pyplot(fig)
- NEVER use plt.show().
- If a comparison is requested, always include:
    1. The numeric result (in `result`)
    2. The visual chart (st.pyplot(fig))
    3. - Before plotting any chart that involves 'month', always sort the dataframe by the month number in ascending order. 
  Example:
      df['month'] = df['month'].astype(int)
      df = df.sort_values('month')

- After drawing the chart, save it as a temporary image file named 'chart.png' using:
    fig.savefig('chart.png', bbox_inches='tight')
- Assign both:
    1. The numeric or tabular output to a variable named `result`
    2. The chart image path to a variable named `chart_path = 'chart.png'`

OUTPUT FORMAT:
- Do not print anything else except valid Python code.
- Assign the final computed value to `result`.***

You are FunnelGPT — a sales and marketing funnel analysis expert.
"""


    template = PromptTemplate(
    input_variables=[ "question", "distinct_values"],
    template=prompt_template,
    )

    llm_chain = LLMChain(prompt=template, llm=llm)




    # ---------------------------
    # Streamlit App Interface
    # ---------------------------









    # Ask the user for a question about the data
    question = query
    # while True:
    if question:
        st.session_state['flag'] = 0
        # if len(st.session_state['df_final']) ==0 :
        with st.spinner("Generating Python code..."):
            attempt = 0
            global result
            while attempt <5 and True:
                try:
                
                    st.write(f'Attempt number is..{attempt}')
                    # Run the LLMChain to generate the Python script based on the question and CSV columns
                    python_script = llm_chain.invoke({
                        
                        "question": question,
                        "distinct_values":distinct_values
                        
                        
                    })

                    # Display the generated Python code
                    # st.write(df.head())
                    # st.write("### 📝 Generated Python Code:")
                    python_script['text'] = python_script['text'].strip('`').replace('python', '')
                    st.code(python_script['text'], language='python')
                    # st.write('Executing the code to give the answers')
                    # Option to execute the generated Python code
                    # if st.button("▶️ Run Code"):
                    try:
                    
                        import matplotlib.pyplot as plt
                        

                        exec_locals = {}
                      
                        exec_globals = {"df": df, "pd": pd}
          

                        python_script['text'] = python_script['text'].strip('`').replace('python', '')
                        # exec_globals = globals().copy()  # Start with all existing globals

                        # Append additional required objects
                        exec_globals.update({
                            "plt": plt
                            
                        })
                        
                        exec(python_script['text'], exec_globals, exec_locals)

                        # st.write(exec(python_script['text'], {'df':df, 'pd':pd}))
                    
                    
                        # If a result variable is present, display it
                        if 'result' in exec_locals:
                            st.write("### 📊 Result:")
                         
                            st.write(exec_locals['result'])
                            st.stop()
                            # if 'chart_path' in exec_locals:
                            #     import os
                            #     if os.path.exists('chart.png'):
                            #         chart_path = 'chart.png'
                            #     else:
                            #         chart_path = None

                            #     with open(chart_path, "rb") as f:
                            #         import base64
                            #         img_base64 = base64.b64encode(f.read()).decode("utf-8")

                            #     message = HumanMessage(content=[
                            #     {"type": "text", "text": "Please analyze this graph and describe the key trends of column of 'stage label'."},
                            #     {
                            #         "type": "image_url",
                            #         "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            #     }
                            # ])
                           
                            #     response = vision_llm([message])
                            #     st.write("📊 Graph Analysis:")
                            #     st.write(response.content)
                                
                               

                                
                               
                              
                            #     st.stop()
                            # st.stop()
                        else:
                            st.warning("⚠️ The code did not produce a 'result' variable.")
                            attempt += 1
                            # st.stop()
                    except Exception as e:
                        st.error(f"🚫 Error running the code 1: {e}")
                        # st.stop()
                        attempt += 1
                        # return ('Error')
                except Exception as e:
                    st.error(f"🚫 Error generating the code 2: {e}")
                    attempt += 1
                        # return('Error')


FM_Misc_Tool_obj= Tool(
    name="FM_Misc_Tool",
    func= FM_Misc_fn,
    description="A tool for answering questions related to funnel. DO NOT USE IT FOR FINDING PROBABILITY OF OPPTS TO BE WON"
)

def FM_Prediction_fn(query: str) -> str:
    


    """
    Perform analysis on the global dataframe `df` 
    and finds probaility of open opportunities using a pkl file.
    Args:
        query (str): The search query.
        

    Returns:
        str: Result of the code execution (without extra commentary)
    """
    global df
    df = load_data()
    st.write('I am in FM_Prediction tool')
    keywords = extract_keywords(st.session_state['prompt'])
   
    df = load_data()
    
    df = df[~df['stage'].isin(['Closed Won', 'Closed Lost'])].reset_index(drop = True)
    df['created_year'] = pd.to_datetime(df['created_date']).dt.year
    df['close_year'] = pd.to_datetime(df['close_date']).dt.year

    df['created_month'] = pd.to_datetime(df['created_date']).dt.month
    df['close_month'] = pd.to_datetime(df['close_date']).dt.month

    df['created_year'] = df['created_year'].astype(str)
    df['close_year'] = df['close_year'].astype(str)
    
    st.write(keywords)





    global distinct_values

    for col in df.columns:
        unique_vals = df[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        # Keep only short, meaningful values (optional)
        unique_vals = [v for v in unique_vals]
        distinct_values[col] = unique_vals[:5]

    
    prompt_template = """
You are an expert in writing Python code and executing it. 

Question is: "{question}".
You have a global dataframe 'df' already. 
Dont ever load any data in 'df' and dont ever write something like
df = pd.read_csv('...')

STRICT REQUIREMENTS:
*** 
Never ever change the case of any column in global dataframe 'df'. 
All the column names are in lower case already — keep them as is.
Just return the code with no descrption or comments.
Always include import streamlit as st in your code.
***

ADDITIONAL RULES FOR DATA TYPES:
- For integer or float columns (like 'funnel'), NEVER compare them to quoted strings.
- For varchar columns (like 'creaated_year'),  compare them to quoted strings.

  Example: use (df['created_year'] == '2025')
- For string columns, always compare in lowercase using `.str.lower()`.
  Example: (df['region'].str.lower() == 'north america').
- Always respect the data types listed below. Do not assume or change them.

Below are the columns with their meaning and possible distinct values 
(only use a column if the keyword or phrase from the {question}
matches one of its values. Convert keyword in lower letters):

{distinct_values}
Column Name	Description /Data type/  Meaning
id": string - Unique identifier for each opportunity (e.g., OPP-1023).

"opportunity_owner": string - Name of the sales representative responsible for this opportunity.

"company_name": string - Division or branch of abc.com managing the lead (e.g., "XYZ Technologies (abc.com Europe)").

"customer_name": string - Name of the individual contact at the customer’s organization.

"customer_company": string - Name of the customer’s company or organization.

"amount_of_lead": integer - Potential deal value or revenue amount associated with this opportunity.

"currency": string - Currency of the deal amount (e.g., USD, EUR, INR, GBP, JPY).

"close_date": date - Expected or actual date when the opportunity is closed (won or lost).

"age_of_opportunity_days": integer - Number of days the opportunity has been active since creation.

"stage": string - Current stage in the sales funnel (e.g., Prospecting, Proposal, Negotiation, Closed Won, Closed Lost).

"lead_source": string - Origin of the lead (e.g., Web, Referral, Partner, Advertisement, Event, Email Campaign).

"region": string - Geographic region of the customer or opportunity (e.g., North America, Europe, Asia Pacific, Middle East, Latin America).

"product_line": string - Product category or offering involved (e.g., Software, Hardware, Services, Cloud, AI Solutions, Consulting).

"industry": string - Industry sector of the customer’s company (e.g., Manufacturing, Retail, Finance, Healthcare, Technology).

"probability_to_close": integer - Estimated likelihood (in percentage) that the deal will close successfully.

"created_date": date - Date the opportunity was created in the CRM system.

"last_contact_date": date - Most recent date when the sales rep contacted the customer.

"next_followup_date": date - Planned next contact or follow-up date for the opportunity.

"priority": string - Priority level assigned to the lead (High, Medium, Low).

"sales_channel": string - Channel through which the lead is being pursued (e.g., Direct, Partner, Online, Distributor).

"campaign_name": string - Marketing campaign or initiative that generated the lead (e.g., Q1 Drive, Summer Promo, Referral Boost).

"created_year": varchar- Year derived from created_date.

"created_month": integer - Month (1–12) derived from created_date.

"close_year": varchar - year derived from close_date.



STRICT REQUIREMENTS:
***
- Your ONLY task is to filter the dataframe `df` based on the user's question.
- NEVER compute counts, ratios, percentages, or probabilities.
- NEVER perform arithmetic operations or aggregations like sum(), mean(), len(), count(), or groupby().
- NEVER return a single number, percentage, or float.
- ONLY return a subset of `df` that matches the user's intent.
- The filtered dataframe must always be assigned to a variable named `result`.
- Always include "import streamlit as st" in the code.
- Do not print or describe anything — only return pure Python code.

Example:
✅ Correct:
result = df[(df['region'].str.lower() == 'north america') ]

❌ Wrong:
result = len(df[...]) / len(df[...])
❌ Wrong:
result = df['amount'].sum()
***

OUTPUT FORMAT:
- Return only valid Python code.
- The final line must assign the filtered dataframe to `result`.
- Example final line:
  result = filtered_df

"""


    template = PromptTemplate(
    input_variables=[ "question", "distinct_values"],
    template=prompt_template,
    )

    llm_chain = LLMChain(prompt=template, llm=llm)




    # ---------------------------
    # Streamlit App Interface
    # ---------------------------









    # Ask the user for a question about the data
    question = query
    # while True:
    if question:
        st.session_state['flag'] = 0
        # if len(st.session_state['df_final']) ==0 :
        with st.spinner("Generating Python code..."):
            attempt = 0
            global result
            while attempt <5 and True:
                try:
                
                    st.write(f'Attempt number is..{attempt}')
                    # Run the LLMChain to generate the Python script based on the question and CSV columns
                    python_script = llm_chain.invoke({
                        
                        "question": question,
                        "distinct_values":distinct_values
                        
                        
                    })

                    # Display the generated Python code
                    # st.write(df.head())
                    # st.write("### 📝 Generated Python Code:")
                    python_script['text'] = python_script['text'].strip('`').replace('python', '')
                    st.code(python_script['text'], language='python')
                    # st.write('Executing the code to give the answers')
                    # Option to execute the generated Python code
                    # if st.button("▶️ Run Code"):
                    try:
                    
                        import matplotlib.pyplot as plt
                        

                        exec_locals = {}
                      
                        exec_globals = {"df": df, "pd": pd}
          

                        python_script['text'] = python_script['text'].strip('`').replace('python', '')
                        # exec_globals = globals().copy()  # Start with all existing globals

                        # Append additional required objects
                        exec_globals.update({
                            "plt": plt
                            
                        })
                        
                        exec(python_script['text'], exec_globals, exec_locals)

                        # st.write(exec(python_script['text'], {'df':df, 'pd':pd}))
                    
                    
                        # If a result variable is present, display it
                        if 'result' in exec_locals:
                            st.write("### 📊 Result:")

                            st.write(exec_locals['result'])
                            df = exec_locals['result']
                     
                        
                            df = df.loc[:, ~df.columns.duplicated()]

                            X = df[feature_cols]

                            # st.write(X)

                            df['prediction'] = model.predict_proba(X)[:, 1].round(2)
                            # st.session_state['df_final'] = df


                            df = df[['prediction'] + [c for c in df.columns if c != 'prediction']]
                            st.write('Prediction of oppts based on past data is...')
                            st.write(df)
                            st.stop()

                    
                        else:
                            st.warning("⚠️ The code did not produce a 'result' variable.")
                            attempt += 1
                            # st.stop()
                    except Exception as e:
                        st.error(f"🚫 Error running the code 1: {e}")
                        # st.stop()
                        attempt += 1
                        # return ('Error')
                except Exception as e:
                    st.error(f"🚫 Error generating the code 2: {e}")
                    attempt += 1
                        # return('Error')


FM_Prediction_Tool_obj= Tool(
    name="FM_Prediction_Tool",
    func= FM_Prediction_fn,
    description="A tool for answering questions related to probability of open oppts to be won"
)



st.write('I am in bot part...')

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot that can do FM analysis"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt:=st.chat_input(placeholder="What is machine learning?"):

    


    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    
    st.session_state['prompt'] = prompt
    # search=DDGS(verify = False).text(prompt, max_results=10) 
    # llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Llama3-8b-8192",streaming=True)
    
    # tools=[KFT_finding_tool, code_tool]

    # tools=[filter_tool,KFT_finding_tool, FM_Yearwise_Tool_obj,]
    FM_tools = [FM_Misc_Tool_obj, FM_Prediction_Tool_obj]
    # FM_tools = [FM_Yearwise_tool]
    


    search_agent=initialize_agent(FM_tools,llm,
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                handle_parsing_errors=True, verbose = True)
    

    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=search_agent,
        tools=FM_tools,
        max_iterations=40,      # ← increase this
        max_execution_time=120,  # ← or this (in seconds)
    )

    with st.chat_message("assistant"):
        # st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        # response=search_agent.run(st.session_state.messages)
    
        # st.write(st.session_state['prompt'])
        # get_data()
        # response=search_agent.run(st.session_state['prompt'],callbacks=[st_cb])
        response=search_agent.run(st.session_state['prompt'])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
        # st.write(matched_cols)








