import logging
import pyodbc
import decimal  # Add this line
import openai
import json
from decimal import Decimal
from datetime import datetime
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain.prompts import FewShotPromptTemplate
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px  # For interactive visualizations
import base64
import speech_recognition as sr  # For voice input
import requests
import logging
from io import StringIO
import folium 
from streamlit_folium import st_folium
 
logging.basicConfig(level=logging.INFO)
 
# Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_key = "cfe7825c59344377b398ed9ec2b984eb"
openai.api_base = "https://mt-openaipractice.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
 
 
 
def get_image_base64(image_path):
    """Convert image to Base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Image not found at {image_path}. Please check the path.")
        return None
#-------------------------------------
 
#-----------------------------------------------------------------
def send_email_alert_with_summary(summary):
    """Send email alerts with analysis summary."""
    try:
        logging.info(f"Sending email with summary: {summary}")
       
        if not summary:
            logging.error("Summary is empty, cannot send email.")
            return False
       
        email_content = f"""
        <html>
            <body>
                <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <h2 style="color: #4CAF50;">Greeting!!,</h2>
                    <p>We have completed analyzing your data. Here is the summary:</p>
                    <blockquote style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #4CAF50;">
                        {summary}
                    </blockquote>
                    <p>If you have any questions or need further assistance, please let us know.</p>
                    <p>Best regards,</p>
                    <p><strong>Data Analysis Team</strong></p>
                </div>
            </body>
        </html>
 
        """
       
        response = requests.post(
            "https://prod-29.centralindia.logic.azure.com:443/workflows/4355a94b62bc4b35ae540c1dea1aa7f1/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=UpLHLBHhdxNtO-pYvtfxsyrkpWBN71gZvSKSgWnpfCo",
            json={
                "to": "venkatama.in@mouritech.com,saisanthoshj.in@mouritech.com,praveenkar.in@mouritech.com,swapnaa.in@mouritech.com",  # Change this to the recipient's email
                "subject": "Data Analysis Summary and Insights",
                "message": email_content,
                "body" : email_content,
               
            },
            headers={"Content-Type": "application/json"},
        )
 
        if response.status_code in [200, 202]:
            logging.info("Email sent successfully.")
            return True
        else:
            logging.error(f"Failed to send email: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return False
 
#-----------------------------------------------------------
 
def detect_low_sales(data, threshold=10):
    """
    Detect items with sales below a given threshold dynamically.
    Handles columns with numeric values stored as strings by converting them.
   
    :param data: DataFrame with sales data.
    :param threshold: Minimum sales value to consider.
    :return: DataFrame with low sales items.
    """
    # Check if the DataFrame is empty
    if data.empty:
        st.error("The uploaded data is empty.")
        return pd.DataFrame()
 
    # Attempt to convert all columns to numeric where possible
    for column in data.columns:
        if data[column].dtype == "object":  # Check if column is of type string
            try:
                data[column] = pd.to_numeric(data[column], errors="ignore")
            except Exception as e:
                st.warning(f"Could not convert column '{column}' to numeric: {e}")
 
    # Identify numeric columns (for threshold comparison)
    numeric_columns = data.select_dtypes(include=["number"]).columns
    if numeric_columns.empty:
        st.error("No numeric columns found in the data.")
        return pd.DataFrame()
 
    # Select the first numeric column for threshold filtering
    sales_column = numeric_columns[0]
 
    # Identify potential descriptive columns (non-numeric)
    descriptive_columns = data.select_dtypes(exclude=["number"]).columns
    if descriptive_columns.empty:
        st.warning("No descriptive columns found. Returning results with numeric data only.")
        descriptive_columns = []
 
    # Filter for low sales
    low_sales = data[data[sales_column] < threshold]
 
    # Retain relevant columns in the result
    columns_to_display = list(descriptive_columns) + [sales_column]
    low_sales = low_sales[columns_to_display] if not low_sales.empty else pd.DataFrame()
 
    return low_sales
 
 
 
# Helper Functions
def serialize_result(query_result):
    """Serialize database query results for Streamlit."""
 
    def custom_serializer(obj):
        if isinstance(obj, Decimal):
            st.session_state.serialize_result = True
            return float(obj)
        elif isinstance(obj, datetime):
            st.session_state.serialize_result = True
            return obj.isoformat()
 
        raise TypeError(f"Type {type(obj)} not serializable")
 
    return json.dumps(query_result, default=custom_serializer)
 
 
def get_sql_query_from_azure_openai(user_input: str, deployment_name: str) -> str:
    """Generate an SQL query using Azure OpenAI."""
 
    try:
        response = openai.ChatCompletion.create(
            deployment_id="openai-demo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SQL assistant specialized in Microsoft SQL Server. "
                        "Include a feature that generates natural language summaries or insights based on the retrieved data. For example: Sales increased by 15% last month compared to the previous month and Top-selling product categories are Electronics and Home Appliances. Use tools like GPT or Google PaLM to generate these narratives.\n"
                        "The database contains the following tables:\n"
                        "1. SalesRecords: Columns [SaleID, CstID, PrdtID, SaleDate, Quantity, UnitPrice, TotalAmount, DiscountApplied, TaxAmount, NetAmount, PaymentMethod, SalesRepresentativeID, Region]\n"
                        "2. Products: Columns [ProductID, ProductName, Category, UntPrice, StockQuantity, SupplierID]\n"
                        "3. Customers: Columns [CustomerID, CustomerName, Email, Phone, Address, RegisteredDate]\n\n"
                        "Generate MSSQL-compatible queries based on the user's input. "
                        "Do not include explanations or code block syntax."
                        "always generate same query for the user_input until it changed"
                        "Do not perform insert , update , delete and drop operation on columns and tables"
                       # "Mask the  Customers columns: Email, Phone"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write an SQL query for the following request: {user_input}",
                },
            ],
            temperature=0.5,
        )
 
        generated_query = response["choices"][0]["message"]["content"].strip()
 
        # Remove any non-SQL text or code block delimiters
        if "```" in generated_query:
            generated_query = generated_query.split("```")[1]  # Extract the SQL block
        return generated_query.strip()
    except Exception as e:
        logging.error(f"Azure OpenAI API error: {e}")
        return None
 
def create_contact_map():
    # Example location (use the actual office address or any other location)
    latitude = 33.592468  # Example latitude (New York)
    longitude = -101.896217  # Example longitude (New York)
 
     # Create a Folium map centered at MOURI Tech's location
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
 
    # Add a marker with a location pointer at the office
    folium.Marker(
        location=[latitude, longitude],
        popup="MOURI Tech, Irving, TX",  # Text that shows when the marker is clicked
        icon=folium.Icon(color='blue', icon='info-sign')  # Custom blue icon for better visibility
    ).add_to(m)
 
    # Add a location pointer (marker) at the office location
    folium.CircleMarker(
        location=[latitude, longitude],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.6
    ).add_to(m)
   
 
     # Address information to display beside the map
    address = """
    **MOURI Tech Limited**  
    400 E. Las Colinas Blvd, Suite 160,  
    Irving, TX 75039, USA  
    Phone: +1-800-555-1234  
    Email: contact@mouritech.com
    """
 
    # Layout: Create two columns - one for the map and one for the address
    col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed
 
    # Place the map in the first column
    with col1:
         st_folium(m, width=600, height=200)
 
    # Place the address in the second column
    with col2:
        st.markdown(address)
 

@st.cache_data
def execute_query_on_mssql(sql_query: str):
    """Execute an SQL query on MSSQL and return a DataFrame."""
    try:
        with pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=db-sipef.database.windows.net;"
            "DATABASE=UNRWA;"
            "UID=dev1;"
            "PWD=Developer@1;"
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                columns = (
                    [column[0] for column in cursor.description]
                    if cursor.description
                    else []
                )
                df = (
                    pd.DataFrame.from_records(rows, columns=columns)
                    if rows
                    else pd.DataFrame()
                )
                
                # Convert decimal.Decimal columns to float
                for col in df.select_dtypes(include=['object']).columns:
                    if isinstance(df[col].iloc[0], decimal.Decimal):  # Check for Decimal type
                        df[col] = df[col].astype(float)

                return df

    except pyodbc.Error as e:
        logging.error(f"Database error: {e}")
        raise
 
 
def analyze_and_plot_data(data):
    """Analyze and plot data based on user input."""
    if 'x_axis' not in st.session_state or 'y_axis' not in st.session_state:
        st.session_state.x_axis = data.columns[0]  # Default to first column
        st.session_state.y_axis = data.columns[1]  # Default to second column
 
    # Dropdowns for selecting columns for x-axis and y-axis
    x_axis = st.selectbox("Select column for X-axis", data.columns, index=data.columns.get_loc(st.session_state.x_axis))
    y_axis = st.selectbox("Select column for Y-axis", data.columns, index=data.columns.get_loc(st.session_state.y_axis))
 
    # Update session state with selections
    st.session_state.x_axis = x_axis
    st.session_state.y_axis = y_axis
 
    # Plot the graph using Plotly
    fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(fig)
 
# Speech Recognition for Voice Input
def get_voice_input():
    """Capture voice input from the user."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
 
    try:
        with microphone as source:
            st.info("Listening for your query...")
            audio = recognizer.listen(source)
            query = recognizer.recognize_google(audio)
            st.success(f"Voice Input: {query}")
            return query
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
    except sr.RequestError:
        st.error("Sorry, the speech recognition service is unavailable.")
    return None
 
def analyze_and_generate_suggestions(data, deployment_name="openai-demo"):
    """
    Analyze the provided data and generate business suggestions.
    :param data: DataFrame containing any dataset.
    :param deployment_name: Azure OpenAI deployment name.
    :return: A string containing the analysis summary and actionable suggestions.
    """
    # Summarize the data dynamically
    summary = ""
    numeric_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.select_dtypes(include="object").columns

    if not data.empty:
        summary += f"Dataset contains {len(data)} rows and {len(data.columns)} columns.\n\n"
        
        # Analyze numeric columns
        if not numeric_columns.empty:
            summary += "Key insights from numeric columns:\n"
            for col in numeric_columns:
                summary += (
                    f"- {col}: mean={data[col].mean():.2f}, "
                    f"median={data[col].median():.2f}, "
                    f"max={data[col].max():.2f}, "
                    f"min={data[col].min():.2f}\n"
                )
            summary += "\n"

        # Analyze categorical columns
        if not categorical_columns.empty:
            summary += "Key insights from categorical columns:\n"
            for col in categorical_columns:
                try:
                    top_values = data[col].value_counts().head(3)
                    if not top_values.empty:
                        summary += (
                            f"- {col}: top categories are {', '.join(top_values.index)} "
                            f"with counts {', '.join(map(str, top_values.values))}\n"
                        )
                    else:
                        summary += f"- {col}: No data or all values are unique.\n"
                except Exception as e:
                    summary += f"- {col}: Could not analyze due to an error: {str(e)}\n"
            summary += "\n"

    # If data is empty, return a message
    else:
        summary = "The provided dataset is empty. No analysis was performed."

    # Generate suggestions using OpenAI
    prompt = (
        f"Based on the following dataset summary:\n{summary}\n\n"
        "Provide actionable business insights focusing on marketing strategies, operational improvements, and "
        "opportunities for growth. insights should be relevant to the data trends and it should be brief.Please make it compatible to email body without signature and body"
    )

    try:
        response = openai.ChatCompletion.create(
            deployment_id=deployment_name,
            messages=[
                {"role": "system", "content": "You are a business analyst providing actionable insights."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        suggestions = response["choices"][0]["message"]["content"].strip()
        return f"{summary}\n\nSuggestions:\n{suggestions}"
    except Exception as e:
        logging.error(f"Error generating business suggestions: {e}")
        return f"{summary}\n\nError generating suggestions. Please check your configuration or input data."

 
 
# Display Base64 Image (useful for showing logos, charts, etc.)
def display_base64_image(image_base64):
    """Display an image from a Base64 string in Streamlit."""
    try:
        image_data = base64.b64decode(image_base64)
        st.image(image_data, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")
 
 
def capture_voice_input():
    """Capture voice input and return as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            st.info("Processing your voice input...")
            voice_input = recognizer.recognize_google(audio)
            return voice_input
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand your voice. Please try again.")
        except sr.RequestError as e:
            st.error(f"Voice recognition error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    return ""
 
# Main User Interface for Streamlit
def main():
    st.set_page_config(page_title="AI SQL Visualizer", page_icon="🤖", layout="wide")
    uploaded_image_path = "C:/Users/praveenkar/OneDrive - MOURI Tech/Desktop/Hackathon/AI_Hackathon_4.jpg"
    image_base64 = get_image_base64(uploaded_image_path)
    if image_base64:
        st.markdown(
            f"""
            <style>
            .center-logo {{
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: 10px;  /* Add space from the top of the page */
                width: 200px;  /* Resize the image */
            }}
            </style>
            <img src="data:image/png;base64,{image_base64}" class="center-logo">
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Could not load the logo image. Please verify the file path.")

    # Initialize session state variables
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "query_result" not in st.session_state:
        st.session_state.query_result = pd.DataFrame()
    if "executed_query" not in st.session_state:
        st.session_state.executed_query = ""  # Store the executed query

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a section:")
    nav_options = ["Introduction", "SQL Query Executor",  "Contact Us"]
    selected_nav = st.sidebar.radio("", nav_options)

    if selected_nav == "Introduction":
        st.title("Welcome to the AI Hackathon 4.0")
        st.markdown(
            """
            **Automate & Innovate Hackathon**  
            Embrace the **AI-First Mindset** and lead the front in innovation!
            """
        )
        st.image(
            "https://artificialpaintings.com/wp-content/uploads/2024/06/565_the_role_of_AI_in_promoting_innovation.webp",
            caption="AI Innovation",
            use_container_width=True,
        )

    elif selected_nav == "SQL Query Executor":
        st.title("AI-Powered SQL Query Executor and Visualizer")

        input_mode = st.radio("Select Input Method:", ("Text Input", "Voice Input"))

        if input_mode == "Text Input":
            st.session_state.user_input = st.text_area(
                "Enter your query (e.g., 'Show top 10 sales regions'):",
                placeholder="Describe your request...",
            )

        elif input_mode == "Voice Input":
            st.info("Click the button to record your query.")
            if st.button("Capture Voice Input"):
                captured_input = capture_voice_input()  # Call the function to capture voice input
                if captured_input:
                    st.session_state.user_input = captured_input
                    st.success(f"Captured Voice Input: {st.session_state.user_input}")
                else:
                    st.error("Failed to capture voice input. Please try again.")

        if st.button("Run Query"):
            if st.session_state.user_input.strip():
                try:
                    sql_query = get_sql_query_from_azure_openai(
                        st.session_state.user_input, deployment_name="openai-demo"
                    )
                    if sql_query:
                        st.session_state.executed_query = sql_query
                        query_result = execute_query_on_mssql(sql_query)
                        st.session_state.query_result = query_result

                        st.success(f"Executed SQL Query:\n{sql_query}")

                    else:
                        st.error("Failed to generate SQL query.")
                except Exception as e:
                    st.error(f"Error executing query: {e}")
            else:
                st.error("Query cannot be empty.")

        if not st.session_state.query_result.empty:
            st.code(st.session_state.executed_query, language='sql')            
            st.subheader("Query Results")
            st.dataframe(st.session_state.query_result)  # Only this call should remain for query results
            analysis_summary_from_openai = analyze_and_generate_suggestions(st.session_state.query_result)
            st.subheader("Graphical Representation")

            if not st.session_state.query_result.empty:
                all_columns = st.session_state.query_result.columns.tolist()
                numeric_columns = st.session_state.query_result.select_dtypes(include="number").columns.tolist()
                string_columns = st.session_state.query_result.select_dtypes(include="object").columns.tolist()

                # Initialize session state for dropdown selections
                if "x_axis_col" not in st.session_state:
                    st.session_state.x_axis_col = string_columns[0] if string_columns else all_columns[0]
                if "y_axis_col" not in st.session_state:
                    st.session_state.y_axis_col = numeric_columns[0] if numeric_columns else None

                # Ensure x_axis_col is valid
                if st.session_state.x_axis_col not in all_columns:
                    st.session_state.x_axis_col = all_columns[0]  # Set to the first available column if invalid

                # Dropdowns for user selection
                st.session_state.x_axis_col = st.selectbox(
                    "Select column for X-axis (categorical):",
                    all_columns,
                    index=all_columns.index(st.session_state.x_axis_col),
                )

                st.session_state.y_axis_col = st.selectbox(
                    "Select column for Y-axis (numeric):",
                    numeric_columns,
                    index=numeric_columns.index(st.session_state.y_axis_col)
                    if st.session_state.y_axis_col in numeric_columns
                    else 0,
                )

                # Add the chart type selection here
                chart_type = st.radio(
                    "Select chart type:",
                    ("Bar Chart", "Pie Chart", "Scatter Plot", "Line Chart")
                )

                if st.session_state.x_axis_col and st.session_state.y_axis_col:
                    chart_data = st.session_state.query_result[
                        [st.session_state.x_axis_col, st.session_state.y_axis_col]
                    ].dropna()
                    chart_data[st.session_state.x_axis_col] = chart_data[st.session_state.x_axis_col].astype(str)
                    chart_data[st.session_state.y_axis_col] = pd.to_numeric(
                        chart_data[st.session_state.y_axis_col], errors="coerce"
                    )
                    chart_data = chart_data.dropna(subset=[st.session_state.y_axis_col])

                    if not chart_data.empty:
                        try:
                            # Conditionally generate the chart based on the selected chart type
                            if chart_type == "Bar Chart":
                                fig = px.bar(
                                    chart_data,
                                    x=st.session_state.x_axis_col,
                                    y=st.session_state.y_axis_col,
                                    color=st.session_state.y_axis_col,
                                    title=f"{st.session_state.y_axis_col} vs {st.session_state.x_axis_col}",
                                    labels={
                                        st.session_state.x_axis_col: f"{st.session_state.x_axis_col} (X-Axis)",
                                        st.session_state.y_axis_col: f"{st.session_state.y_axis_col} (Y-Axis)",
                                    },
                                    hover_data={st.session_state.y_axis_col: ":.2f"},
                                )
                            elif chart_type == "Pie Chart":
                                # Pie chart logic
                                # Pie charts typically use categorical data for the segments
                                # Use the x-axis column for categorizing and y-axis for the values
                                pie_data = chart_data.groupby(st.session_state.x_axis_col).sum().reset_index()
                                fig = px.pie(
                                    pie_data,
                                    names=st.session_state.x_axis_col,
                                    values=st.session_state.y_axis_col,
                                    title=f"Pie Chart of {st.session_state.y_axis_col} by {st.session_state.x_axis_col}",
                                    labels={st.session_state.x_axis_col: f"{st.session_state.x_axis_col} (Categories)",
                                            st.session_state.y_axis_col: f"{st.session_state.y_axis_col} (Values)"}
                                )
                            elif chart_type == "Scatter Plot":
                                fig = px.scatter(
                                    chart_data,
                                    x=st.session_state.x_axis_col,
                                    y=st.session_state.y_axis_col,
                                    color=st.session_state.y_axis_col,
                                    title=f"Scatter Plot of {st.session_state.y_axis_col} vs {st.session_state.x_axis_col}",
                                    labels={
                                        st.session_state.x_axis_col: f"{st.session_state.x_axis_col} (X-Axis)",
                                        st.session_state.y_axis_col: f"{st.session_state.y_axis_col} (Y-Axis)",
                                    },
                                )
                            elif chart_type == "Line Chart":
                                # Line chart code here
                                fig = px.line(
                                    chart_data,
                                    x=st.session_state.x_axis_col,
                                    y=st.session_state.y_axis_col,
                                    title=f"Line Chart of {st.session_state.y_axis_col} vs {st.session_state.x_axis_col}",
                                    labels={
                                        st.session_state.x_axis_col: f"{st.session_state.x_axis_col} (X-Axis)",
                                        st.session_state.y_axis_col: f"{st.session_state.y_axis_col} (Y-Axis)",
                                    },
                                )

                            # Add annotations for max/min if needed
                            max_value = chart_data[st.session_state.y_axis_col].max()
                            max_rows = chart_data[chart_data[st.session_state.y_axis_col] == max_value]
                            for _, max_row in max_rows.iterrows():
                                fig.add_annotation(
                                    x=max_row[st.session_state.x_axis_col],
                                    y=max_value,
                                    text="Highest Value",
                                    showarrow=True,
                                    arrowhead=2,
                                    font=dict(color="darkred"),
                                    arrowcolor="darkred",
                                )

                            min_value = chart_data[st.session_state.y_axis_col].min()
                            min_rows = chart_data[chart_data[st.session_state.y_axis_col] == min_value]
                            for _, min_row in min_rows.iterrows():
                                fig.add_annotation(
                                    x=min_row[st.session_state.x_axis_col],
                                    y=min_value,
                                    text="Lowest Value",
                                    showarrow=True,
                                    arrowhead=2,
                                    font=dict(color="darkblue"),
                                    arrowcolor="darkblue",
                                )

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error generating chart: {e}")
                    else:
                        st.warning("No valid data available for the selected columns.")
                else:
                    st.warning("Please select valid columns for both X-axis and Y-axis.")
                
                st.subheader("Analysis Summary")
                st.write(analysis_summary_from_openai)

                if st.button("Send Email with Summary"):
                    try:
                        if send_email_alert_with_summary(analysis_summary_from_openai):
                            st.success("Email alert sent successfully!")
                        else:
                            st.error("Failed to send email alert.")
                    except Exception as e:
                        st.error(f"Error sending email: {e}")


            

    elif selected_nav == "Live Leaderboard":
        st.title("Live Leaderboard")
        st.info("Leaderboard will update in real-time during the hackathon.")

    elif selected_nav == "Contact Us":
        st.title("Contact Us")
        st.markdown(
            """
            **For queries, reach us at:**  

            """
        )
        create_contact_map()
    st.sidebar.markdown("---")
    st.sidebar.markdown("🎉 **Good Luck!** 🎉") 
 
if __name__ == "__main__":
    main()