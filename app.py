import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
from datetime import datetime, date
import calendar

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect("expense.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        date TEXT, 
        category TEXT, 
        amount REAL, 
        note TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_expense(date, category, amount, note):
    conn = sqlite3.connect("expense.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO expenses (date, category, amount, note) VALUES (?, ?, ?, ?)", 
                   (str(date), category, amount, note))
    conn.commit()
    conn.close()

def get_expenses():
    conn = sqlite3.connect("expense.db")
    df = pd.read_sql_query("SELECT date, category, amount, note FROM expenses", conn)
    conn.close()
    return df

# Initialize database
init_db() 

# --- WEBSITE UI SETUP ---
st.set_page_config(page_title="Spendwise", page_icon="💰", layout="wide")
st.title("💰 Spendwise Dashboard")
st.write("Dynamic Daily Budget Active! 📅")

# Fetch current records
df = get_expenses()

# --- SIDEBAR: FILTERS & BUDGET ---
st.sidebar.header("🔍 Filter Your Data")

if not df.empty:
    categories = ["All"] + df['category'].unique().tolist()
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    if selected_category != "All":
        df = df[df['category'] == selected_category]
        
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input("Filter by Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
else:
    st.sidebar.write("Add expenses to unlock filters.")

# Budget Limits
st.sidebar.divider()
st.sidebar.header("⚠️ Monthly Budget")
budget = st.sidebar.number_input("Set Limit (₹)", min_value=0.0, value=2000.0, step=500.0)
total_spent = df['amount'].sum() if not df.empty else 0
remaining = budget - total_spent

if total_spent >= budget and budget > 0:
    st.sidebar.error(f"Over Budget! Spent: ₹{total_spent:,.2f} / ₹{budget:,.2f}")
else:
    st.sidebar.success(f"Within Budget. Spent: ₹{total_spent:,.2f} / ₹{budget:,.2f}")

# --- KPI METRIC CARDS (2x2 Grid for Wide Spacing) ---
if not df.empty:
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.metric("Total Spent", f"₹{total_spent:,.2f}")
    with kpi2:
        st.metric("Budget Remaining", f"₹{remaining:,.2f}" if remaining > 0 else "₹0.00")
        
    kpi3, kpi4 = st.columns(2)
    with kpi3:
        unique_days = df['date'].nunique()
        avg_daily = total_spent / unique_days if unique_days > 0 else 0
        st.metric("Avg Daily Spend", f"₹{avg_daily:,.2f}")
    with kpi4:
        today = date.today()
        _, days_in_month = calendar.monthrange(today.year, today.month)
        days_remaining = days_in_month - today.day + 1
        daily_power = remaining / days_remaining if remaining > 0 else 0
        st.metric("Safe to Spend Today", f"₹{daily_power:,.2f}")

st.divider()

# --- ADD EXPENSE SECTION ---
with st.expander("➕ Click to Add a New Expense", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date_input = st.date_input("Date", date.today())
    with col2:
        category = st.selectbox("Category", ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Other"])
    with col3:
        amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
    with col4:
        note = st.text_input("Note")
        
    if st.button("Save Expense", use_container_width=True):
        if amount > 0:
            insert_expense(date_input, category, amount, note)
            st.success(f"Success! Saved ₹{amount:,.2f} for {category}.")
            st.rerun() 
        else:
            st.error("Please enter an amount greater than 0.")

# --- RECENT TRANSACTIONS ---
st.subheader("Recent Transactions")
if not df.empty:
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No expenses found. Click the expander above to add your first transaction!")

# --- CHARTS ---
st.divider() 
st.subheader("📊 Expense Dashboard")

if not df.empty:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        category_df = df.groupby("category", as_index=False)["amount"].sum()
        fig_pie = px.pie(category_df, names="category", values="amount", 
                         title="Spending by Category", hole=0.4)
        fig_pie.update_layout(dragmode=False)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
    with chart_col2:
        date_df = df.groupby("date", as_index=False)["amount"].sum()
        fig_bar = px.bar(date_df, x="date", y="amount", 
                         title="Daily Expenses", text_auto=True)
        fig_bar.update_layout(dragmode=False)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
else:
    st.write("Not enough data to display charts.")

# --- AI PREDICTION & PDF EXPORT ---
st.divider()
col_ai, col_export = st.columns(2)

with col_ai:
    st.subheader("🤖 AI Expense Prediction")
    if not df.empty:
        daily_df = df.groupby("date", as_index=False)["amount"].sum()
        if len(daily_df) > 1:
            daily_df["day_index"] = np.arange(len(daily_df))
            X = daily_df[["day_index"]]
            y = daily_df["amount"]
            
            model = LinearRegression()
            model.fit(X, y)
            
            next_day_index = [[len(daily_df)]]
            predicted_amount = model.predict(next_day_index)[0]
            st.info(f"Predicted next active day spending: **₹{max(0, round(predicted_amount, 2)):,.2f}**")
        else:
            st.warning("Need expenses on at least 2 different dates for AI predictions.")
    else:
        st.write("Add some data to unlock AI predictions.")

with col_export:
    st.subheader("📄 Export Report")
    if not df.empty:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(190, 10, "Spendwise Financial Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        
        for index, row in df.iterrows():
            pdf.cell(190, 10, txt=f"Date: {row['date']} | {row['category']} | Rs.{row['amount']:.2f} | {row['note']}", ln=True)
        
        pdf_file_path = "expense_report.pdf"
        pdf.output(pdf_file_path)
        
        with open(pdf_file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=f,
                file_name="Spendwise_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.write("No data to export.")
