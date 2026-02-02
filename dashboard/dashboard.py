# Import library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_theme(style="dark")

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Dashboard",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("dashboard/main_data.csv")

all_df = load_data()

# Datetime conversion
all_df["order_purchase_timestamp"] = pd.to_datetime(
    all_df["order_purchase_timestamp"]
)

# Sidebar - Date Filter
st.sidebar.header("Filter")

min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

start_date, end_date = st.sidebar.date_input(
    label="Rentang Waktu",
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

filtered_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
]

# Helper Functions
def create_monthly_orders_df(df):
    monthly_orders = (
        df.set_index("order_purchase_timestamp")
        .resample("ME")["order_id"]
        .nunique()
        .reset_index()
        .rename(columns={"order_id": "order_count"})
    )
    monthly_orders["year_month"] = (
        monthly_orders["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )
    return monthly_orders

# Monthly Revenue
def create_monthly_revenue_df(df):
    monthly_revenue = (
        df.set_index("order_purchase_timestamp")
        .resample("ME")["payment_value"]
        .sum()
        .reset_index()
    )
    monthly_revenue["year_month"] = (
        monthly_revenue["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )
    monthly_revenue["total_revenue_million"] = (
        monthly_revenue["payment_value"] / 1_000_000
    )
    return monthly_revenue

# RFM Distribution
def create_rfm_df(df):
    rfm_df = (
        df.groupby("customer_id", as_index=False)
        .agg({
            "order_purchase_timestamp": "max",
            "order_id": "nunique",
            "payment_value": "sum"
        })
    )

    rfm_df.columns = [
        "customer_id",
        "last_purchase_date",
        "frequency",
        "monetary"
    ]

    max_date = df["order_purchase_timestamp"].max()
    rfm_df["recency"] = (
        max_date - rfm_df["last_purchase_date"]
    ).dt.days

    return rfm_df

# RFM Segmentation
def create_rfm_segment_df(rfm_df):
    rfm_df = rfm_df.copy()

    rfm_df["r_rank"] = pd.qcut(rfm_df["recency"], 5, labels=[5,4,3,2,1])
    rfm_df["f_rank"] = pd.qcut(
        rfm_df["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]
    )
    rfm_df["m_rank"] = pd.qcut(rfm_df["monetary"], 5, labels=[1,2,3,4,5])

    rfm_df["rfm_score"] = (
        rfm_df["r_rank"].astype(int) +
        rfm_df["f_rank"].astype(int) +
        rfm_df["m_rank"].astype(int)
    )

    def segment(score):
        if score >= 13:
            return "Top Customers"
        elif score >= 10:
            return "High Value Customers"
        elif score >= 7:
            return "Medium Value Customers"
        elif score >= 4:
            return "Low Value Customers"
        else:
            return "Lost Customers"

    rfm_df["customer_segment"] = rfm_df["rfm_score"].apply(segment)
    return rfm_df

# Dashboard Header
st.title("E-Commerce Sales & Customer Dashboard")
st.caption("Analisis Penjualan dan Segmentasi Customer Menggunakan RFM")

# Business Overview
st.subheader("Business Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Orders", f"{filtered_df['order_id'].nunique():,}")

with col2:
    st.metric(
        "Total Revenue",
        f"{filtered_df['payment_value'].sum() / 1_000_000:.2f} M"
    )

with col3:
    st.metric(
        "Total Customers",
        f"{filtered_df['customer_id'].nunique():,}"
    )

# Monthly Performance Overview
st.subheader("Monthly Performance Overview")

orders_per_month = create_monthly_orders_df(filtered_df)
revenue_per_month = create_monthly_revenue_df(filtered_df)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        orders_per_month["year_month"],
        orders_per_month["order_count"],
        marker="o",
        linewidth=2
    )
    ax.set_title("Monthly Order Trend")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Orders")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        revenue_per_month["year_month"],
        revenue_per_month["total_revenue_million"],
        marker="o",
        linewidth=2
    )
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Revenue (Million)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# Product Performance
st.subheader("Best & Worst Performing Product Categories")

category_sales_df = (
    filtered_df
    .groupby("product_category_name_english")
    .agg(total_items_sold=("order_item_id", "count"))
    .reset_index()
    .sort_values("total_items_sold", ascending=False)
)

top_5 = category_sales_df.head(5)
bottom_5 = (
    category_sales_df.tail(5)
    .sort_values("total_items_sold", ascending=True)
)

best_colors = ["#2C7BE5"] + ["#D3D3D3"] * (len(top_5) - 1)
worst_colors = ["#2C7BE5"] + ["#D3D3D3"] * (len(bottom_5) - 1)

fig, ax = plt.subplots(1, 2, figsize=(20, 6))

sns.barplot(
    data=top_5,
    x="total_items_sold",
    y="product_category_name_english",
    palette=best_colors,
    ax=ax[0]
)
ax[0].set_title("Best Performing Categories")
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].margins(y=0.15)

sns.barplot(
    data=bottom_5,
    x="total_items_sold",
    y="product_category_name_english",
    palette=worst_colors,
    ax=ax[1]
)
ax[1].set_title("Worst Performing Categories")
ax[1].invert_xaxis()
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].margins(y=0.15)

plt.tight_layout(pad=3)
st.pyplot(fig)

# Payment Method Distribution
st.subheader("Payment Method Distribution")

payment_df = (
    filtered_df.groupby("payment_type")["customer_id"]
    .nunique()
    .reset_index(name="customer_count")
    .sort_values("customer_count", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=payment_df,
    x="payment_type",
    y="customer_count",
    palette=["#2C7BE5"] + ["#D3D3D3"] * (len(payment_df) - 1),
    ax=ax
)
ax.set_xlabel(None)
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

# Geographic Analysis
st.subheader("Top 10 States by Number of Customers")

state_df = (
    filtered_df.groupby("customer_state")["customer_id"]
    .nunique()
    .reset_index(name="customer_count")
    .sort_values("customer_count", ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=state_df,
    y="customer_state",
    x="customer_count",
    palette=["#2C7BE5"] + ["#D3D3D3"] * (len(state_df) - 1),
    ax=ax
)
ax.set_xlabel("Number of Customers")
ax.set_ylabel(None)
st.pyplot(fig)

# RFM Distribution
st.subheader("RFM Distribution")

rfm_df = create_rfm_df(filtered_df)

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots()
    sns.boxplot(y=rfm_df["recency"], color="#2C7BE5", ax=ax)
    ax.set_title("Recency (days)")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(y=rfm_df["frequency"], color="#2C7BE5", ax=ax)
    ax.set_title("Frequency")
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots()
    sns.boxplot(y=rfm_df["monetary"], color="#2C7BE5", ax=ax)
    ax.set_title("Monetary")
    st.pyplot(fig)

# RFM Customer Segmentation
st.subheader("Customer Segmentation Based on RFM")

rfm_segment_df = create_rfm_segment_df(rfm_df)

segment_df = (
    rfm_segment_df
    .groupby("customer_segment")["customer_id"]
    .nunique()
    .reset_index(name="customer_count")
    .sort_values("customer_count", ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=segment_df,
    y="customer_segment",
    x="customer_count",
    palette=["#2C7BE5"] + ["#D3D3D3"] * (len(segment_df) - 1),
    ax=ax
)
ax.set_xlabel("Number of Customers")
ax.set_ylabel(None)
st.pyplot(fig)

# Footer
st.caption("© 2026 | Makruf Alkarkhi | E-Commerce Sales & Customer Analysis Dashboard")