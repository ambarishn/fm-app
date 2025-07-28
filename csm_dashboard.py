import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# Page configuration
st.set_page_config(
    page_title="CSM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dashboard styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .critical-sr {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .expansion-opportunity {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .referenceable {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .not-referenceable {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .pillar-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Customer data structure
CUSTOMER_DATA = {
    "Customer A": {
        "products": ["Oracle Fusion ERP", "Oracle Fusion HCM", "Oracle Fusion SCM"],
        "pillars": ["ERP", "HCM", "SCM"]
    },
    "Customer B": {
        "products": ["Oracle Fusion HCM"],
        "pillars": ["HCM"]
    }
}

def load_customer_data():
    """Load customer data from JSON file"""
    if os.path.exists('customer_data.json'):
        with open('customer_data.json', 'r') as f:
            return json.load(f)
    return {}

def save_customer_data(data):
    """Save customer data to JSON file"""
    with open('customer_data.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)

def display_header():
    """Display the dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 CSM Dashboard</h1>
        <h3>Customer Success Management Overview</h3>
    </div>
    """, unsafe_allow_html=True)

def display_utilization_metrics(customer_name, pillars):
    """Display utilization metrics for each pillar"""
    st.subheader("📈 Utilization Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    for i, pillar in enumerate(pillars):
        col = col1 if i == 0 else col2 if i == 1 else col3
        
        with col:
            utilization_key = f"{customer_name}_utilization_{pillar}"
            utilization = st.slider(
                f"{pillar} Utilization (%)",
                min_value=0,
                max_value=100,
                value=st.session_state.get(utilization_key, 50),
                key=utilization_key,
                help=f"Current utilization percentage for {pillar}"
            )
            
            # Color coding based on utilization
            if utilization >= 80:
                color = "#28a745"  # Green
                status = "Excellent"
            elif utilization >= 60:
                color = "#ffc107"  # Yellow
                status = "Good"
            else:
                color = "#dc3545"  # Red
                status = "Needs Attention"
            
            st.markdown(f"""
            <div class="pillar-card">
                <h4 style="color: {color};">{pillar}: {utilization}%</h4>
                <p style="color: {color}; font-weight: bold;">{status}</p>
            </div>
            """, unsafe_allow_html=True)

def display_critical_srs(customer_name):
    """Display critical SRs section"""
    st.subheader("🚨 Critical Service Requests")
    
    critical_sr_key = f"{customer_name}_critical_srs"
    critical_srs = st.number_input(
        "Number of Critical SRs",
        min_value=0,
        max_value=50,
        value=st.session_state.get(critical_sr_key, 0),
        key=critical_sr_key,
        help="Number of critical service requests currently open"
    )
    
    if critical_srs > 0:
        st.markdown(f"""
        <div class="critical-sr">
            <h3>⚠️ {critical_srs} Critical SR(s) Open</h3>
            <p>Immediate attention required</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical SRs currently open")

def display_css_leads(customer_name):
    """Display CSS leads section"""
    st.subheader("👥 CSS Leads")
    
    css_leads_key = f"{customer_name}_css_leads"
    css_leads = st.number_input(
        "Number of CSS Leads",
        min_value=0,
        max_value=100,
        value=st.session_state.get(css_leads_key, 0),
        key=css_leads_key,
        help="Number of Customer Success Services leads"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("CSS Leads", css_leads)
    
    with col2:
        if css_leads > 0:
            st.info(f"📞 {css_leads} lead(s) to follow up")
        else:
            st.info("📞 No CSS leads currently")

def display_expansion_opportunities(customer_name):
    """Display expansion opportunities section"""
    st.subheader("🚀 Expansion Opportunities")
    
    # Predefined product options
    product_options = [
        "Oracle Fusion ERP",
        "Oracle Fusion HCM", 
        "Oracle Fusion SCM",
        "Oracle Fusion CX",
        "Oracle Analytics Cloud",
        "Oracle Integration Cloud",
        "Oracle Autonomous Database",
        "Oracle Cloud Infrastructure"
    ]
    
    expansion_key = f"{customer_name}_expansion_products"
    selected_products = st.multiselect(
        "Select Expansion Products",
        options=product_options,
        default=st.session_state.get(expansion_key, []),
        key=expansion_key,
        help="Select products for expansion opportunities"
    )
    
    if selected_products:
        st.markdown("""
        <div class="expansion-opportunity">
            <h4>🎯 Expansion Opportunities Identified</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for product in selected_products:
            st.markdown(f"• **{product}**")
    else:
        st.info("📋 No expansion opportunities identified yet")

def display_referenceability(customer_name):
    """Display referenceability section"""
    st.subheader("📞 Referenceability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        referenceable_key = f"{customer_name}_referenceable"
        is_referenceable = st.selectbox(
            "Referenceable Status",
            options=["Yes", "No"],
            index=0 if st.session_state.get(referenceable_key, "Yes") == "Yes" else 1,
            key=referenceable_key,
            help="Is this customer willing to be a reference?"
        )
    
    with col2:
        expiry_key = f"{customer_name}_referenceable_expiry"
        
        # Handle date value properly
        default_date = datetime.now() + timedelta(days=365)
        saved_date = st.session_state.get(expiry_key, default_date)
        
        # Convert string to date if needed
        if isinstance(saved_date, str):
            try:
                saved_date = datetime.strptime(saved_date, "%Y-%m-%d").date()
            except:
                saved_date = default_date
        elif isinstance(saved_date, datetime):
            saved_date = saved_date.date()
        
        expiry_date = st.date_input(
            "Expiry Date",
            value=saved_date,
            key=expiry_key,
            help="When does the referenceability expire?"
        )
    
    # Display status with color coding
    if is_referenceable == "Yes":
        days_remaining = (expiry_date - datetime.now().date()).days
        
        if days_remaining > 30:
            status_color = "referenceable"
            status_icon = "✅"
            status_text = f"Referenceable (Expires in {days_remaining} days)"
        else:
            status_color = "not-referenceable"
            status_icon = "⚠️"
            status_text = f"Referenceable (Expires soon - {days_remaining} days)"
    else:
        status_color = "not-referenceable"
        status_icon = "❌"
        status_text = "Not Referenceable"
    
    st.markdown(f"""
    <div class="{status_color}">
        <h4>{status_icon} {status_text}</h4>
        <p>Expiry Date: {expiry_date.strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

def display_customer_summary(customer_name, customer_info):
    """Display customer summary"""
    st.markdown(f"""
    <div class="metric-card">
        <h2>🏢 {customer_name}</h2>
        <h4>Products: {', '.join(customer_info['products'])}</h4>
        <h4>Pillars: {', '.join(customer_info['pillars'])}</h4>
    </div>
    """, unsafe_allow_html=True)

def save_data_button(customer_name):
    """Save data button"""
    if st.button(f"💾 Save {customer_name} Data", key=f"save_{customer_name}"):
        # Collect all session state data for this customer
        customer_data = {}
        
        # Utilization data
        for pillar in CUSTOMER_DATA[customer_name]['pillars']:
            key = f"{customer_name}_utilization_{pillar}"
            if key in st.session_state:
                customer_data[f"utilization_{pillar}"] = st.session_state[key]
        
        # Other metrics
        metrics = ['critical_srs', 'css_leads', 'expansion_products', 'referenceable', 'referenceable_expiry']
        for metric in metrics:
            key = f"{customer_name}_{metric}"
            if key in st.session_state:
                customer_data[metric] = st.session_state[key]
        
        # Save to file
        all_data = load_customer_data()
        all_data[customer_name] = customer_data
        save_customer_data(all_data)
        
        st.success(f"✅ {customer_name} data saved successfully!")

def load_data_button(customer_name):
    """Load data button"""
    if st.button(f"📂 Load {customer_name} Data", key=f"load_{customer_name}"):
        all_data = load_customer_data()
        if customer_name in all_data:
            customer_data = all_data[customer_name]
            
            # Load utilization data
            for pillar in CUSTOMER_DATA[customer_name]['pillars']:
                key = f"{customer_name}_utilization_{pillar}"
                if f"utilization_{pillar}" in customer_data:
                    st.session_state[key] = customer_data[f"utilization_{pillar}"]
            
            # Load other metrics
            metrics = ['critical_srs', 'css_leads', 'expansion_products', 'referenceable']
            for metric in metrics:
                key = f"{customer_name}_{metric}"
                if metric in customer_data:
                    st.session_state[key] = customer_data[metric]
            
            # Handle date conversion for referenceable_expiry
            if 'referenceable_expiry' in customer_data:
                expiry_key = f"{customer_name}_referenceable_expiry"
                expiry_value = customer_data['referenceable_expiry']
                
                # Convert string date to datetime if needed
                if isinstance(expiry_value, str):
                    try:
                        st.session_state[expiry_key] = datetime.strptime(expiry_value, "%Y-%m-%d").date()
                    except:
                        st.session_state[expiry_key] = datetime.now() + timedelta(days=365)
                else:
                    st.session_state[expiry_key] = expiry_value
            
            st.success(f"✅ {customer_name} data loaded successfully!")
            st.rerun()
        else:
            st.warning(f"No saved data found for {customer_name}")

def main():
    """Main dashboard function"""
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        st.write("Manage your customer data")
        
        # Global actions
        st.subheader("🔄 Global Actions")
        
        if st.button("💾 Save All Data"):
            # Save all customer data
            all_data = {}
            for customer_name in CUSTOMER_DATA.keys():
                customer_data = {}
                
                # Utilization data
                for pillar in CUSTOMER_DATA[customer_name]['pillars']:
                    key = f"{customer_name}_utilization_{pillar}"
                    if key in st.session_state:
                        customer_data[f"utilization_{pillar}"] = st.session_state[key]
                
                # Other metrics
                metrics = ['critical_srs', 'css_leads', 'expansion_products', 'referenceable', 'referenceable_expiry']
                for metric in metrics:
                    key = f"{customer_name}_{metric}"
                    if key in st.session_state:
                        customer_data[metric] = st.session_state[key]
                
                all_data[customer_name] = customer_data
            
            save_customer_data(all_data)
            st.success("✅ All data saved successfully!")
        
        if st.button("📂 Load All Data"):
            all_data = load_customer_data()
            for customer_name, customer_data in all_data.items():
                if customer_name in CUSTOMER_DATA:
                    # Load utilization data
                    for pillar in CUSTOMER_DATA[customer_name]['pillars']:
                        key = f"{customer_name}_utilization_{pillar}"
                        if f"utilization_{pillar}" in customer_data:
                            st.session_state[key] = customer_data[f"utilization_{pillar}"]
                    
                    # Load other metrics
                    metrics = ['critical_srs', 'css_leads', 'expansion_products', 'referenceable']
                    for metric in metrics:
                        key = f"{customer_name}_{metric}"
                        if metric in customer_data:
                            st.session_state[key] = customer_data[metric]
                    
                    # Handle date conversion for referenceable_expiry
                    if 'referenceable_expiry' in customer_data:
                        expiry_key = f"{customer_name}_referenceable_expiry"
                        expiry_value = customer_data['referenceable_expiry']
                        
                        # Convert string date to datetime if needed
                        if isinstance(expiry_value, str):
                            try:
                                st.session_state[expiry_key] = datetime.strptime(expiry_value, "%Y-%m-%d").date()
                            except:
                                st.session_state[expiry_key] = datetime.now() + timedelta(days=365)
                        else:
                            st.session_state[expiry_key] = expiry_value
            
            st.success("✅ All data loaded successfully!")
            st.rerun()
        
        st.divider()
        
        # Data export
        st.subheader("📤 Export Data")
        if st.button("📊 Export to CSV"):
            # Create DataFrame for export
            export_data = []
            for customer_name, customer_info in CUSTOMER_DATA.items():
                row = {"Customer": customer_name}
                
                # Utilization data
                for pillar in customer_info['pillars']:
                    key = f"{customer_name}_utilization_{pillar}"
                    row[f"{pillar} Utilization %"] = st.session_state.get(key, 0)
                
                # Other metrics
                row["Critical SRs"] = st.session_state.get(f"{customer_name}_critical_srs", 0)
                row["CSS Leads"] = st.session_state.get(f"{customer_name}_css_leads", 0)
                row["Expansion Products"] = ", ".join(st.session_state.get(f"{customer_name}_expansion_products", []))
                row["Referenceable"] = st.session_state.get(f"{customer_name}_referenceable", "No")
                row["Expiry Date"] = st.session_state.get(f"{customer_name}_referenceable_expiry", "N/A")
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"csm_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Main content with tabs
    tab1, tab2 = st.tabs(["🏢 Customer A", "🏢 Customer B"])
    
    # Customer A Tab
    with tab1:
        customer_name = "Customer A"
        customer_info = CUSTOMER_DATA[customer_name]
        
        display_customer_summary(customer_name, customer_info)
        
        # Data management buttons
        col1, col2 = st.columns(2)
        with col1:
            save_data_button(customer_name)
        with col2:
            load_data_button(customer_name)
        
        st.divider()
        
        # Dashboard metrics
        display_utilization_metrics(customer_name, customer_info['pillars'])
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_critical_srs(customer_name)
            st.divider()
            display_css_leads(customer_name)
        
        with col2:
            display_expansion_opportunities(customer_name)
            st.divider()
            display_referenceability(customer_name)
    
    # Customer B Tab
    with tab2:
        customer_name = "Customer B"
        customer_info = CUSTOMER_DATA[customer_name]
        
        display_customer_summary(customer_name, customer_info)
        
        # Data management buttons
        col1, col2 = st.columns(2)
        with col1:
            save_data_button(customer_name)
        with col2:
            load_data_button(customer_name)
        
        st.divider()
        
        # Dashboard metrics
        display_utilization_metrics(customer_name, customer_info['pillars'])
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_critical_srs(customer_name)
            st.divider()
            display_css_leads(customer_name)
        
        with col2:
            display_expansion_opportunities(customer_name)
            st.divider()
            display_referenceability(customer_name)

if __name__ == "__main__":
    main() 