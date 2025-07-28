import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time

# Page configuration
st.set_page_config(
    page_title="CSM Dashboard v2 - AI Powered",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .next-steps {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .trend-analysis {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .product-yes {
        background: #d4edda;
        color: #155724;
        padding: 0.3rem 0.6rem;
        border-radius: 3px;
        font-weight: bold;
    }
    .product-no {
        background: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.6rem;
        border-radius: 3px;
        font-weight: bold;
    }
    .loading {
        text-align: center;
        padding: 2rem;
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Display the dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 CSM Dashboard v2 - AI Powered</h1>
        <h3>Intelligent Customer Success Management with GenAI Insights</h3>
    </div>
    """, unsafe_allow_html=True)

def load_sample_data():
    """Create sample CSV data for demonstration"""
    sample_data = {
        'Customer_Name': ['TechCorp Inc', 'Global Manufacturing', 'StartupXYZ', 'Enterprise Solutions'],
        'ARR_$': [750000, 1200000, 300000, 2000000],
        'ERP': ['Y', 'Y', 'N', 'Y'],
        'EPM': ['N', 'Y', 'N', 'Y'],
        'HCM': ['Y', 'Y', 'Y', 'Y'],
        'SCM': ['Y', 'Y', 'N', 'Y'],
        'FDI': ['N', 'Y', 'N', 'Y'],
        'Redwood_Adoption_%': [85, 95, 30, 90],
        'AI_Adoption_%': [70, 80, 20, 85],
        'CSS_Leads': [8, 12, 3, 15],
        'Referenceability': ['Y', 'Y', 'N', 'Y'],
        'ERP_Utilization_%': [85, 90, 0, 95],
        'EPM_Utilization_%': [0, 88, 0, 92],
        'HCM_Utilization_%': [92, 95, 65, 98],
        'SCM_Utilization_%': [78, 85, 0, 90],
        'FDI_Utilization_%': [0, 82, 0, 88],
        'ERP_Risk': ['Low', 'Low', 'N/A', 'Low'],
        'EPM_Risk': ['N/A', 'Low', 'N/A', 'Low'],
        'HCM_Risk': ['Low', 'Low', 'Medium', 'Low'],
        'SCM_Risk': ['Medium', 'Low', 'N/A', 'Low'],
        'FDI_Risk': ['N/A', 'Low', 'N/A', 'Low']
    }
    return pd.DataFrame(sample_data)

def call_free_llm_api(prompt, customer_data=None, all_data=None):
    """Call a free LLM API for AI insights"""
    try:
        # Initialize context
        context = ""
        
        # Prepare the context for customer data
        if customer_data is not None:
            context = f"""
            Customer: {customer_data['Customer_Name']}
            ARR: ${customer_data['ARR_$']:,.0f}
            Products: {', '.join([p for p in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI'] if customer_data[p] == 'Y'])}
            Redwood Adoption: {customer_data['Redwood_Adoption_%']}%
            AI Adoption: {customer_data['AI_Adoption_%']}%
            CSS Leads: {customer_data['CSS_Leads']}
            Referenceable: {customer_data['Referenceability']}
            """
            
            # Add utilization data
            utilizations = []
            for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
                if customer_data[product] == 'Y':
                    util_key = f"{product}_Utilization_%"
                    risk_key = f"{product}_Risk"
                    utilizations.append(f"{product}: {customer_data[util_key]}% utilization, {customer_data[risk_key]} risk")
            
            context += f"Utilization & Risk: {', '.join(utilizations)}"
        
        # Add portfolio data if available
        if all_data is not None:
            context += f"\n\nPortfolio Overview: {len(all_data)} customers with average ARR of ${all_data['ARR_$'].mean():,.0f}"
        
        # Create the full prompt
        full_prompt = f"{context}\n\n{prompt}"
        
        # Simulated response based on the prompt type
        if "summary" in prompt.lower():
            return generate_ai_summary(customer_data)
        elif "next steps" in prompt.lower():
            return generate_next_steps(customer_data)
        elif "trend" in prompt.lower():
            return generate_trend_analysis(all_data)
        else:
            return "AI analysis completed successfully."
            
    except Exception as e:
        return f"AI analysis could not be completed: {str(e)}"

def generate_ai_summary(customer_data):
    """Generate AI-powered customer summary"""
    arr = customer_data['ARR_$']
    products = [p for p in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI'] if customer_data[p] == 'Y']
    
    # Analyze utilization patterns
    utilizations = []
    for product in products:
        util_key = f"{product}_Utilization_%"
        utilizations.append(customer_data[util_key])
    
    avg_utilization = np.mean(utilizations) if utilizations else 0
    
    # Generate insights based on data patterns
    insights = []
    
    if arr > 1000000:
        insights.append("This is a high-value enterprise customer with significant revenue potential.")
    elif arr > 500000:
        insights.append("This is a mid-market customer with good growth potential.")
    else:
        insights.append("This is a smaller customer that may need more attention for growth.")
    
    if avg_utilization > 85:
        insights.append("Excellent utilization across all products indicates strong adoption.")
    elif avg_utilization > 70:
        insights.append("Good utilization with room for improvement in some areas.")
    else:
        insights.append("Low utilization suggests potential adoption challenges.")
    
    if customer_data['Redwood_Adoption_%'] > 80:
        insights.append("Strong Redwood adoption shows modern platform engagement.")
    
    if customer_data['AI_Adoption_%'] > 70:
        insights.append("High AI adoption indicates forward-thinking technology strategy.")
    
    if customer_data['CSS_Leads'] > 5:
        insights.append("Multiple CSS leads suggest active expansion opportunities.")
    
    if customer_data['Referenceability'] == 'Y':
        insights.append("Customer is referenceable, valuable for sales enablement.")
    
    # Risk assessment
    high_risk_products = []
    for product in products:
        risk_key = f"{product}_Risk"
        if customer_data[risk_key] == 'High':
            high_risk_products.append(product)
    
    if high_risk_products:
        insights.append(f"⚠️ High risk identified in: {', '.join(high_risk_products)}")
    
    return " ".join(insights)

def generate_next_steps(customer_data):
    """Generate AI-powered next steps for CSM"""
    steps = []
    
    # Analyze utilization gaps
    low_util_products = []
    for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
        if customer_data[product] == 'Y':
            util_key = f"{product}_Utilization_%"
            if customer_data[util_key] < 70:
                low_util_products.append(product)
    
    if low_util_products:
        steps.append(f"📈 **Utilization Improvement**: Focus on {', '.join(low_util_products)} with utilization below 70%")
    
    # Risk mitigation
    high_risk_products = []
    for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
        if customer_data[product] == 'Y':
            risk_key = f"{product}_Risk"
            if customer_data[risk_key] == 'High':
                high_risk_products.append(product)
    
    if high_risk_products:
        steps.append(f"🚨 **Risk Mitigation**: Immediate attention needed for {', '.join(high_risk_products)}")
    
    # Expansion opportunities
    if customer_data['CSS_Leads'] > 0:
        steps.append(f"💼 **Expansion**: {customer_data['CSS_Leads']} CSS leads to pursue")
    
    # Adoption acceleration
    if customer_data['Redwood_Adoption_%'] < 60:
        steps.append("🔄 **Modernization**: Accelerate Redwood adoption")
    
    if customer_data['AI_Adoption_%'] < 50:
        steps.append("🤖 **AI Enablement**: Increase AI feature adoption")
    
    # Referenceability
    if customer_data['Referenceability'] == 'N':
        steps.append("📞 **Referenceability**: Work on making customer referenceable")
    
    # Success planning
    if customer_data['ARR_$'] > 1000000:
        steps.append("🎯 **Strategic Planning**: Schedule quarterly business review")
    else:
        steps.append("📋 **Success Planning**: Schedule monthly check-in")
    
    return "\n\n".join(steps)

def generate_trend_analysis(all_data):
    """Generate AI-powered trend analysis across all customers"""
    if len(all_data) < 2:
        return "Insufficient data for trend analysis. Need at least 2 customers."
    
    insights = []
    
    # ARR analysis
    avg_arr = all_data['ARR_$'].mean()
    max_arr = all_data['ARR_$'].max()
    min_arr = all_data['ARR_$'].min()
    
    insights.append(f"💰 **Revenue Distribution**: Average ARR ${avg_arr:,.0f} (Range: ${min_arr:,.0f} - ${max_arr:,.0f})")
    
    # Product adoption trends
    product_adoption = {}
    for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
        adoption_rate = (all_data[product] == 'Y').mean() * 100
        product_adoption[product] = adoption_rate
        insights.append(f"📦 **{product}**: {adoption_rate:.1f}% adoption rate")
    
    # Utilization trends
    utilization_cols = [col for col in all_data.columns if 'Utilization_%' in col]
    avg_utilizations = {}
    for col in utilization_cols:
        product = col.replace('_Utilization_%', '')
        avg_util = all_data[col].mean()
        avg_utilizations[product] = avg_util
        insights.append(f"📊 **{product} Utilization**: {avg_util:.1f}% average")
    
    # Risk distribution
    risk_cols = [col for col in all_data.columns if '_Risk' in col]
    risk_summary = {}
    for col in risk_cols:
        product = col.replace('_Risk', '')
        risk_counts = all_data[col].value_counts()
        risk_summary[product] = risk_counts.to_dict()
    
    insights.append("⚠️ **Risk Distribution**: " + ", ".join([f"{p}: {r}" for p, r in risk_summary.items() if r]))
    
    # Overall health score
    health_scores = []
    for _, customer in all_data.iterrows():
        score = 0
        # Utilization score
        for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
            if customer[product] == 'Y':
                util_key = f"{product}_Utilization_%"
                score += customer[util_key] / 100
        
        # Risk score (inverse)
        for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
            if customer[product] == 'Y':
                risk_key = f"{product}_Risk"
                if customer[risk_key] == 'Low':
                    score += 1
                elif customer[risk_key] == 'Medium':
                    score += 0.5
        
        health_scores.append(score)
    
    avg_health = np.mean(health_scores)
    insights.append(f"🏥 **Portfolio Health**: Average health score {avg_health:.2f}/10")
    
    return "\n\n".join(insights)

def display_customer_overview(customer_data):
    """Display customer overview section"""
    st.markdown(f"""
    <div class="metric-card">
        <h2>🏢 {customer_data['Customer_Name']}</h2>
        <h3>ARR: ${customer_data['ARR_$']:,.0f}</h3>
    </div>
    """, unsafe_allow_html=True)

def display_products_and_adoption(customer_data):
    """Display products and adoption metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Products")
        
        # Display products
        for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
            if customer_data[product] == 'Y':
                st.markdown(f'<span class="product-yes">{product} ✓</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="product-no">{product} ✗</span>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Adoption Metrics")
        
        # Redwood Adoption
        if customer_data['Redwood_Adoption_%'] > 0:
            st.metric("Redwood Adoption", f"{customer_data['Redwood_Adoption_%']}%")
        
        # AI Adoption
        if customer_data['AI_Adoption_%'] > 0:
            st.metric("AI Adoption", f"{customer_data['AI_Adoption_%']}%")

def display_utilization_and_risk(customer_data):
    """Display utilization and risk metrics"""
    st.subheader("📊 Utilization & Risk Analysis")
    
    # Create utilization data for products the customer has
    util_data = []
    for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
        if customer_data[product] == 'Y':
            util_key = f"{product}_Utilization_%"
            risk_key = f"{product}_Risk"
            
            if util_key in customer_data:
                util_data.append({
                    'Product': product,
                    'Utilization': customer_data[util_key],
                    'Risk': customer_data.get(risk_key, 'N/A')
                })
    
    if util_data:
        # Create a DataFrame for visualization
        df_util = pd.DataFrame(util_data)
        
        # Display metrics
        cols = st.columns(len(util_data))
        for i, (_, row) in enumerate(df_util.iterrows()):
            with cols[i]:
                st.metric(f"{row['Product']} Utilization", f"{row['Utilization']}%")
                
                # Risk indicator
                risk = row['Risk']
                if risk == 'High':
                    st.markdown(f'<div class="risk-high">{risk} Risk</div>', unsafe_allow_html=True)
                elif risk == 'Medium':
                    st.markdown(f'<div class="risk-medium">{risk} Risk</div>', unsafe_allow_html=True)
                elif risk == 'Low':
                    st.markdown(f'<div class="risk-low">{risk} Risk</div>', unsafe_allow_html=True)
                else:
                    st.info(f"Risk: {risk}")

def display_css_and_referenceability(customer_data):
    """Display CSS leads and referenceability"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 CSS Leads")
        st.metric("CSS Leads", customer_data['CSS_Leads'])
        
        if customer_data['CSS_Leads'] > 0:
            st.info(f"📞 {customer_data['CSS_Leads']} lead(s) to follow up")
        else:
            st.info("📞 No CSS leads currently")
    
    with col2:
        st.subheader("📞 Referenceability")
        
        if customer_data['Referenceability'] == 'Y':
            st.success("✅ Customer is referenceable")
        else:
            st.error("❌ Customer is not referenceable")

def display_ai_insights(customer_data, all_data):
    """Display AI-generated insights with generate buttons"""
    st.subheader("🤖 AI-Powered Insights")
    
    # Initialize session state for AI insights
    customer_name = customer_data['Customer_Name']
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {}
    if customer_name not in st.session_state.ai_insights:
        st.session_state.ai_insights[customer_name] = {
            'summary': None,
            'next_steps': None,
            'trends': None
        }
    
    # Customer Summary
    with st.expander("📋 Customer Summary", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🤖 Generate Summary", key=f"gen_summary_{customer_name}"):
                with st.spinner("🤖 AI is analyzing customer data..."):
                    # Simulate AI thinking time
                    time.sleep(1.5)
                    summary = call_free_llm_api("Provide a comprehensive summary of this customer's current state, highlighting key metrics, strengths, and areas of concern.", customer_data)
                    st.session_state.ai_insights[customer_name]['summary'] = summary
                    st.rerun()
        
        with col2:
            if st.session_state.ai_insights[customer_name]['summary']:
                st.success("✅ Generated")
        
        # Display summary if available
        if st.session_state.ai_insights[customer_name]['summary']:
            st.markdown(f"""
            <div class="ai-insight">
                <h4>🎯 Customer Summary</h4>
                <p>{st.session_state.ai_insights[customer_name]['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Click 'Generate Summary' to get AI-powered customer insights")
    
    # Next Steps
    with st.expander("🎯 Next Steps", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 Generate Next Steps", key=f"gen_steps_{customer_name}"):
                with st.spinner("🤖 AI is planning actionable steps..."):
                    # Simulate AI thinking time
                    time.sleep(2)
                    next_steps = call_free_llm_api("Based on this customer's data, provide specific, actionable next steps for the CSM to take. Focus on utilization improvement, risk mitigation, expansion opportunities, and success planning.", customer_data)
                    st.session_state.ai_insights[customer_name]['next_steps'] = next_steps
                    st.rerun()
        
        with col2:
            if st.session_state.ai_insights[customer_name]['next_steps']:
                st.success("✅ Generated")
        
        # Display next steps if available
        if st.session_state.ai_insights[customer_name]['next_steps']:
            st.markdown(f"""
            <div class="next-steps">
                <h4>🚀 Recommended Actions</h4>
                <p>{st.session_state.ai_insights[customer_name]['next_steps']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Click 'Generate Next Steps' to get AI-powered recommendations")
    
    # Portfolio Trends
    with st.expander("📈 Portfolio Trends", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("📊 Generate Trends", key=f"gen_trends_{customer_name}"):
                with st.spinner("🤖 AI is analyzing portfolio patterns..."):
                    # Simulate AI thinking time
                    time.sleep(2.5)
                    trends = call_free_llm_api("Analyze trends across the entire customer portfolio. Provide insights on revenue distribution, product adoption patterns, utilization trends, risk distribution, and overall portfolio health.", all_data=all_data)
                    st.session_state.ai_insights[customer_name]['trends'] = trends
                    st.rerun()
        
        with col2:
            if st.session_state.ai_insights[customer_name]['trends']:
                st.success("✅ Generated")
        
        # Display trends if available
        if st.session_state.ai_insights[customer_name]['trends']:
            st.markdown(f"""
            <div class="trend-analysis">
                <h4>📊 Portfolio Analysis</h4>
                <p>{st.session_state.ai_insights[customer_name]['trends']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💡 Click 'Generate Trends' to get AI-powered portfolio analysis")
    
    # Generate All button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🤖 Generate All Insights", key=f"gen_all_{customer_name}", type="primary"):
            with st.spinner("🤖 AI is generating comprehensive insights..."):
                # Generate all insights with different thinking times
                time.sleep(1)
                summary = call_free_llm_api("Provide a comprehensive summary of this customer's current state, highlighting key metrics, strengths, and areas of concern.", customer_data)
                st.session_state.ai_insights[customer_name]['summary'] = summary
                
                time.sleep(1)
                next_steps = call_free_llm_api("Based on this customer's data, provide specific, actionable next steps for the CSM to take. Focus on utilization improvement, risk mitigation, expansion opportunities, and success planning.", customer_data)
                st.session_state.ai_insights[customer_name]['next_steps'] = next_steps
                
                time.sleep(1)
                trends = call_free_llm_api("Analyze trends across the entire customer portfolio. Provide insights on revenue distribution, product adoption patterns, utilization trends, risk distribution, and overall portfolio health.", all_data=all_data)
                st.session_state.ai_insights[customer_name]['trends'] = trends
                
                st.success("🎉 All AI insights generated successfully!")
                st.rerun()

def create_utilization_chart(customer_data):
    """Create utilization chart"""
    util_data = []
    for product in ['ERP', 'EPM', 'HCM', 'SCM', 'FDI']:
        if customer_data[product] == 'Y':
            util_key = f"{product}_Utilization_%"
            if util_key in customer_data:
                util_data.append({
                    'Product': product,
                    'Utilization': customer_data[util_key]
                })
    
    if util_data:
        df_chart = pd.DataFrame(util_data)
        
        fig = px.bar(
            df_chart, 
            x='Product', 
            y='Utilization',
            title=f"Utilization by Product - {customer_data['Customer_Name']}",
            color='Utilization',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            yaxis_title="Utilization (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function"""
    display_header()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data"
        )
        
        if st.button("📋 Download Sample CSV"):
            sample_df = load_sample_data()
            csv = sample_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_csm_data_v2.csv",
                mime="text/csv"
            )
        
        st.divider()
        
        st.markdown("""
        ### 🤖 AI Features:
        - **Intelligent Customer Summary**
        - **AI-Generated Next Steps**
        - **Portfolio Trend Analysis**
        - **Risk Assessment**
        - **Expansion Opportunities**
        
        ### 📋 Required CSV Columns:
        1. Customer_Name
        2. ARR_$
        3. ERP (Y/N)
        4. EPM (Y/N)
        5. HCM (Y/N)
        6. SCM (Y/N)
        7. FDI (Y/N)
        8. Redwood_Adoption_%
        9. AI_Adoption_%
        10. CSS_Leads
        11. Referenceability (Y/N)
        12-16. [Product]_Utilization_%
        17-21. [Product]_Risk (High/Med/Low)
        """)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display customer selector
            if 'Customer_Name' in df.columns:
                customer_names = df['Customer_Name'].tolist()
                selected_customer = st.selectbox(
                    "Select Customer",
                    customer_names,
                    help="Choose a customer to view their AI-powered dashboard"
                )
                
                # Get selected customer data
                customer_data = df[df['Customer_Name'] == selected_customer].iloc[0]
                
                # Display dashboard sections
                display_customer_overview(customer_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    display_products_and_adoption(customer_data)
                
                with col2:
                    display_css_and_referenceability(customer_data)
                
                display_utilization_and_risk(customer_data)
                
                # Charts
                st.subheader("📈 Utilization Chart")
                create_utilization_chart(customer_data)
                
                # AI Insights
                display_ai_insights(customer_data, df)
                
                # Raw data
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df)
                
            else:
                st.error("CSV file must contain 'Customer_Name' column")
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="metric-card">
            <h2>Welcome to CSM Dashboard v2 - AI Powered! 🤖</h2>
            <p>Upload a CSV file to get started with intelligent customer success analysis powered by GenAI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data
        st.subheader("📋 Sample Data Preview")
        sample_df = load_sample_data()
        st.dataframe(sample_df)
        
        st.info("💡 Use the sidebar to upload your CSV file or download the sample template")
        
        # AI Features showcase
        st.subheader("🤖 AI-Powered Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="ai-insight">
                <h4>🎯 Intelligent Summaries</h4>
                <p>AI-generated customer insights based on utilization, risk, and adoption patterns.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="next-steps">
                <h4>🚀 Next Steps</h4>
                <p>Actionable recommendations for CSM activities and customer success planning.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="trend-analysis">
                <h4>📊 Trend Analysis</h4>
                <p>Portfolio-wide insights and trend analysis across all customers.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 