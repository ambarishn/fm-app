# 📊 CSM Dashboard

A comprehensive Customer Success Manager (CSM) dashboard built with Streamlit for managing customer relationships, utilization metrics, and expansion opportunities.

## 🎯 Overview

This dashboard is designed for CSMs to track and manage customer success metrics across multiple customers with different Oracle Fusion product portfolios.

## 🏢 Customer Profiles

### Customer A
- **Products**: Oracle Fusion ERP, HCM, SCM
- **Pillars**: ERP, HCM, SCM

### Customer B
- **Products**: Oracle Fusion HCM
- **Pillars**: HCM

## 📈 Dashboard Elements

### 1. **Utilization Metrics** 📊
- **Interactive sliders** for each pillar (ERP, HCM, SCM)
- **Color-coded status indicators**:
  - 🟢 Green (80%+): Excellent
  - 🟡 Yellow (60-79%): Good
  - 🔴 Red (<60%): Needs Attention
- **Real-time percentage display**

### 2. **Critical Service Requests** 🚨
- **Number input** for tracking critical SRs
- **Visual alerts** when critical SRs are present
- **Status indicators** for immediate attention

### 3. **CSS Leads** 👥
- **Lead count tracking**
- **Follow-up reminders**
- **Metric display with visual feedback**

### 4. **Expansion Opportunities** 🚀
- **Multi-select dropdown** with predefined Oracle products:
  - Oracle Fusion ERP
  - Oracle Fusion HCM
  - Oracle Fusion SCM
  - Oracle Fusion CX
  - Oracle Analytics Cloud
  - Oracle Integration Cloud
  - Oracle Autonomous Database
  - Oracle Cloud Infrastructure
- **Visual identification** of opportunities

### 5. **Referenceability** 📞
- **Yes/No selection** for reference status
- **Expiry date picker** with automatic calculation
- **Color-coded status**:
  - 🔵 Blue: Referenceable (30+ days remaining)
  - 🟡 Yellow: Referenceable (expiring soon)
  - 🔴 Red: Not referenceable

## 🎮 Features

### **Data Management**
- 💾 **Save/Load functionality** for each customer
- 🔄 **Global save/load** for all customers
- 📤 **CSV export** with timestamped filenames
- 📂 **Persistent storage** using JSON files

### **User Interface**
- 🎨 **Professional styling** with gradients and cards
- 📱 **Responsive design** for all screen sizes
- 🏷️ **Tabbed interface** for easy customer switching
- 📊 **Real-time updates** and visual feedback

### **Data Persistence**
- 💾 **Automatic saving** to `customer_data.json`
- 📂 **Session state management** for seamless experience
- 🔄 **Data recovery** and backup functionality

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the Dashboard
```bash
python -m streamlit run csm_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 📋 Usage Instructions

### **Adding Customer Data**
1. **Navigate** to the customer tab (Customer A or B)
2. **Adjust utilization sliders** for each pillar
3. **Enter critical SR count** if any
4. **Set CSS lead count**
5. **Select expansion opportunities** from dropdown
6. **Set referenceability status** and expiry date
7. **Save data** using the save button

### **Data Management**
- **Individual Save**: Save data for specific customer
- **Individual Load**: Load previously saved data
- **Global Save**: Save all customer data at once
- **Global Load**: Load all customer data at once
- **CSV Export**: Download data as CSV file

### **Best Practices**
- **Save regularly** to prevent data loss
- **Export data** for reporting and analysis
- **Monitor critical SRs** closely
- **Track referenceability expiry** dates
- **Update expansion opportunities** as they arise

## 🎨 Customization

### **Adding New Customers**
1. **Modify** `CUSTOMER_DATA` dictionary in the code
2. **Add** new customer with products and pillars
3. **Create** new tab in the main function
4. **Update** export functionality if needed

### **Adding New Products**
1. **Update** `product_options` list in `display_expansion_opportunities()`
2. **Add** new Oracle products as needed

### **Modifying Metrics**
1. **Adjust** slider ranges and thresholds
2. **Update** color coding logic
3. **Modify** status messages and indicators

## 📊 Data Structure

### **JSON Storage Format**
```json
{
  "Customer A": {
    "utilization_ERP": 75,
    "utilization_HCM": 85,
    "utilization_SCM": 60,
    "critical_srs": 2,
    "css_leads": 5,
    "expansion_products": ["Oracle Fusion CX", "Oracle Analytics Cloud"],
    "referenceable": "Yes",
    "referenceable_expiry": "2024-12-31"
  }
}
```

### **CSV Export Format**
- Customer name
- Utilization percentages for each pillar
- Critical SR count
- CSS lead count
- Expansion products (comma-separated)
- Referenceability status
- Expiry date

## 🔧 Technical Details

### **Dependencies**
- `streamlit>=1.28.0`: Web application framework
- `pandas>=1.5.0`: Data manipulation and CSV export

### **File Structure**
```
csm-dashboard/
├── csm_dashboard.py          # Main dashboard application
├── requirements.txt          # Python dependencies
├── customer_data.json        # Persistent data storage (auto-generated)
└── CSM_DASHBOARD_README.md   # This documentation
```

### **Browser Compatibility**
- Chrome, Firefox, Safari, Edge
- Mobile browsers
- Any modern web browser

## 🎯 Use Cases

### **Daily CSM Activities**
- **Morning review** of customer metrics
- **Critical SR monitoring** and escalation
- **Lead follow-up** tracking
- **Expansion opportunity** identification

### **Weekly/Monthly Reviews**
- **Utilization trend analysis**
- **Referenceability status** updates
- **Expansion pipeline** management
- **Performance reporting**

### **Quarterly Planning**
- **Customer health assessment**
- **Expansion opportunity** prioritization
- **Reference customer** identification
- **Success metric** tracking

## 🤝 Contributing

Feel free to:
- **Add new features** and metrics
- **Improve the UI/UX**
- **Add more customers** or products
- **Enhance data visualization**
- **Report bugs** or issues

## 📄 License

This project is open source and available under the MIT License.

---

**Happy Customer Success Management! 📊🎯** 