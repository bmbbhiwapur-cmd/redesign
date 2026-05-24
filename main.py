def apply_ui_styling():
    st.markdown("""
        <style>
            /* Global Dark Background & Font */
            .stApp { 
                background-color: #0b1121; /* Deep slate dark */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #f8fafc;
            }
            
            /* Professional Dark Card Style for Dataframes and UI Blocks */
            div[data-testid="stDataFrame"], .stImage {
                border-radius: 12px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
                border: 1px solid #1e293b !important;
                background-color: #0f172a;
                padding: 10px;
            }
            
            /* Enhanced Headers for Dark Mode */
            h1, h2, h3 { 
                color: #f1f5f9 !important; 
                font-weight: 600;
            }
            
            /* Custom Interactive Primary Button */
            div.stButton > button[kind="primary"] {
                background-color: #3b82f6;
                color: white;
                border-radius: 8px;
                border: none;
                font-weight: bold;
                padding: 0.5rem 1rem;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            div.stButton > button[kind="primary"]:hover {
                background-color: #2563eb;
                box-shadow: 0 6px 12px rgba(59,130,246,0.3);
                transform: translateY(-2px);
            }
            
            /* Info & Success Boxes Styling (Darkened) */
            div[data-testid="stAlert"] {
                background-color: #1e293b !important;
                border-radius: 10px !important;
                border: 1px solid #334155 !important;
                color: #e2e8f0 !important;
            }
            
            /* Override for the Custom Metric Cards we built */
            .metric-card {
                background-color: #1e293b !important;
                border: 1px solid #334155 !important;
                color: #f8fafc !important;
            }
            .metric-card h4, .metric-card p {
                color: #cbd5e1 !important;
            }
            .metric-card h1 {
                color: #60a5fa !important; /* Bright blue for the numbers */
            }
        </style>
    """, unsafe_allow_html=True)
