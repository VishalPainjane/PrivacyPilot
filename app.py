import streamlit as st
from main2 import get_json
import os

st.set_page_config(
    page_title="Frosty Dark T&C Analyzer ❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary-color: #ff6b6b;
        --secondary-color: #ffe66d;
        --accent-color: #b8ff72;
        --background-color: #1a1a2e;
        --card-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        --text-color: #ffffff;
        --quote-bg: rgba(255, 255, 255, 0.1);
        --quote-highlight: rgba(255, 255, 255, 0.2);
        --progress-height: 50px;
        --progress-width: 500px;
        --progress-responsive-width: 90%;
    }

    body {
        background: linear-gradient(135deg, var(--background-color), #3a0088, #fdbb2d, #e7173f, #c400c6);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: var(--text-color);
        font-family: Arial, sans-serif;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .sidebar .sidebar-content {
        background: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    .sidebar a {
        color: var(--text-color) !important;
    }

    .css-1aumxhk, .css-1v0mbdj {
        text-align: center;
    }

    .container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 30px;
        padding: 40px 20px;
        box-sizing: border-box;
        justify-items: center;
    }

    .card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        width: 100%;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 30px;
    }

    .card-title {
        color: var(--secondary-color);
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px #000;
        font-weight: 700;
        font-size: 1.6rem;
    }

    .circular-score {
        position: relative;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center; 
        justify-content: center;
        color: #ffffff;
        font-weight: bold; 
        font-size: 1.8rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.5), inset 0 0 30px rgba(255,255,255,0.4);
        margin: 20px 0;
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    }

    .score-text {
        position: relative;
        z-index: 2;
    }

    .quotes-block {
        width: 100%;
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .quotes-title {
        color: var(--accent-color);
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }

    @import url('https://fonts.googleapis.com/css2?family=Caveat&display=swap');

    .quote-card {
        font-family: 'Caveat', cursive;
        background: linear-gradient(135deg, rgba(255,182,193,0.2), rgba(255,255,255,0.1));
        border-left: 4px solid #ff6b6b;
        color: #ffc0cb;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(255,20,147,0.4);
        text-align: left;
        word-wrap: break-word;
        white-space: pre-wrap;
        transition: background 0.3s ease, box-shadow 0.3s ease;
    }

    .quote-card a {
        color: #ff6b6b;
        text-decoration: underline;
    }

    .quote-card:hover {
        background: rgba(255,182,193,0.3);
        box-shadow: 0 4px 12px rgba(255,20,147,0.6);
    }

    .progress {
        width: var(--progress-width);
        height: var(--progress-height);
        margin: 2em auto;
        border: 1px solid #fff;
        padding: 12px 10px;
        box-shadow: 0 0 10px #aaa;
        background: black;
        border-radius: 10px;
        overflow: hidden;
    }

    .progress .bar {
        width: 0%;
        height: 100%;
        background: linear-gradient(gold, #c85, gold);
        box-shadow: 0 0 10px 0px orange;
        transition: width 0.3s ease;
    }

    @keyframes shine {
        0% { background-position: 0 0; }
        100% { background-position: 0 50px; }
    }

    @keyframes end {
        0%, 100% { box-shadow: 0 0 10px 0px orange; }
        50% { box-shadow: 0 0 15px 5px orange; }
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--secondary-color) !important;
        text-shadow: 1px 1px 2px #000;
    }

    .css-1ka2qzg { 
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 9999;
    }

    @media (max-width: 768px) {
        .container {
            grid-template-columns: repeat(auto-fill, minmax(90%, 1fr));
            padding: 20px;
        }
        .card {
            width: 90%;
        }
        .progress {
            width: var(--progress-responsive-width);
            height: 30px;
            margin: 1em auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_score_gradient(score):
    if score >= 4:
        return "linear-gradient(135deg, #00b09b, #96c93d)"
    elif score >= 2.5:
        return "linear-gradient(135deg, #f7971e, #ffd200)"
    else:
        return "linear-gradient(135deg, #ff416c, #ff4b2b)"

def display_scores(scores):
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    for section, data in scores.items():
        score = data.get('score', 0)
        percentage = (score / 5) * 100
        gradient = get_score_gradient(score)
        
        quotes_html = ""
        for quote in data.get("quotes", []):
            quotes_html += f"{quote}<br>"
        
        card_html = f"""
            <div class="card">
                <div class="card-title">{section.replace('_', ' ').title()}</div>
                <div class="circular-score" style="background: {gradient};">
                    <span class="score-text">{score}</span>
                </div>
                <div class="quotes-block">
                    <div class="quotes-title">Quotes:</div>
                        <div class="quote-card">
                        <em>"{quotes_html}"</em>
                        </div>
                </div>
            </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.sidebar.title("Frosty T&C Analyzer")
    st.sidebar.caption("Analyze Terms & Conditions with a winter twist!")
    st.sidebar.markdown("""---""")
    st.sidebar.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", use_container_width=True)
    
    with st.sidebar.expander("🔍 **How to Use This Analyzer**"):
        st.markdown("""
        1. **Enter the URL** of the Terms & Conditions you want to analyze.
        2. **Click 'Analyze'** to process the document.
        3. **View Scores & Quotes** to understand different sections.
        4. **Check Metadata** for overall risk assessment.
        """)

    with st.sidebar.expander("ℹ**About**"):
        st.markdown("""
        **Frosty Dark T&C Analyzer** is designed to provide a festive and engaging way to evaluate the complexities of Terms & Conditions documents. Leveraging advanced analytics and a winter-themed interface, it offers clear insights and highlights crucial information.
    
        *Made with ❤️.*
        """)

    st.title("Frosty Dark T&C Analyzer ")
    st.subheader("Embrace the Winter Sparkle and Evaluate T&C with Festive Cheer!")

    link_input = st.text_input("🔗 Enter the URL of the Terms & Conditions:", "")

    progress_placeholder = st.empty()

    if st.button("Analyze"):
        st.snow()
        with st.spinner("Carving insights from T&C ice blocks..."):
            try:
                progress_placeholder.markdown("""
                    <div class="progress">
                        <div class="bar" style="width: 0%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                def update_progress(value: int):
                    progress_placeholder.markdown(f"""
                        <div class="progress">
                            <div class="bar" style="width: {value}%;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                results = get_json(link_input, progress_callback=update_progress)

                progress_placeholder.markdown("""
                    <div class="progress">
                        <div class="bar" style="width: 100%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.success("Analysis complete! Enjoy your winter read")

                st.markdown("## Scores & Quotes")
                display_scores(results.get("scores", {}))

                st.markdown("## 🗂 Metadata")
                metadata = results.get("metadata", {})
                risk_percentage = metadata.get("risk_percentage", 0)

                st.markdown(f"""
                    <div class="progress">
                        <div class="bar" style="width: {risk_percentage}%;"></div>
                    </div>
                    <p style='text-align: center; color: var(--text-color);'><strong>Risk Percentage: {risk_percentage}%</strong></p>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Risk Level:** {metadata.get('risk_level', 'N/A')}")
                st.markdown(f"**GDPR Compliance Score:** {metadata.get('GDPR_compliance_score', 'N/A')}")

                with st.expander("📌 Additional Notes", expanded=True):
                    st.write(metadata.get("additional_notes", "No additional notes available."))

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--secondary-color);'>
            <p>🎄 Made with Frosty Love</p>
            <p>Stay frosty and keep your T&C under control!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
