import streamlit as st

# Page config
st.set_page_config(page_title="Wildlife Guard Login", page_icon="🦌", layout="centered")

# Background CSS
page_bg = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Center the login box */
.login-box {
    background: rgba(255, 255, 255, 0.9);
    padding: 40px;
    border-radius: 12px;
    max-width: 400px;
    margin: 100px auto;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    text-align: center;
}

/* Header styles */
h2 {
    color: #4CAF50; /* Green color for the header */
    font-family: 'Arial', sans-serif;
}

/* Input fields */
input[type="text"], input[type="password"] {
    background-color: #e8f5e9 !important; /* Light green background */
    color: #333 !important;
    border-radius: 8px;
    border: 1px solid #4CAF50 !important; /* Green border */
    padding: 10px;
    margin-bottom: 20px;
    width: 100%;
}

/* Login button */
.stButton>button {
    background-color: #4CAF50; /* Green color for the button */
    color: white;
    border-radius: 8px;
    padding: 10px;
    width: 100%;
    border: none;
    font-weight: bold;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #45a049; /* Darker green on hover */
}

/* Logo styles */
.logo {
    margin-bottom: 20px;
}

/* Footer styles */
.footer {
    margin-top: 20px;
    font-size: 12px;
    color: #555;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Login box UI
st.markdown('<div class="login-box">', unsafe_allow_html=True)
st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80, class_="logo")  # small logo
st.markdown("<h2>Wildlife Guard Login</h2>", unsafe_allow_html=True)

username = st.text_input("Username", placeholder="Enter username")
password = st.text_input("Password", type="password", placeholder="Enter password")

if st.button("Login"):
    if username == "admin" and password == "1234":
        st.success("✅ Login successful!")
        if st.button("🚨 Poacher"):
            st.switch_page("streamlit_app.py")  # go to your next file
    else:
        st.error("❌ Invalid username or password")

st.markdown('</div>', unsafe_allow_html=True)

# Footer with additional visuals
st.markdown('<div class="footer">© 2023 Wildlife Guard. All rights reserved.</div>', unsafe_allow_html=True)

# Add a decorative element (optional)
st.markdown('<hr style="border: 1px solid #4CAF50; margin-top: 40px; margin-bottom: 20px;">', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4CAF50;">Protecting Wildlife Together!</p>', unsafe_allow_html=True)
