import streamlit as st

# Create a title for the app
st.title("Mahithi : A Government Cognitive Assistant")

# Create a dropdown menu for navigation
nav = st.selectbox("Navigate to", ["Home", "About", "Agriculture", "Education", "Personal", "Marriage"])

# Create sections for each navigation option
if nav == "Home":
    st.header("Welcome to Mahithi!")
    st.write("Explore various government schemes and services that cater to your needs.")

elif nav == "About":
    st.header("About Mahithi")
    st.write("""
        Mahithi is a government cognitive assistant designed to help citizens easily find and access various government schemes. 
        Whether you are looking for support in agriculture, education, personal services, or marriage-related information, 
        Mahithi aims to simplify the process of discovering relevant schemes and services tailored to your needs. 
        With Mahithi, solving problems related to finding the right schemes has never been easier.
    """)

elif nav == "Agriculture":
    st.header("Agriculture Information")
    with st.expander("Overview of Agricultural Schemes"):
        st.write("""
            This section provides a comprehensive list of government schemes available for farmers, 
            covering subsidies, loans, insurance, and support for sustainable farming practices.
        """)
    with st.expander("Subsidies and Financial Support"):
        st.write("""
            Explore schemes that offer financial support for purchasing seeds, fertilizers, and farming equipment. 
            Find out how to apply and what eligibility criteria need to be met.
        """)
    with st.expander("Sustainable Farming Initiatives"):
        st.write("""
            Learn about government initiatives that promote sustainable and organic farming, 
            including training programs, grants, and certification processes.
        """)

elif nav == "Education":
    st.header("Education Information")
    with st.expander("Educational Grants and Scholarships"):
        st.write("""
            Discover a variety of scholarships and grants available for students at different levels, 
            from primary education to higher studies, including specialized programs for underprivileged communities.
        """)
    with st.expander("Skill Development Programs"):
        st.write("""
            Access information about skill development programs designed to enhance employability 
            and vocational skills, focusing on both rural and urban populations.
        """)
    with st.expander("School and College Infrastructure Support"):
        st.write("""
            Explore schemes aimed at improving educational infrastructure, 
            including funding for new schools, digital classrooms, and educational resources.
        """)

elif nav == "Personal":
    st.header("Personal Information")
    with st.expander("Identity and Documentation"):
        st.write("""
            Learn about the process of obtaining essential documents like Aadhar cards, 
            voter IDs, and birth certificates, along with government services that assist with these processes.
        """)
    with st.expander("Healthcare and Insurance"):
        st.write("""
            Find out about government healthcare schemes and insurance policies that provide coverage for medical expenses, 
            including specific programs for senior citizens, women, and children.
        """)
    with st.expander("Social Security and Welfare"):
        st.write("""
            Understand the various social security and welfare programs available for different demographics, 
            including pensions, disability benefits, and unemployment support.
        """)

elif nav == "Marriage":
    st.header("Marriage Information")
    with st.expander("Marriage Registration"):
        st.write("""
            Get details on how to register a marriage, including the necessary documents, 
            the application process, and where to apply.
        """)
    with st.expander("Government Marriage Schemes"):
        st.write("""
            Explore government schemes that provide financial assistance for marriages, 
            especially for those belonging to economically weaker sections and marginalized communities.
        """)
    with st.expander("Family and Child Welfare"):
        st.write("""
            Learn about government programs aimed at supporting families, 
            including maternity benefits, child welfare schemes, and family counseling services.
        """)

