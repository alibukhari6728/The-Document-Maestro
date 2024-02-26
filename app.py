import tempfile
import streamlit as st

from pdf_analysis import pdf_analyzer  

# Define custom CSS to change button color
button_style = """
<style>
.stButton>button {
    background-color: blue;
    color: white;
}
</style>
"""

# Apply the custom CSS style
st.markdown(button_style, unsafe_allow_html=True)


def st_app( ):
    st.title("The Document Maestro")

    # UI Space to upload PDF
    pdf_file = st.file_uploader("Summon a PDF chronicle", type=["pdf"])

    # UI Space to add query 
    input_string = st.text_input("Pose your riddle")

    # Button to trigger processing
    if st.button("Commence the document odyssey"):

        if pdf_file is not None and input_string:

            # Create a temporary file (deleted upon finishing request)
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

                try:
                    # API call with the temporary file path
                    output_string, output_image = pdf_analyzer(tmp_file_path, input_string)

                except Exception as e:
                    st.error(f"An unforeseen twist in our analytical journey: {e}")
                    return                

                # Display the results
                if output_string is not None:
                    st.write(output_string)
                if output_image is not None:
                    st.image(output_image, caption="A vision plucked from the wisdom vault")
        else:
            st.error("A mysterious interruption in transmission. Please refresh and resume")

if __name__ == "__main__":
    st_app()