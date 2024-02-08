import streamlit as st
import json
import os

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def main():
    st.title("JSON Editor")

    # File selection
    file_path = st.sidebar.file_uploader("Select JSON File", type=["json"])
    
    if file_path:
        st.sidebar.text(f"Selected File: {file_path.name}")
        data = read_json(file_path.name)

        # Display current JSON content
        st.subheader("Current JSON Content:")
        st.code(json.dumps(data, indent=2), language='json')

        # Key selection dropdowns
        st.subheader("Select Keys to Edit:")
        selected_ip_key = st.selectbox("Select Destination Node IP Key", options=list(data.keys()), index=0)
        selected_port_key = st.selectbox("Select Destination Node Port Key", options=list(data.keys()), index=1)

        # JSON Editor
        st.subheader("Edit JSON:")
        json_editor = st.text_area("Edit JSON", value=json.dumps(data, indent=2), height=300)
        try:
            edited_data = json.loads(json_editor)
        except json.JSONDecodeError:
            st.error("Invalid JSON syntax. Please correct the JSON data.")
            return

        # Save Button
        if st.button("Save Changes"):
            data[selected_ip_key] = edited_data[selected_ip_key]
            data[selected_port_key] = edited_data[selected_port_key]
            write_json(file_path.name, data)
            st.success("Changes saved successfully!")

if __name__ == "__main__":
    main()