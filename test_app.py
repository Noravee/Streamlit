import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


with st.form('test'):
    A = np.random.randint(0, 5, size=(10,1))
    B = np.random.randint(0, 5, size=(10,1))
    C = np.random.randint(0, 5, size=(10,1))
    D = np.random.randn(10,1)
    E = np.random.randn(10,1)
    df = pd.DataFrame(np.concatenate([A, B, C, D, E], axis=1), columns=['A', 'B', 'C', 'D', 'E'])
    df.A = df.A.astype('int')
    df.B = df.B.astype('int')
    df.C = df.C.astype('int')
    generated = st.form_submit_button('Generate DF')

def reset(df: pd.DataFrame):
    selectors_filters = [col+'selector' for col in df.columns]
    selectors_filters.append('filters')
    for key in selectors_filters:
        if key in st.session_state:
            del st.session_state[key]

if generated:
    st.session_state.df = df
    reset(st.session_state.df)

if 'df' in st.session_state:

    # Initialize session state for filters if not already set
    if 'filters' not in st.session_state:
        st.session_state.filters = {
            i: (st.session_state.df[i].min(), st.session_state.df[i].max()) 
            if st.session_state.df[i].dtype == 'float' 
            else sorted(st.session_state.df[i].unique().tolist()) 
            for i in st.session_state.df.columns
            }

    # Function to filter the dataframe based on current session state filters
    def filter_df():
        filtered_df = st.session_state.df.copy()
        for col in filtered_df.columns:
            if st.session_state.filters[col]:
                if filtered_df[col].dtype == 'float':
                    filtered_df = filtered_df[
                        (filtered_df[col] >= st.session_state.filters[col][0]) & 
                        (filtered_df[col] <= st.session_state.filters[col][1])
                        ]
                else:
                    filtered_df = filtered_df[filtered_df[col].isin(st.session_state.filters[col])]
        return filtered_df

    filters_changed = False

    # Sidebar for user inputs
    with st.sidebar:
        st.write('Filters')

        def generate_filter(col: str, dtype: str):

            filters_changed = False
            # Filter based on current selection
            filtered_df = filter_df()
            if dtype == 'float':
                # Dynamically update range based on filtered data
                min_default, max_default = filtered_df[col].min(), filtered_df[col].max()
                # Slider for float values
                selected = st.slider(
                    'Select range of ' + col, 
                    min_value = st.session_state.df[col].min(), 
                    max_value = st.session_state.df[col].max(), 
                    value = (min_default, max_default),
                    key = col + '_selector'
                    )
            else:
                # Update options for A based on current filtered data
                available = sorted(filtered_df[col].unique().tolist())
                # Multi-select for category
                selected = st.multiselect(
                    'Select classes of ' + col, 
                    options = sorted(st.session_state.df[col].unique().tolist()),  # Always show full range of options
                    default = available,
                    key = col + '_selector'
                    )
            # Check if filter changed
            if selected != st.session_state.filters[col]:
                st.session_state.filters[col] = selected  # Update session state based on selection
                filters_changed = True

            return filters_changed

        changed = []
        for col in st.session_state.df.columns:
            changed.append(generate_filter(col, st.session_state.df[col].dtype))
        filters_changed = any(changed)
        if st.button('Reset Filters', type='primary'):
            reset(st.session_state.df)
            st.rerun()

        # Update filter for column A (classes)
        # filtered_df = filter_df()  # Filter based on current selection
        # available_A = sorted(filtered_df['A'].unique().tolist())  # Update options for A based on current filtered data
    
        # # Multi-select for A
        # selected_A = st.multiselect(
        #     'Select A classes', 
        #     options=sorted(st.session_state.df['A'].unique().tolist()),  # Always show full range of A options
        #     default=available_A,
        #     key='A_selector'
        # )
        # # Check if A filter changed
        # if selected_A != st.session_state.filters['A']:
        #     st.session_state.filters['A'] = selected_A  # Update session state based on selection
        #     filters_changed = True

        # # Update filter for column B (numeric range)
        # filtered_df = filter_df()  # Re-filter based on updated A selection
        # min_B, max_B = filtered_df['B'].min(), filtered_df['B'].max()  # Dynamically update B's range based on filtered data
    
        # # Slider for B range
        # selected_B_range = st.slider(
        #     'Select B range', 
        #     min_value=float(st.session_state.df['B'].min()), 
        #     max_value=float(st.session_state.df['B'].max()), 
        #     value=(float(min_B), float(max_B)),
        #     key='B_selector'
        #     )
        # # Check if B filter changed
        # if selected_B_range != st.session_state.filters['B']:
        #     st.session_state.filters['B'] = selected_B_range  # Update session state based on selection
        #     filters_changed = True

        # # Update filter for column C (classes)
        # filtered_df = filter_df()  # Filter based on current selection
        # available_C = sorted(filtered_df['C'].unique().tolist())  # Update options for A based on current filtered data
    
        # # Multi-select for C
        # selected_C = st.multiselect(
        #     'Select C classes', 
        #     options=sorted(st.session_state.df['C'].unique().tolist()),  # Always show full range of A options
        #     default=available_C,
        #     key='C_selector'
        # )
        # # Check if C filter changed
        # if selected_C != st.session_state.filters['C']:
        #     st.session_state.filters['C'] = selected_C  # Update session state based on selection
        #     filters_changed = True

    # Trigger rerun if filters changed
    if filters_changed:
        st.rerun()

    # Display filtered DataFrame
    filtered_df = filter_df()
    st.dataframe(filtered_df, hide_index=True, width=1000)

    # with st.expander("See plot"):
    #     fig = px.scatter(df, x='B', y='A')
    #     st.plotly_chart(fig)