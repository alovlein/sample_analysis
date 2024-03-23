import streamlit as st
import analysis_functions as af

st.title('Alec Lovlein sample analysis')

populations = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']

st.write('Relative frequencies of the populations')
st.dataframe(af.relative_frequency(populations), hide_index=True)

st.write('Population comparisons')
population = st.selectbox('Select a population to analyze', populations)
st.plotly_chart(af.response_comparison(population, populations))

st.write(
    '''
    The populations are compared with Welch's t-test. It is a two-sample location test which is used to test the 
    (null) hypothesis that two populations have equal means. In this case, a low p-value indicates a possibility that 
    the null hypothesis can be rejected, and the two populations might be different. In our case, we have extremely 
    low sample sizes so any concrete conclusions would require more thorough testing.
    '''
)
