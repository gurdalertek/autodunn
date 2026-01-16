# autodunn
Online App for the Automated Testing of Group Means and Posterior Analysis

The *AutoDunn app* fills a simple yet important gap for statisticians and data analysts. Many times, it is necessary to conduct a non-parametric comparison of means across groups, first with Kruskal-Wallis test and then with pairwise Dunn tests (with Bonferroni or other corrections) across all pairs. However, the current practice among many analysts is to conduct this analysis manually using a software package such as SPSS or writing custom code. 

The AutoDunn software provides an easy way to conduct non-parametric comparison of means _automatically_, saving considerable amount of time to analysts. Furthermore, the code is open source, enabling transperancy and reproducibility. Most importantly, AutoDunn software visualizes the results of all conducted statistical analysis using the most effective visualization techniques, including network-based visualization of statistical dominances.

AutoDunn can speed data analysis projects significantly. The big limitation is that the software accepts only one (categorical) factor and one (numerical) response. So, if there are multiple factors or responses, each data pair (factor response pair) has to be prepared separately and the software needs to be used multiple times, once for each pair of interest.


Live running app under Streamlit (may need to wake up):

https://autodunn-7thsq9y48yrjyfdr4ezr6n.streamlit.app


Documentation:

https://autodunn.com

https://ertekprojects.com/autodunn

Technology Stack: Python, Python packages, Streamlit, ChatGPT 5 for vibe coding
Credits: The original R code that conducted the analysis (without a GUI or visualization) was written by Dr. Gul Tokdemir. 

